import os
import sqlite3
import pandas as pd
from openai import OpenAI
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIInventoryChatbot:
    def __init__(self, db_path='../db/diagnostics.db'):
        """Initialize the OpenAI-powered inventory chatbot."""
        self.db_path = db_path
        
        # Check if database exists
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Get available inventory tables and their schemas
        self.inventory_tables = self._get_inventory_tables_info()
        
        logger.info(f"OpenAI Inventory Chatbot initialized with {len(self.inventory_tables)} tables")
    
    def _get_inventory_tables_info(self):
        """Get information about available inventory tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables that are not patient_data or patient tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # Filter out patient tables
            inventory_tables = {}
            for table_name in all_tables:
                if not table_name.startswith('table_20') and table_name != 'patient_data':
                    # Get column information for each table
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns_info = cursor.fetchall()
                    columns = [col[1] for col in columns_info]  # col[1] is column name
                    
                    # Get sample data to understand table content
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_data = cursor.fetchall()
                    
                    inventory_tables[table_name] = {
                        'columns': columns,
                        'sample_data': sample_data
                    }
            
            conn.close()
            return inventory_tables
            
        except Exception as e:
            logger.error(f"Error getting inventory tables info: {e}")
            return {}
    
    def identify_relevant_table(self, question):
        """
        Use OpenAI to identify which inventory table is most relevant to the question.
        Enhanced with better table categorization and keyword matching.
        """
        try:
            # First, categorize tables by their likely purpose
            table_categories = {
                'stock_tables': [],
                'purchase_tables': [],
                'challan_tables': [],
                'order_tables': [],
                'other_tables': []
            }
            
            for table_name in self.inventory_tables.keys():
                table_lower = table_name.lower()
                if 'stock' in table_lower or 'item' in table_lower:
                    table_categories['stock_tables'].append(table_name)
                elif 'purchase' in table_lower and 'register' in table_lower:
                    table_categories['purchase_tables'].append(table_name)
                elif 'challan' in table_lower or 'delivery' in table_lower:
                    table_categories['challan_tables'].append(table_name)
                elif 'order' in table_lower or 'po' in table_lower:
                    table_categories['order_tables'].append(table_name)
                else:
                    table_categories['other_tables'].append(table_name)
            
            # Simple keyword-based pre-filtering
            question_lower = question.lower()
            
            # Direct keyword matching
            if any(keyword in question_lower for keyword in ['stock', 'inventory', 'available', 'closing', 'balance']):
                if table_categories['stock_tables']:
                    logger.info(f"Using stock table for stock-related question: {table_categories['stock_tables'][0]}")
                    return table_categories['stock_tables'][0]
            
            elif any(keyword in question_lower for keyword in ['purchase', 'bought', 'procurement', 'vendor payment']):
                if table_categories['purchase_tables']:
                    logger.info(f"Using purchase table for purchase-related question: {table_categories['purchase_tables'][0]}")
                    return table_categories['purchase_tables'][0]
            
            elif any(keyword in question_lower for keyword in ['order', 'po', 'requisition', 'ordered']):
                if table_categories['order_tables']:
                    logger.info(f"Using order table for order-related question: {table_categories['order_tables'][0]}")
                    return table_categories['order_tables'][0]
            
            elif any(keyword in question_lower for keyword in ['challan', 'delivery', 'received', 'goods']):
                if table_categories['challan_tables']:
                    logger.info(f"Using challan table for delivery-related question: {table_categories['challan_tables'][0]}")
                    return table_categories['challan_tables'][0]
            
            # If no direct match, use OpenAI with better context
            tables_info = ""
            for category, tables in table_categories.items():
                if tables:
                    tables_info += f"\n{category.upper()}:\n"
                    for table_name in tables:
                        info = self.inventory_tables[table_name]
                        tables_info += f"  - {table_name}: {info['columns'][:5]}...\n"  # Show first 5 columns
            
            prompt = f"""
You are analyzing an inventory question to select the best database table.

AVAILABLE TABLE CATEGORIES:
{tables_info}

QUESTION: "{question}"

SELECTION RULES:
1. STOCK_TABLES: For current inventory, item availability, stock levels
2. PURCHASE_TABLES: For purchase history, vendor payments, procurement data
3. ORDER_TABLES: For purchase orders, requisitions, planned purchases
4. CHALLAN_TABLES: For goods received, deliveries, incoming stock

EXAMPLES:
- "What items are low in stock?" → Use STOCK table
- "Show me recent purchases" → Use PURCHASE table  
- "Which orders are pending?" → Use ORDER table
- "What did we receive today?" → Use CHALLAN table

Return ONLY the exact table name from the list above that best matches the question:
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a database expert. Analyze the question and return ONLY the most relevant table name."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            identified_table = response.choices[0].message.content.strip()
            
            # Validate that the identified table exists
            if identified_table in self.inventory_tables:
                logger.info(f"OpenAI identified table: {identified_table} for question: {question}")
                return identified_table
            else:
                # Smart fallback based on question type
                if any(keyword in question_lower for keyword in ['stock', 'inventory']):
                    fallback = table_categories['stock_tables'][0] if table_categories['stock_tables'] else None
                elif any(keyword in question_lower for keyword in ['purchase', 'buy']):
                    fallback = table_categories['purchase_tables'][0] if table_categories['purchase_tables'] else None
                else:
                    fallback = list(self.inventory_tables.keys())[0] if self.inventory_tables else None
                
                logger.warning(f"OpenAI returned invalid table '{identified_table}'. Using fallback: {fallback}")
                return fallback
                
        except Exception as e:
            logger.error(f"Error identifying table: {e}")
            # Return most appropriate fallback based on question
            question_lower = question.lower()
            for category, tables in table_categories.items():
                if tables:
                    if 'stock' in question_lower and category == 'stock_tables':
                        return tables[0]
                    elif 'purchase' in question_lower and category == 'purchase_tables':
                        return tables[0]
            
            # Final fallback
            return list(self.inventory_tables.keys())[0] if self.inventory_tables else None

    def debug_available_tables(self):
        """Debug function to see all available tables and their purposes."""
        print("=== AVAILABLE INVENTORY TABLES ===")
        for table_name, info in self.inventory_tables.items():
            print(f"\nTable: {table_name}")
            print(f"Columns: {info['columns']}")
            print(f"Sample data: {info['sample_data'][:1] if info['sample_data'] else 'No data'}")
            
            # Guess table purpose
            table_lower = table_name.lower()
            if 'stock' in table_lower:
                purpose = "📦 Stock/Inventory data"
            elif 'purchase' in table_lower:
                purpose = "💰 Purchase transactions"
            elif 'challan' in table_lower:
                purpose = "📋 Delivery/Receipt data"
            elif 'order' in table_lower:
                purpose = "📝 Purchase orders"
            else:
                purpose = "❓ Other inventory data"
            
            print(f"Likely purpose: {purpose}")
            print("-" * 50)
    
    def generate_sql_for_table(self, question, table_name):
        """
        Generate SQL query for specific inventory table using OpenAI.
        
        Args:
            question (str): User question
            table_name (str): Target table name
            
        Returns:
            str: Generated SQL query
        """
        try:
            table_info = self.inventory_tables.get(table_name, {})
            columns = table_info.get('columns', [])
            sample_data = table_info.get('sample_data', [])
            
            prompt = f"""
You are an expert SQL query generator for inventory management systems.

TARGET TABLE: {table_name}
COLUMNS: {columns}
SAMPLE DATA: {sample_data[:3]}

USER QUESTION: "{question}"

IMPORTANT SQL GENERATION RULES:
1. Table name is ALWAYS '{table_name}'
2. Use EXACT column names from the list above
3. Handle various data formats (dates, text, numbers)
4. Use LIKE operator for text searches (e.g., column LIKE '%search_term%')
5. Use proper SQL aggregate functions: COUNT(), SUM(), AVG(), MAX(), MIN()
6. Always include ORDER BY for better results
7. Use LIMIT when showing top results
8. For amount/quantity columns, use SUM() for totals
9. Handle NULL values appropriately

EXAMPLES:
Question: "What items are running low in stock?"
SQL: SELECT item, closing_stock FROM {table_name} WHERE closing_stock < 10 ORDER BY closing_stock ASC

Question: "Show me recent orders"
SQL: SELECT * FROM {table_name} ORDER BY date DESC LIMIT 10

Question: "Total purchase amount from supplier ABC"
SQL: SELECT SUM(total_amt) as total_amount FROM {table_name} WHERE party LIKE '%ABC%'

Now generate a SQL query for the given question.
Return ONLY the SQL query, no explanations:
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SQL generator for inventory databases. Generate clean, accurate SQL queries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up response
            if sql_query.startswith('```sql'):
                sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            elif sql_query.startswith('```'):
                sql_query = sql_query.replace('```', '').strip()
            
            logger.info(f"Generated SQL for {table_name}: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return f"SELECT * FROM {table_name} LIMIT 10"
    
    def execute_sql_query(self, sql_query):
        """Execute SQL query using sqlite3."""
        try:
            # Basic validation
            if not self._is_safe_query(sql_query):
                raise ValueError("Query failed safety validation")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # Get results
            column_names = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to DataFrame
            result_df = pd.DataFrame(rows, columns=column_names)
            logger.info(f"Query executed successfully, returned {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            return pd.DataFrame()
    
    def _is_safe_query(self, sql_query):
        """Basic SQL safety validation."""
        query_lower = sql_query.lower().strip()
        
        # Check for dangerous operations
        dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'alter', 'create', 'truncate']
        
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                return False
        
        # Must be a SELECT statement
        if not query_lower.startswith('select'):
            return False
        
        return True
    
    def explain_inventory_results(self, question, table_name, sql_query, result_df):
        """
        Generate human-friendly explanation of inventory results using OpenAI.
        
        Args:
            question (str): Original question
            table_name (str): Table that was queried
            sql_query (str): SQL query executed
            result_df (pandas.DataFrame): Query results
            
        Returns:
            str: Human-friendly explanation
        """
        try:
            # Prepare result summary
            if result_df.empty:
                data_summary = "No data found"
            else:
                data_summary = f"Number of records: {len(result_df)}\n"
                data_summary += f"Columns: {list(result_df.columns)}\n"
                
                if len(result_df) <= 5:
                    data_summary += f"All results: {result_df.to_dict('records')}\n"
                else:
                    data_summary += f"Sample results: {result_df.head(5).to_dict('records')}\n"
                
                # Add summaries for numeric columns
                numeric_cols = result_df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if 'amt' in col.lower() or 'qty' in col.lower() or 'stock' in col.lower():
                        data_summary += f"\n{col} - Total: {result_df[col].sum():.2f}, Average: {result_df[col].mean():.2f}"
            
            prompt = f"""
You are explaining inventory management query results to warehouse managers and procurement staff.

CONTEXT:
Original Question: "{question}"
Table Queried: {table_name}
SQL Query: {sql_query}
Results: {data_summary}

TASK: Convert these technical results into clear, business-friendly language.

EXPLANATION RULES:
1. Use business terminology (not technical database terms)
2. Focus on actionable insights
3. Format amounts as ₹X,XXX (Indian Rupees)
4. Highlight important patterns or issues
5. For stock data: mention if items are low/high/out of stock
6. For purchase data: mention suppliers, amounts, dates
7. For order data: mention quantities, parties, totals
8. Be concise but informative
9. If no data: suggest alternative searches

EXAMPLES:
- Stock query → "You have 15 items running low on stock. Item XYZ has only 2 units left."
- Purchase query → "Total purchases from ABC Supplier: ₹45,000 across 8 orders."
- Order query → "Latest order was from DEF Company for ₹12,500 on 15th Jan."

Provide a clear, helpful business explanation:
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a business analyst explaining inventory data to managers in simple, actionable language."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            if result_df.empty:
                return "No inventory data found matching your query. Please try rephrasing or check if the data exists."
            else:
                return f"Found {len(result_df)} inventory records matching your query."
    
    def ask_inventory_question(self, question):
        """
        Main function to process inventory questions.
        
        Args:
            question (str): Natural language question about inventory
            
        Returns:
            dict: Complete response with table, SQL, results, and explanation
        """
        try:
            logger.info(f"Processing inventory question: {question}")
            
            # Step 1: Identify relevant table
            table_name = self.identify_relevant_table(question)
            
            if not table_name:
                return {
                    'question': question,
                    'identified_table': None,
                    'sql_query': None,
                    'result_df': pd.DataFrame(),
                    'explanation': "Could not identify relevant inventory table for your question.",
                    'success': False
                }
            
            # Step 2: Generate SQL query
            sql_query = self.generate_sql_for_table(question, table_name)
            
            # Step 3: Execute query
            result_df = self.execute_sql_query(sql_query)
            
            # Step 4: Generate explanation
            explanation = self.explain_inventory_results(question, table_name, sql_query, result_df)
            
            return {
                'question': question,
                'identified_table': table_name,
                'sql_query': sql_query,
                'result_df': result_df,
                'explanation': explanation,
                'success': True,
                'row_count': len(result_df)
            }
            
        except Exception as e:
            logger.error(f"Error processing inventory question: {e}")
            return {
                'question': question,
                'identified_table': None,
                'sql_query': None,
                'result_df': pd.DataFrame(),
                'explanation': f"Sorry, I couldn't process your inventory question. Error: {str(e)}",
                'success': False,
                'row_count': 0
            }

# Example usage and testing
if __name__ == "__main__":
    # Initialize inventory chatbot
    chatbot = OpenAIInventoryChatbot()
    
    print("=== INVENTORY CHATBOT DEBUG ===\n")
    
    # First, show all available tables
    chatbot.debug_available_tables()
    
    # Test table identification for different question types
    test_questions = [
        "tell me about the stock value of 'TEST OBJECT'"       # Should use stock table
        # "Show me recent purchase orders",              # Should use order table
        # "Which suppliers have we bought from?",        # Should use purchase table
        # "What goods did we receive today?",            # Should use challan table
        # "List all items with zero stock",              # Should use stock table
        # "Total purchase amount this month",            # Should use purchase table
        # "Show me pending orders"                       # Should use order table
    ]
    
    print("\n=== TABLE IDENTIFICATION TEST ===")
    for question in test_questions:
        identified_table = chatbot.identify_relevant_table(question)
        print(f"Q: {question}")
        print(f"Selected Table: {identified_table}")
        print("-" * 60)
    
    print("\n=== FULL QUESTION TEST ===")
    # Test first question completely
    result = chatbot.ask_inventory_question("What items are running low in stock?")
    print(f"Table: {result['identified_table']}")
    print(f"SQL: {result['sql_query']}")
    print(f"Success: {result['success']}")
    print(f"Answer: {result['explanation']}")