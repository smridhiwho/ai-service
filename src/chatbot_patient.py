import os
import openai
from openai import OpenAI
import pandas as pd
import sqlite3
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIPatientChatbot:
    def __init__(self, db_path='../db/diagnostics.db'):
        """Initialize the OpenAI-powered patient chatbot."""
        self.db_path = db_path
        
        # Check if database exists
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Define the database schema for context
        self.schema_info = """
        Table: patient_data
        Columns:
        - sr_no (INTEGER): Serial number
        - case_no (TEXT): Case number  
        - opd_dt (TEXT): OPD date (format varies: YYYY-MM-DD, DD/MM/YYYY, etc.)
        - patient (TEXT): Patient name
        - doctor (TEXT): Doctor name
        - rpt_name (TEXT): Report name/test type
        - city (TEXT): Patient's city
        - total_amt (REAL): Total amount
        - discount (REAL): Discount amount
        - net_amt (REAL): Net amount after discount
        - paid_amt (REAL): Amount actually paid
        """
        
        logger.info(f"OpenAI Patient Chatbot initialized with database: {db_path}")
    
    def execute_sql_query(self, sql_query):
        """
        Execute the SQL query using sqlite3.
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            pandas.DataFrame: Query results
        """
        try:
            # Validate query first
            if not self.validate_sql_query(sql_query):
                raise ValueError("SQL query failed safety validation")
            
            # Execute query using sqlite3
            conn = sqlite3.connect(self.db_path)
            
            # Execute query and get results
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            # Fetch all results
            rows = cursor.fetchall()
            
            # Close connection
            conn.close()
            
            # Convert to DataFrame
            result_df = pd.DataFrame(rows, columns=column_names)
            
            logger.info(f"Query executed successfully, returned {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return pd.DataFrame()
    
    def create_patient_data_table_if_missing(self):
        """Create patient_data table by combining yearly tables if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if patient_data table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patient_data'")
            if cursor.fetchone():
                conn.close()
                logger.info("patient_data table already exists")
                return True
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # Find patient tables (table_YYYY format)
            patient_tables = [t for t in all_tables if t.startswith('table_20')]
            
            if not patient_tables:
                conn.close()
                logger.error("No patient tables found to combine")
                return False
            
            logger.info(f"Found patient tables to combine: {patient_tables}")
            
            # Create patient_data table by combining all yearly tables
            first_table = True
            for table_name in patient_tables:
                if first_table:
                    # Create table from first yearly table
                    cursor.execute(f"CREATE TABLE patient_data AS SELECT * FROM {table_name}")
                    first_table = False
                    logger.info(f"Created patient_data table from {table_name}")
                else:
                    # Insert data from subsequent tables
                    cursor.execute(f"INSERT INTO patient_data SELECT * FROM {table_name}")
                    logger.info(f"Added data from {table_name}")
            
            # Commit changes
            conn.commit()
            
            # Verify the new table
            cursor.execute("SELECT COUNT(*) FROM patient_data")
            row_count = cursor.fetchone()[0]
            
            conn.close()
            
            logger.info(f"✅ Successfully created patient_data table with {row_count} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error creating patient_data table: {e}")
            return False
    
    def create_sql_prompt(self, question):
        """Create a detailed prompt for OpenAI to generate SQL."""
        prompt = f"""
You are an expert SQL query generator for a medical diagnostics center database.

DATABASE SCHEMA:
{self.schema_info}

IMPORTANT SQL GENERATION RULES:
1. Table name is ALWAYS 'patient_data'
2. Use EXACT column names as provided above
3. For date queries, use LIKE operator since date formats vary (e.g., opd_dt LIKE '%2023%' for year 2023)
4. For name searches, use LIKE with wildcards (e.g., doctor LIKE '%Smith%')
5. Use proper SQL aggregate functions: COUNT(), SUM(), AVG(), MAX(), MIN()
6. Always include ORDER BY for better results presentation
7. Use LIMIT when showing top results
8. For revenue/amount queries, use total_amt unless specifically asked for net_amt or paid_amt

EXAMPLES:
Question: "How many patients did Dr. Smith see in 2023?"
SQL: SELECT COUNT(*) as patient_count FROM patient_data WHERE doctor LIKE '%Smith%' AND opd_dt LIKE '%2023%'

Question: "What's the total revenue by each doctor?"
SQL: SELECT doctor, SUM(total_amt) as total_revenue FROM patient_data GROUP BY doctor ORDER BY total_revenue DESC

Question: "Show me top 5 patients by amount paid"
SQL: SELECT patient, doctor, total_amt, opd_dt FROM patient_data ORDER BY total_amt DESC LIMIT 5

Question: "Average consultation amount in Mumbai"
SQL: SELECT AVG(total_amt) as avg_amount FROM patient_data WHERE city LIKE '%Mumbai%'

Now generate a SQL query for this question:
Question: "{question}"

Return ONLY the SQL query, no explanations or additional text.
"""
        return prompt
    
    def generate_sql_query(self, question):
        """
        Use OpenAI to generate SQL query from natural language question.
        
        Args:
            question (str): Natural language question about patient data
            
        Returns:
            str: Generated SQL query
        """
        try:
            # Create the prompt
            prompt = self.create_sql_prompt(question)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" for faster/cheaper
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert SQL generator for medical database queries. Generate clean, accurate SQL queries based on the provided schema."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500
            )
            
            # Extract SQL query from response
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response (remove any markdown formatting)
            if sql_query.startswith('```sql'):
                sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            elif sql_query.startswith('```'):
                sql_query = sql_query.replace('```', '').strip()
            
            logger.info(f"Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            # Fallback to a simple query
            return "SELECT * FROM patient_data LIMIT 10"
    
    def validate_sql_query(self, sql_query):
        """
        Validate the generated SQL query for safety.
        
        Args:
            sql_query (str): SQL query to validate
            
        Returns:
            bool: True if query is safe, False otherwise
        """
        # Convert to lowercase for checking
        query_lower = sql_query.lower().strip()
        
        # Check for dangerous operations
        dangerous_keywords = [
            'drop', 'delete', 'insert', 'update', 'alter', 'create', 
            'truncate', 'exec', 'execute', 'sp_', 'xp_'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                logger.warning(f"Dangerous keyword detected: {keyword}")
                return False
        
        # Must contain 'select' and 'patient_data'
        if 'select' not in query_lower or 'patient_data' not in query_lower:
            logger.warning("Query must be a SELECT statement from patient_data table")
            return False
        
        return True
    
    def execute_sql_query(self, sql_query):
        """
        Execute the SQL query using sqlite3.
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            pandas.DataFrame: Query results
        """
        try:
            # Validate query first
            if not self.validate_sql_query(sql_query):
                raise ValueError("SQL query failed safety validation")
            
            # Execute query using sqlite3
            conn = sqlite3.connect(self.db_path)
            
            # Execute query and get results
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            # Fetch all results
            rows = cursor.fetchall()
            
            # Close connection
            conn.close()
            
            # Convert to DataFrame
            result_df = pd.DataFrame(rows, columns=column_names)
            
            logger.info(f"Query executed successfully, returned {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return pd.DataFrame()
    
    def explain_results_with_openai(self, question, sql_query, result_df):
        """
        Use OpenAI to generate human-friendly explanation of results.
        
        Args:
            question (str): Original question
            sql_query (str): SQL query that was executed
            result_df (pandas.DataFrame): Query results
            
        Returns:
            str: Human-friendly explanation
        """
        try:
            # Prepare comprehensive data summary for OpenAI
            if result_df.empty:
                data_summary = "No data found"
            else:
                data_summary = f"Number of rows: {len(result_df)}\n"
                data_summary += f"Columns: {list(result_df.columns)}\n"
                
                # Show sample data based on result type
                if len(result_df) <= 5:
                    # Show all rows if few results
                    data_summary += f"All results: {result_df.to_dict('records')}\n"
                else:
                    # Show first few rows for large results
                    data_summary += f"Sample results: {result_df.head(5).to_dict('records')}\n"
                
                # Add statistical summaries for numeric columns
                numeric_columns = result_df.select_dtypes(include=['number']).columns
                for col in numeric_columns:
                    if 'amt' in col.lower() or 'count' in col.lower() or 'revenue' in col.lower():
                        data_summary += f"\n{col} - Total: {result_df[col].sum():.2f}, Average: {result_df[col].mean():.2f}"
                        if len(result_df) > 1:
                            data_summary += f", Max: {result_df[col].max():.2f}, Min: {result_df[col].min():.2f}"
                
                # Add unique value counts for categorical columns
                for col in ['doctor', 'city', 'patient']:
                    if col in result_df.columns:
                        unique_count = result_df[col].nunique()
                        if unique_count <= 10:
                            data_summary += f"\nUnique {col}s: {list(result_df[col].unique())}"
                        else:
                            data_summary += f"\nNumber of unique {col}s: {unique_count}"
            
            # Create enhanced explanation prompt
            explanation_prompt = f"""
You are explaining medical database query results to healthcare staff. Convert the technical SQL results into natural, conversational language.

CONTEXT:
Original Question: "{question}"
SQL Query: {sql_query}
Results Data: {data_summary}

EXPLANATION RULES:
1. Start with a direct answer to the question
2. Use natural, conversational language (avoid technical terms)
3. Include specific names, numbers, and details
4. Format money as ₹X,XXX (Indian Rupees with commas)
5. For counts: "Dr. Smith saw 25 patients" not "patient_count: 25"
6. For revenue: "Dr. Patel earned ₹1,45,000" not "total_revenue: 145000"
7. For lists: Use bullet points or numbered lists for clarity
8. For comparisons: Highlight top performers or interesting insights
9. If no data: Be helpful and suggest alternative questions
10. Keep it concise but informative (2-4 sentences max unless listing data)

EXAMPLES:
- Count query → "Dr. Smith treated 42 patients in 2023."
- Revenue query → "Dr. Patel generated the highest revenue of ₹2,45,000, followed by Dr. Kumar with ₹1,89,500."
- List query → "Here are the top 3 patients by consultation amount: John Doe (₹5,000), Jane Smith (₹4,500), and Mike Johnson (₹4,200)."
- Average query → "The average consultation fee is ₹1,850."

Now provide a clear, helpful explanation:
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical data analyst who explains database results in simple, friendly language for healthcare professionals. Always be specific, accurate, and helpful."
                    },
                    {
                        "role": "user",
                        "content": explanation_prompt
                    }
                ],
                temperature=0.2,  # Low temperature for consistent explanations
                max_tokens=400
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # Enhanced fallback explanations
            return self._generate_fallback_explanation(question, result_df)
    
    def _generate_fallback_explanation(self, question, result_df):
        """
        Generate fallback explanation when OpenAI fails.
        
        Args:
            question (str): Original question
            result_df (pandas.DataFrame): Query results
            
        Returns:
            str: Fallback human-friendly explanation
        """
        if result_df.empty:
            return "No data found matching your query. Please try rephrasing your question or check if the data exists."
        
        # Basic pattern-based explanations as fallback
        explanation = f"Found {len(result_df)} record(s). "
        
        # Count-based results
        if 'patient_count' in result_df.columns:
            if len(result_df) == 1:
                count = result_df['patient_count'].iloc[0]
                if 'doctor' in result_df.columns:
                    doctor = result_df['doctor'].iloc[0]
                    explanation = f"Dr. {doctor} saw {count} patients."
                else:
                    explanation = f"Total patients: {count}"
            else:
                explanation = "Patient count breakdown:\n"
                for _, row in result_df.head(5).iterrows():
                    if 'doctor' in row:
                        explanation += f"• Dr. {row['doctor']}: {row['patient_count']} patients\n"
        
        # Revenue-based results
        elif any(col in result_df.columns for col in ['total_revenue', 'total_amt']):
            revenue_col = 'total_revenue' if 'total_revenue' in result_df.columns else 'total_amt'
            if len(result_df) == 1:
                revenue = result_df[revenue_col].iloc[0]
                explanation = f"Total revenue: ₹{revenue:,.2f}"
            else:
                explanation = "Revenue breakdown:\n"
                for _, row in result_df.head(5).iterrows():
                    if 'doctor' in row:
                        explanation += f"• Dr. {row['doctor']}: ₹{row[revenue_col]:,.2f}\n"
        
        # Average results
        elif any('avg' in col.lower() for col in result_df.columns):
            avg_cols = [col for col in result_df.columns if 'avg' in col.lower()]
            for col in avg_cols:
                value = result_df[col].iloc[0]
                explanation += f"Average amount: ₹{value:.2f}. "
        
        # List results
        else:
            if 'patient' in result_df.columns:
                explanation = f"Found {len(result_df)} patient records."
            elif 'doctor' in result_df.columns:
                doctors = result_df['doctor'].unique()
                explanation = f"Found {len(doctors)} doctors: {', '.join(doctors[:5])}"
            
    def translate_sql_results_to_layman(self, sql_query, result_df, context_question=None):
        """
        Standalone function to translate SQL results into layman-friendly explanations.
        
        Args:
            sql_query (str): The SQL query that was executed
            result_df (pandas.DataFrame): Results from the SQL query
            context_question (str, optional): Original question for better context
            
        Returns:
            str: Human-friendly explanation of the results
        """
        try:
            # Analyze the SQL query to understand what was asked
            query_type = self._analyze_query_type(sql_query)
            
            # Prepare detailed result summary
            result_summary = self._prepare_result_summary(result_df)
            
            # Create context-aware prompt
            prompt = f"""
You are a medical data analyst explaining database results to healthcare staff who may not be technical.

QUERY ANALYSIS:
SQL Query: {sql_query}
Query Type: {query_type}
{f'Original Question: {context_question}' if context_question else ''}

RESULTS DATA:
{result_summary}

TRANSLATION TASK:
Convert these technical SQL results into natural, conversational explanations that a doctor or medical administrator would easily understand.

FORMATTING RULES:
1. Use conversational language ("Dr. Smith treated 25 patients" not "COUNT: 25")
2. Format currency as ₹X,XXX with Indian Rupee symbol
3. Use proper names and titles (Dr., Patient, etc.)
4. For multiple results, use bullet points or numbered lists
5. Highlight key insights or notable patterns
6. If showing top results, mention it's a ranking
7. For empty results, be helpful and suggest alternatives

EXAMPLES OF GOOD TRANSLATIONS:
• COUNT query → "Dr. Patel saw 67 patients this month, making him the busiest doctor."
• SUM query → "The total revenue from cardiology consultations was ₹4,25,000."
• AVG query → "The average consultation fee across all doctors is ₹1,250."
• TOP query → "Here are the top 3 revenue-generating doctors: Dr. Kumar (₹3,45,000), Dr. Shah (₹2,89,000), and Dr. Patel (₹2,67,000)."

Provide a clear, professional, and friendly explanation:
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at translating technical database results into clear, professional explanations for healthcare professionals. Always be specific, accurate, and helpful."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Very low for consistent translations
                max_tokens=500
            )
            
            explanation = response.choices[0].message.content.strip()
            logger.info(f"Generated layman explanation: {explanation[:100]}...")
            return explanation
            
        except Exception as e:
            logger.error(f"Error in translate_sql_results_to_layman: {e}")
            return self._generate_fallback_explanation(context_question or "query", result_df)
    
    def _analyze_query_type(self, sql_query):
        """Analyze SQL query to determine its type and purpose."""
        query_lower = sql_query.lower()
        
        if 'count(' in query_lower:
            return "COUNT - Counting records/patients"
        elif 'sum(' in query_lower:
            return "SUM - Calculating totals/revenue"
        elif 'avg(' in query_lower:
            return "AVERAGE - Calculating averages"
        elif 'max(' in query_lower:
            return "MAXIMUM - Finding highest values"
        elif 'min(' in query_lower:
            return "MINIMUM - Finding lowest values"
        elif 'group by' in query_lower:
            return "GROUPED - Data grouped by categories"
        elif 'order by' in query_lower and 'limit' in query_lower:
            return "TOP/RANKED - Showing top results"
        elif 'distinct' in query_lower:
            return "UNIQUE - Showing unique values"
        else:
            return "SELECT - Retrieving specific records"
    
    def _prepare_result_summary(self, result_df):
        """Prepare comprehensive summary of results for OpenAI."""
        if result_df.empty:
            return "No data returned from query"
        
        summary = f"Rows returned: {len(result_df)}\n"
        summary += f"Columns: {list(result_df.columns)}\n"
        
        # Show data based on size
        if len(result_df) <= 3:
            summary += f"Complete results:\n{result_df.to_string(index=False)}\n"
        elif len(result_df) <= 10:
            summary += f"All results:\n{result_df.to_dict('records')}\n"
        else:
            summary += f"Sample of results (first 5):\n{result_df.head(5).to_dict('records')}\n"
            summary += f"Last 2 results:\n{result_df.tail(2).to_dict('records')}\n"
        
        # Add statistical summary for numeric columns
        numeric_cols = result_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            summary += f"\n{col} statistics:"
            summary += f" Total={result_df[col].sum():.2f},"
            summary += f" Average={result_df[col].mean():.2f},"
            summary += f" Max={result_df[col].max():.2f},"
            summary += f" Min={result_df[col].min():.2f}"
        
        # Add info about text columns
        text_cols = result_df.select_dtypes(include=['object']).columns
        for col in text_cols:
            unique_vals = result_df[col].nunique()
            if unique_vals <= 5:
                summary += f"\nUnique {col}: {list(result_df[col].unique())}"
            else:
                summary += f"\nUnique {col} count: {unique_vals}"
        
        return summary
    
    def ask_question(self, question):
        """
        Main function to ask a question about patient data using OpenAI.
        
        Args:
            question (str): Natural language question
            
        Returns:
            dict: Contains question, SQL query, results, and explanation
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Ensure patient_data table exists
            if not self.create_patient_data_table_if_missing():
                return {
                    'question': question,
                    'sql_query': None,
                    'result_df': pd.DataFrame(),
                    'explanation': "Could not access patient data. Please check if the database contains patient tables.",
                    'success': False,
                    'row_count': 0
                }
            
            # Generate SQL query using OpenAI
            sql_query = self.generate_sql_query(question)
            
            # Execute the query
            result_df = self.execute_sql_query(sql_query)
            
            # Generate human-friendly explanation
            explanation = self.explain_results_with_openai(question, sql_query, result_df)
            
            return {
                'question': question,
                'sql_query': sql_query,
                'result_df': result_df,
                'explanation': explanation,
                'success': True,
                'row_count': len(result_df)
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'question': question,
                'sql_query': None,
                'result_df': pd.DataFrame(),
                'explanation': f"Sorry, I couldn't process your question. Error: {str(e)}",
                'success': False,
                'row_count': 0
            }

# Example usage and testing
if __name__ == "__main__":
    # Initialize OpenAI chatbot
    chatbot = OpenAIPatientChatbot()
    
    # Test questions
    test_questions = [
        # "How many patients did Dr. Smith see in 2023?",
        "What is the total revenue generated by each doctor?",
        # "Show me the top 5 patients by amount paid",
        "What is the average consultation amount?",
        # "List all patients from Mumbai who paid more than 2000",
        "How many patients visited in January 2024?",
        "Which doctor generated the highest revenue?",
        # "Show me all blood test reports",
        "What's the total discount given last year?"
    ]
    
    print("=== OPENAI PATIENT CHATBOT TEST ===\n")
    
    # Test first few questions
    for question in test_questions[:3]:
        print(f"Q: {question}")
        result = chatbot.ask_question(question)
        print(f"SQL: {result['sql_query']}")
        print(f"Rows: {result['row_count']}")
        print(f"A: {result['explanation']}")
        print("-" * 60)