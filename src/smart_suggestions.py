import streamlit as st
import pandas as pd
import sqlite3
import logging
from openai import OpenAI
from datetime import datetime
import random
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

logger = logging.getLogger(__name__)

class SmartSuggestionEngine:
    def __init__(self, db_path):
        self.db_path = db_path
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.patient_schema = self._get_patient_schema()
        self.inventory_schema = self._get_inventory_schema()
        
        
    def _get_patient_schema(self):
        """Get actual patient data schema and sample data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get column info
            columns_df = pd.read_sql("PRAGMA table_info(patient_data)", conn)
            columns = columns_df['name'].tolist()
            
            # Get sample data for context
            sample_df = pd.read_sql("SELECT * FROM patient_data LIMIT 5", conn)
            
            # Get distinct values for key columns
            doctors = pd.read_sql("SELECT DISTINCT doctor FROM patient_data WHERE doctor IS NOT NULL LIMIT 10", conn)['doctor'].tolist()
            cities = pd.read_sql("SELECT DISTINCT city FROM patient_data WHERE city IS NOT NULL LIMIT 10", conn)['city'].tolist()
            reports = pd.read_sql("SELECT DISTINCT rpt_name FROM patient_data WHERE rpt_name IS NOT NULL LIMIT 10", conn)['rpt_name'].tolist()
            
            # Get date range
            date_range = pd.read_sql("SELECT MIN(opd_dt) as min_date, MAX(opd_dt) as max_date FROM patient_data", conn)
            
            conn.close()
            
            return {
                'columns': columns,
                'sample_data': sample_df.to_dict('records')[:3],
                'doctors': doctors,
                'cities': cities,
                'reports': reports,
                'date_range': date_range.to_dict('records')[0],
                'total_patients': len(sample_df)
            }
        except Exception as e:
            logger.error(f"Error getting patient schema: {e}")
            return {}
    
    def _get_inventory_schema(self):
        """Get actual inventory schema and sample data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all inventory tables
            tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name != 'patient_data'", conn)
            tables = tables_df['name'].tolist()
            
            inventory_info = {}
            for table in tables:
                try:
                    # Get columns
                    columns_df = pd.read_sql(f"PRAGMA table_info({table})", conn)
                    columns = columns_df['name'].tolist()
                    
                    # Get sample data
                    sample_df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
                    
                    inventory_info[table] = {
                        'columns': columns,
                        'sample_data': sample_df.to_dict('records')
                    }
                except:
                    continue
            
            conn.close()
            return inventory_info
            
        except Exception as e:
            logger.error(f"Error getting inventory schema: {e}")
            return {}
    
    
    
    def generate_patient_suggestions(self, current_input="", context_history=[]):
        """Generate smart patient data suggestions based on actual data"""
        try:
            # Analyze current input for context
            input_context = self._analyze_input_context(current_input, "patient")
            
            # Create data-driven prompt
            prompt = f"""
Generate 6 precise, actionable questions for a medical center's patient data chatbot.

ACTUAL DATABASE SCHEMA:
- Columns: {self.patient_schema.get('columns', [])}
- Date Range: {self.patient_schema.get('date_range', {})}
- Sample Doctors: {self.patient_schema.get('doctors', [])[:5]}
- Sample Cities: {self.patient_schema.get('cities', [])[:5]}
- Sample Reports: {self.patient_schema.get('reports', [])[:3]}

CURRENT USER CONTEXT:
- User Input: "{current_input}"
- Context Type: {input_context}
- Recent Questions: {context_history[-3:] if context_history else []}

QUESTION GENERATION RULES:
1. Use ACTUAL doctor names from the database: {', '.join(self.patient_schema.get('doctors', [])[:3])}
2. Use ACTUAL cities from the database: {', '.join(self.patient_schema.get('cities', [])[:3])}
3. Use ACTUAL report types: {', '.join(self.patient_schema.get('reports', [])[:2])}
4. Include specific years (2018-2025) - avoid relative terms like "last year"
5. Focus on: revenue analysis, patient counts, doctor performance, city trends
6. Make questions actionable for medical staff
7. Vary complexity - some simple counts, some analytical queries

EXAMPLES OF GOOD QUESTIONS:
- "How many patients did Dr. {self.patient_schema.get('doctors', ['Smith'])[0]} treat in 2024?"
- "What's the total revenue from {self.patient_schema.get('cities', ['Mumbai'])[0]} patients?"
- "Show me all {self.patient_schema.get('reports', ['X-RAY'])[0]} reports this year"
- "Which doctor has the highest average consultation fee?"
- "Compare patient counts between 2023 and 2024"

Generate 6 questions, one per line, no numbering:"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at generating precise, data-driven questions for medical staff using actual database content."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            questions = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
            return questions[:6]
            
        except Exception as e:
            logger.error(f"Error generating patient suggestions: {e}")
            return self._get_fallback_patient_questions()
    
    def generate_inventory_suggestions(self, current_input="", context_history=[]):
        """Generate smart inventory suggestions based on actual tables"""
        try:
            # Get available tables and their context
            available_tables = list(self.inventory_schema.keys())
            
            # Analyze input context
            input_context = self._analyze_input_context(current_input, "inventory")
            
            # Create context-aware prompt
            prompt = f"""
Generate 6 precise inventory management questions using ACTUAL database tables and data.

ACTUAL INVENTORY TABLES:
{self._format_inventory_tables_for_prompt()}

CURRENT CONTEXT:
- User Input: "{current_input}"
- Context: {input_context}
- Available Tables: {available_tables}

QUESTION RULES:
1. Use ACTUAL table names and column names from above
2. Reference specific suppliers, items, or data from samples
3. Focus on: stock levels, purchase analysis, supplier performance, reorder alerts
4. Make questions immediately actionable for inventory managers
5. Include specific amounts, quantities, dates where relevant
6. Vary between operational (stock levels) and analytical (trends, comparisons)

EXAMPLES OF GOOD QUESTIONS:
- "Show current stock levels for all items below 10 units"
- "What's the total purchase amount from [actual supplier name]?"
- "Which items have been ordered but not received?"
- "List all stock movements for [actual item name]"
- "Compare purchase costs between suppliers for common items"
- "Show items that need immediate restocking"

Generate 6 specific, actionable questions, one per line:"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate precise inventory questions using actual database schema and data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            questions = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
            return questions[:6]
            
        except Exception as e:
            logger.error(f"Error generating inventory suggestions: {e}")
            return self._get_fallback_inventory_questions()
    
    def _analyze_input_context(self, user_input, chatbot_type):
        """Analyze user input to understand context and intent"""
        if not user_input:
            return "general"
        
        input_lower = user_input.lower()
        
        if chatbot_type == "patient":
            if any(word in input_lower for word in ['revenue', 'money', 'amount', 'payment']):
                return "financial"
            elif any(word in input_lower for word in ['doctor', 'physician']):
                return "doctor_focused"
            elif any(word in input_lower for word in ['patient', 'count', 'number']):
                return "patient_analytics"
            elif any(word in input_lower for word in ['city', 'location', 'place']):
                return "geographic"
            elif any(word in input_lower for word in ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']):
                return "time_based"
        
        elif chatbot_type == "inventory":
            if any(word in input_lower for word in ['stock', 'inventory', 'available']):
                return "stock_levels"
            elif any(word in input_lower for word in ['purchase', 'buy', 'procurement']):
                return "purchasing"
            elif any(word in input_lower for word in ['supplier', 'vendor', 'party']):
                return "supplier_analysis"
            elif any(word in input_lower for word in ['order', 'requisition']):
                return "ordering"
        
        return "general"
    
    def _format_inventory_tables_for_prompt(self):
        """Format inventory schema for AI prompt"""
        formatted = ""
        for table_name, info in self.inventory_schema.items():
            formatted += f"\nTable: {table_name}\n"
            formatted += f"  Columns: {info.get('columns', [])}\n"
            if info.get('sample_data'):
                formatted += f"  Sample: {info['sample_data'][0]}\n"
        return formatted
    
    def _get_fallback_patient_questions(self):
        """Enhanced fallback questions based on actual schema"""
        doctors = self.patient_schema.get('doctors', ['Dr. Smith', 'Dr. Patel', 'Dr. Kumar'])
        cities = self.patient_schema.get('cities', ['Mumbai', 'Delhi', 'Bangalore'])
        
        questions = [
            f"How many patients did {doctors[0]} see in 2024?",
            f"What's the total revenue from {cities[0]} patients?",
            "Which doctor generated the highest revenue in 2023?",
            f"Show me all patients from {cities[1]} with amounts over ₹2000",
            "What's the average consultation fee across all doctors?",
            f"Compare patient counts between {doctors[0]} and {doctors[1] if len(doctors) > 1 else doctors[0]}"
        ]
        return questions
    
    def _get_fallback_inventory_questions(self):
        """Enhanced fallback questions for inventory"""
        tables = list(self.inventory_schema.keys())
        
        questions = [
            "Show me all items with stock levels below 10 units",
            "What's the total purchase amount this month?",
            "List all recent orders from our top suppliers",
            "Which items need immediate restocking?",
            "Show me all pending purchase orders",
            "What's the current value of our total inventory?"
        ]
        return questions

def get_default_sample_questions(question_type):
    """Enhanced fallback default sample questions based on actual data structure."""
    if question_type == "patient":
        return [
            "How many patients were treated in 2024?",
            "What's the total revenue generated in 2023?",
            "Which doctor has the highest patient count in 2022?",
            "Show me revenue breakdown by city",
            "What's the average consultation amount in 2024?",
            "List all emergency cases in January 2023"
        ]
    elif question_type == "inventory":
        return [
            "Show me all items with low stock levels",
            "What's the total inventory value?",
            "List recent purchase orders",
            "Which party do we buy from most?",
            "Show items that have 0 stock",
            "What's our monthly purchase spending in 2024?"
        ]
    else:
        return [
            "Show me a summary of the data",
            "What are the recent trends in 2025?",
            "Give me key performance metrics",
            "Show me top 10 units",
            "What's the total count?",
            "Display recent activity"
        ]

# Updated function for app.py
@st.cache_resource
def initialize_suggestion_engine():
    """Initialize the smart suggestion engine"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        db_path = os.path.join(os.path.dirname(__file__), "db/diagnostics.db")
        return SmartSuggestionEngine(db_path, client)
    except Exception as e:
        st.error(f"Error initializing suggestion engine: {e}")
        return None

# Replace your existing generate_dynamic_sample_questions function with this:
def generate_smart_suggestions(question_type, current_query="", chat_history=[], suggestion_engine=None):
    """Generate smart suggestions using actual database context"""
    
    if not suggestion_engine:
        # Fallback to basic suggestions
        return get_default_sample_questions(question_type)
    
    try:
        if question_type == "patient":
            return suggestion_engine.generate_patient_suggestions(current_query, chat_history)
        elif question_type == "inventory":
            return suggestion_engine.generate_inventory_suggestions(current_query, chat_history)
        else:
            return get_default_sample_questions(question_type)
            
    except Exception as e:
        logger.error(f"Error generating smart suggestions: {e}")
        return get_default_sample_questions(question_type)

# Enhanced version of your existing function for better caching
@st.cache_data(ttl=180, show_spinner=False)  # Cache for 3 minutes
def cached_smart_suggestions(question_type, input_hash, _suggestion_engine):
    """Cached version of smart suggestions"""
    return generate_smart_suggestions(question_type, "", [], _suggestion_engine)