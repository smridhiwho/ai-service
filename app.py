import streamlit as st
import pandas as pd
from src.data_uploader import render_data_upload_page
import sqlite3
import os
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# Load environment variables

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Medical Center AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .tab-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
    
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .metric-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin-top: 1 rem;
    }
    
    .chat-container {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history_patient' not in st.session_state:
    st.session_state.chat_history_patient = []
if 'chat_history_inventory' not in st.session_state:
    st.session_state.chat_history_inventory = []

@st.cache_resource
def load_chatbot_modules():
    """Load chatbot modules with error handling and caching."""
    try:
        # Import from src directory
        from chatbot_patient import OpenAIPatientChatbot
        from chatbot_inventory import OpenAIInventoryChatbot
        return OpenAIPatientChatbot, OpenAIInventoryChatbot, True
    except ImportError as e:
        st.error(f"Error loading chatbot modules: {e}")
        st.info("Make sure chatbot_patient.py and chatbot_inventory.py are in the src/ directory")
        return None, None, False

@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_dynamic_sample_questions(question_type, current_query="", _chatbot=None):
    """Generate dynamic sample questions based on current context using OpenAI."""
    try:
        # Only try to import OpenAI if we have a chatbot instance
        if _chatbot is None:
            return get_default_sample_questions(question_type)
        # Create context-aware prompts based on question type
        # Create context-aware prompts based on question type
        if question_type == "patient":
            context = """
            You are generating sample questions for a medical diagnostic center's patient data chatbot.
            
            The database contains patient records with these columns:
            - case_no: Case number (e.g., "091169 - 0101")
            - opd_dt: OPD date (01/01/2021 format)
            - patient: Patient name (e.g., "Faridabibi I Pathan")
            - doctor: Doctor/Hospital name (e.g., "Raisa Anand", "Anand Laproscopy Centre", "Apara Hospital")
            - rpt_name: Report/scan type (e.g., "BRAIN WITH VENOGRAPHY", "H.R.C.T. CHEST SCREENING", "ABDOMEN PLAIN STUDY")
            - city: Patient city (e.g., "Anand", "Khambhat", "nAVLI")
            - total_amt, discount, net_amt, paid_amt: Financial amounts
            
            Common report types: Chest screenings, Abdomen studies, Emergency charges
            Common doctors: Raisa Anand, Anand Laproscopy Centre, Apara Hospital
            Common cities: Anand, Khambat, Navli 
            
            Current user context: "{current_query}"
            
            Generate 6 short, practical sample questions that medical staff would ask. Always mention the month and year, do not use refer to time relatively like "last month", or "last year". Also use different years.
            Focus on: patient counts, revenue analysis, doctor performance, city-wise data, time-based trends.
            Use actual names and terms from the data when relevant.
            Return only the questions, one per line, no numbering or bullets:
            """.format(current_query=current_query)
        
        elif question_type == "inventory":
            context = """
            The database contains inventory data from a medical center with these tables:

            STOCK TABLE:
            - Item: Product name (e.g., "HAND BRUSH", "NEPHTHALENE BALLS")
            - Unit: Measurement unit (NOS, GRAM)
            - Opening, Receipt, Total, Issue, Closing Stock: Stock movement data

            PURCHASE CHALLAN:
            - Document Name, Challan No, Challan Date
            - Party Name: Suppliers (e.g., "PARAMOUNT", "ASHWINI MEDICAL STORE")
            - Item Name: Products (e.g., "SCALP VEIN SET NG 23", "B I VALVE")
            - TR QTY, TR RATE: Transaction quantity and rates

            PURCHASE REGISTER & ORDERS:
            - V No, Date, Party, City, Total Amt
            - Common suppliers: ASHWINI MEDICAL STORE, PARAMOUNT, ENERGY PRINT
            - Common cities: ANAND, SURAT, BARODA
            - Amounts in rupees

            Focus on: stock levels, supplier analysis, city-wise purchases, item availability, reorder needs, transaction amounts.
            
            Current user context: "{current_query}"
            
            Generate 6 short, practical sample questions that inventory managers would ask.
            Focus on: stock levels, reorder needs, supplier analysis, purchase trends, item usage.
            
            Return only the questions, one per line, no numbering or bullets:
            """.format(current_query=current_query)
        
        else:
            return get_default_sample_questions(question_type)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=150,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at generating relevant, practical questions for medical center staff. Generate exactly 6 questions."
                },
                {
                    "role": "user", 
                    "content": context
                }
            ],
            temperature=0.7,

        )
        
        # Parse response into questions
        questions_text = response.choices[0].message.content.strip()
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        # Ensure we have exactly 6 questions
        if len(questions) >= 6:
            return questions[:6]
        elif len(questions) > 0:
            # If we got fewer than 6, pad with defaults
            defaults = get_default_sample_questions(question_type)
            return questions + defaults[:6-len(questions)]
        else:
            return get_default_sample_questions(question_type)
    
    except Exception as e:
        st.warning(f"Could not generate dynamic questions: {e}")
        return get_default_sample_questions(question_type)
    
def get_default_sample_questions(question_type):
    """Fallback default sample questions based on actual data structure."""
    if question_type == "patient":
        return [
            "Who are the top 3 doctors in 2024 with highest number of patients",
            "What's the total revenue from brain scans and chest screenings?", 
            "Show me all H.R.C.T. chest screening appointments",
            "Which cities do most of our patients come from?",
            "What's the average amount for different report types?",
            "How much emergency charges revenue did we collect?"
        ]
    elif question_type == "inventory":
        return [
            "show me a list of all items in the inventory with their closing stock",
            "Show me recent orders from PARAMOUNT supplier",
            "What's the total purchase amount from Anand city vendors?", 
            "How much NEPHTHALENE BALLS do we have in stock?",
            "Which items need restocking based on issue vs receipt?",
            "Show me all orders from ENERGY PRINT or G V SALES"
        ]
    else:
        return []
    """Load chatbot modules with error handling and caching."""
    try:
        # Import from src directory
        from chatbot_patient import OpenAIPatientChatbot
        from chatbot_inventory import OpenAIInventoryChatbot
        return OpenAIPatientChatbot, OpenAIInventoryChatbot, True
    except ImportError as e:
        st.error(f"Error loading chatbot modules: {e}")
        st.info("Make sure chatbot_patient.py and chatbot_inventory.py are in the src/ directory")
        return None, None, False


def check_database_connection(db_path="db/diagnostics.db"):
    """Check if database exists and has data."""
    if not db_path.startswith('/'):
        # Convert relative path from project root
        db_path = os.path.join(os.path.dirname(__file__), db_path)
    
    try:
        if not os.path.exists(db_path):
            return False, f"Database file not found: {db_path}"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        if not tables:
            return False, "Database is empty"
        
        return True, f"Connected! Found {len(tables)} tables"
    except Exception as e:
        return False, f"Database error: {e}"

def get_database_stats(db_path="db/diagnostics.db"):
    """Get basic statistics from the database."""
    if not db_path.startswith('/'):
        db_path = os.path.join(os.path.dirname(__file__), db_path)
    
    try:
        conn = sqlite3.connect(db_path)
        
        stats = {}
        
        # Patient data stats
        try:
            patient_count = pd.read_sql("SELECT COUNT(*) as count FROM patient_data", conn)
            stats['total_patients'] = patient_count['count'].iloc[0]
            
            revenue_total = pd.read_sql("SELECT SUM(total_amt) as revenue FROM patient_data", conn)
            stats['total_revenue'] = revenue_total['revenue'].iloc[0] if revenue_total['revenue'].iloc[0] else 0
            
            # Latest month stats
            latest_month = pd.read_sql("""
                SELECT COUNT(*) as count, SUM(total_amt) as revenue 
                FROM patient_data 
                WHERE strftime('%Y-%m', opd_dt) = strftime('%Y-%m', 'now')
            """, conn)
            stats['current_month_patients'] = latest_month['count'].iloc[0]
            stats['current_month_revenue'] = latest_month['revenue'].iloc[0] if latest_month['revenue'].iloc[0] else 0
            
        except Exception as e:
            st.warning(f"Patient data not available: {e}")
            stats['total_patients'] = 0
            stats['total_revenue'] = 0
            stats['current_month_patients'] = 0
            stats['current_month_revenue'] = 0
        
        # Table count
        tables = pd.read_sql("SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'", conn)
        stats['total_tables'] = tables['count'].iloc[0]
        
        conn.close()
        return stats
    except Exception as e:
        st.error(f"Error getting database stats: {e}")
        return {
            'total_patients': 0, 
            'total_revenue': 0, 
            'total_tables': 0,
            'current_month_patients': 0,
            'current_month_revenue': 0
        }

def render_patient_chatbot():
    """Render the Patient Chatbot tab."""
    st.markdown('<div class="tab-header">🩺 Patient Data Chatbot</div>', unsafe_allow_html=True)
    
    # Database connection status
    is_connected, message = check_database_connection()
    if is_connected:
        st.success(f"✅ {message}")
    else:
        st.error(f"❌ {message}")
        st.stop()
    
    # Load chatbot
    PatientChatbot, _, modules_loaded = load_chatbot_modules()
    
    if not modules_loaded or not PatientChatbot:
        st.error("❌ Patient chatbot module not available")
        st.info("Make sure `chatbot_patient.py` exists in the `src/` directory with `OpenAIPatientChatbot` class")
        return
    
    # Initialize chatbot
    try:
        db_path = os.path.join(os.path.dirname(__file__), "db/diagnostics.db")
        chatbot = PatientChatbot(db_path=db_path)
        
        # Chat interface
        st.markdown('<div class="info-box">💡 Ask questions about patient data, revenue, doctors, appointments, etc.</div>', unsafe_allow_html=True)
        
        # Get current query context for dynamic questions
        current_context = st.session_state.get('patient_question_input', '')
        
        # Generate dynamic sample questions
        with st.spinner("💭 Generating smart suggestions..."):
            sample_questions = generate_dynamic_sample_questions("patient", current_context, chatbot)
        
        # Sample questions with refresh option
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("**💭 Smart suggestions based on your query:**")
        with col2:
            if st.button("🔄 New Ideas", key="refresh_patient_suggestions", help="Generate fresh question suggestions"):
                # Clear cache for this specific context to force regeneration
                st.cache_data.clear()
                st.rerun()
        
        cols = st.columns(3)
        for i, question in enumerate(sample_questions):
            col = cols[i % 3]
            if col.button(f"💡 {question}", key=f"smart_patient_{i}"):
                st.session_state.current_patient_question = question
                st.rerun()  # Refresh to update the input field
        
        # Question input
        user_question = st.text_input(
            "🤔 Your Question:",
            value=st.session_state.get('current_patient_question', ''),
            placeholder="e.g., How many patients did Dr. Smith see in 2023?",
            key="patient_question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Ask Question", key="ask_patient", type="primary")
        with col2:
            if st.button("🗑️ Clear History", key="clear_patient_history"):
                st.session_state.chat_history_patient = []
                st.success("Chat history cleared!")
        
        if ask_button and user_question.strip():
            with st.spinner("🤖 Processing your question..."):
                try:
                    result = chatbot.ask_question(user_question)
                    
                    # Display results
                    if result['success']:
                        st.success("✅ Query Successful!")
                        
                        # Show results in columns
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown("### 💬 Answer:")
                            st.markdown(f"<div class='chat-container'>{result['explanation']}</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("📊 Rows Found", result['row_count'])
                            
                            # Show SQL query in expandable section
                            with st.expander("🔧 SQL Query"):
                                st.code(result['sql_query'], language='sql')
                        
                        # Show data table if available and not too large
                        if not result['result_df'].empty:
                            if len(result['result_df']) <= 50:
                                with st.expander("📋 View Raw Data"):
                                    st.dataframe(result['result_df'], use_container_width=True)
                            else:
                                with st.expander("📋 View Sample Data (first 20 rows)"):
                                    st.dataframe(result['result_df'].head(20), use_container_width=True)
                                    st.info(f"Showing 20 of {len(result['result_df'])} total rows")
                        
                        # Add to chat history
                        st.session_state.chat_history_patient.append({
                            'question': user_question,
                            'answer': result['explanation'],
                            'rows': result['row_count'],
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Clear the input
                        st.session_state.current_patient_question = ""
                    else:
                        st.error(f"❌ {result['explanation']}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure your OpenAI API key is set in .env file")
        
        elif ask_button:
            st.warning("⚠️ Please enter a question!")
        
        # Chat history
        if st.session_state.chat_history_patient:
            st.markdown("---")
            st.markdown("### 💭 Recent Chat History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history_patient[-5:])):
                with st.expander(f"🕐 {chat['timestamp']} - {chat['question'][:50]}..."):
                    st.markdown(f"**❓ Question:** {chat['question']}")
                    st.markdown(f"**💬 Answer:** {chat['answer']}")
                    st.markdown(f"**📊 Data:** {chat['rows']} rows returned")
    
    except Exception as e:
        st.error(f"❌ Error initializing patient chatbot: {e}")
        st.info("Please check your database connection and module imports")

def render_inventory_chatbot():
    """Render the Inventory Chatbot tab."""
    st.markdown('<div class="tab-header">📦 Inventory Management Chatbot</div>', unsafe_allow_html=True)
    
    # Database connection status
    is_connected, message = check_database_connection()
    if is_connected:
        st.success(f"✅ {message}")
    else:
        st.error(f"❌ {message}")
        st.stop()
    
    # Load chatbot  
    _, InventoryChatbot, modules_loaded = load_chatbot_modules()
    
    if not modules_loaded or not InventoryChatbot:
        st.error("❌ Inventory chatbot module not available")
        st.info("Make sure `chatbot_inventory.py` exists in the `src/` directory with `OpenAIInventoryChatbot` class")
        return
    
    # Initialize chatbot
    try:
        db_path = os.path.join(os.path.dirname(__file__), "db/diagnostics.db")
        chatbot = InventoryChatbot(db_path=db_path)
        
        # Chat interface
        st.markdown('<div class="info-box">💡 Ask questions about inventory, stock levels, suppliers, purchase orders, etc.</div>', unsafe_allow_html=True)
        
        # Get current query context for dynamic questions
        current_context = st.session_state.get('inventory_question_input', '')
        
        # Generate dynamic sample questions
        with st.spinner("💭 Generating smart suggestions..."):
            sample_questions = generate_dynamic_sample_questions("inventory", current_context, chatbot)
        
        # Sample questions with refresh option
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("**💭 Smart suggestions based on your query:**")
        with col2:
            if st.button("🔄 New Ideas", key="refresh_inventory_suggestions", help="Generate fresh question suggestions"):
                # Clear cache for this specific context to force regeneration
                st.cache_data.clear()
                st.rerun()
        
        cols = st.columns(3)
        for i, question in enumerate(sample_questions):
            col = cols[i % 3]
            if col.button(f"💡 {question}", key=f"smart_inventory_{i}"):
                st.session_state.current_inventory_question = question
                st.rerun()  # Refresh to update the input field
        
        # Question input
        user_question = st.text_input(
            "🤔 Your Question:",
            value=st.session_state.get('current_inventory_question', ''),
            placeholder="e.g., Which items need restocking?",
            key="inventory_question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Ask Question", key="ask_inventory", type="primary")
        with col2:
            if st.button("🗑️ Clear History", key="clear_inventory_history"):
                st.session_state.chat_history_inventory = []
                st.success("Chat history cleared!")
        
        if ask_button and user_question.strip():
            with st.spinner("🤖 Processing your question..."):
                try:
                    result = chatbot.ask_inventory_question(user_question)
                    
                    # Display results
                    if result['success']:
                        st.success("✅ Query Successful!")
                        
                        # Show table identification and metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.info(f"📋 **Table:** {result['identified_table']}")
                        with col2:
                            st.metric("📊 Rows Found", result['row_count'])
                        with col3:
                            with st.expander("🔧 SQL Query"):
                                st.code(result['sql_query'], language='sql')
                        
                        # Show answer
                        st.markdown("### 💬 Answer:")
                        st.markdown(f"<div class='chat-container'>{result['explanation']}</div>", unsafe_allow_html=True)
                        
                        # Show data table if available
                        if not result['result_df'].empty:
                            if len(result['result_df']) <= 30:
                                with st.expander("📊 View Raw Data"):
                                    st.dataframe(result['result_df'], use_container_width=True)
                            else:
                                with st.expander("📊 View Sample Data (first 15 rows)"):
                                    st.dataframe(result['result_df'].head(15), use_container_width=True)
                                    st.info(f"Showing 15 of {len(result['result_df'])} total rows")
                        
                        # Add to chat history
                        st.session_state.chat_history_inventory.append({
                            'question': user_question,
                            'answer': result['explanation'],
                            'table': result['identified_table'],
                            'rows': result['row_count'],
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Clear the input
                        st.session_state.current_inventory_question = ""
                    else:
                        st.error(f"❌ {result['explanation']}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure your OpenAI API key is set in .env file")
        
        elif ask_button:
            st.warning("⚠️ Please enter a question!")
        
        # Chat history
        if st.session_state.chat_history_inventory:
            st.markdown("---")
            st.markdown("### 💭 Recent Chat History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history_inventory[-5:])):
                with st.expander(f"🕐 {chat['timestamp']} - Table: `{chat['table']}` - {chat['question'][:40]}..."):
                    st.markdown(f"**❓ Question:** {chat['question']}")
                    st.markdown(f"**📋 Table Used:** `{chat['table']}`")
                    st.markdown(f"**💬 Answer:** {chat['answer']}")
                    st.markdown(f"**📊 Data:** {chat['rows']} rows returned")
    
    except Exception as e:
        st.error(f"❌ Error initializing inventory chatbot: {e}")
        st.info("Please check your database connection and module imports")

def render_forecasting_dashboard():
    """Render the Forecasting Dashboard tab."""
    st.markdown('<div class="tab-header">📈 Forecasting Dashboard</div>', unsafe_allow_html=True)
    
    # Check database connection
    is_connected, message = check_database_connection()
    if is_connected:
        st.success(f"✅ Database connected: {message}")
    else:
        st.error(f"❌ {message}")
        st.stop()
    
    # Get database stats
    stats = get_database_stats()
    
    # Display current stats
    st.markdown("### 📊 Current Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.metric("Total Patients", f"{stats['total_patients']:,}")

    with col2:
        with st.container():
            st.metric("Total Revenue", f"₹{stats['total_revenue']:,.0f}")

    with col3:
        with st.container():
            st.metric("This Month Patients", f"{stats['current_month_patients']:,}")

    with col4:
        with st.container():
            st.metric("This Month Revenue", f"₹{stats['current_month_revenue']:,.0f}")
    
    st.markdown("---")
    
    # Forecast settings
    st.markdown("### ⚙️ Forecast Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_type = st.selectbox(
            "📊 Select Forecast Type",
            ["Monthly Revenue", "Patient Count"],
            # ["Monthly Revenue", "Patient Count", "Inventory Stock"],
            help="Choose what metric you want to forecast"
        )
    
    with col2:
        months_ahead = st.slider(
            "📅 Forecast Period (Months)",
            min_value=3,
            max_value=24,
            value=12,
            help="How many months into the future to predict"
        )
    
    with col3:
        confidence_level = st.selectbox(
            "🎯 Confidence Level",
            ["80%", "90%", "95%"],
            index=1,
            help="Confidence interval for predictions"
        )
    
    # Generate forecast button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        generate_forecast = st.button("🔮 Generate Forecast", key="generate_forecast", type="primary")
    
    if generate_forecast:
        with st.spinner(f"🤖 Generating {forecast_type.lower()} forecast for {months_ahead} months..."):
            try:
                # Try to import forecasting module
                try:
                    from forecasting import MedicalForecastingSystem
                    
                    # Initialize forecasting system
                    db_path = os.path.join(os.path.dirname(__file__), "db/diagnostics.db")
                    forecasting_system = MedicalForecastingSystem(db_path=db_path)
                    
                    # Generate forecast based on type
                    if forecast_type == "Monthly Revenue":
                        result = forecasting_system.forecast_monthly_revenue(months_ahead=months_ahead)
                    elif forecast_type == "Patient Count":
                        result = forecasting_system.forecast_patient_count(months_ahead=months_ahead)
                    # else:
                    #     result = forecasting_system.forecast_inventory_stock(months_ahead=months_ahead)
                    
                    if result['success']:
                        st.success(f"✅ {forecast_type} forecast generated successfully!")
                        
                        # Display the forecast chart
                        st.plotly_chart(result['plotly_figure'], use_container_width=True)
                        
                        # Show insights
                        if 'insights' in result:
                            st.markdown("### 💡 Forecast Insights")
                            insights = result['insights']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            for i, (key, value) in enumerate(insights.items()):
                                col = [col1, col2, col3][i % 3]
                                with col:
                                    st.info(f"**{key.replace('_', ' ').title()}:** {value}")
                    
                    else:
                        st.error(f"❌ Forecast generation failed: {result.get('error', 'Unknown error')}")
                
                except ImportError:
                    # Fallback: Create mock forecast for demonstration
                    st.warning("⚠️ Forecasting module not available. Showing demo forecast.")
                    
                    # Create mock data for demonstration
                    import numpy as np
                    dates = pd.date_range(start='2024-01-01', periods=months_ahead, freq='M')
                    
                    if forecast_type == "Monthly Revenue":
                        base_value = stats['total_revenue'] / 12 if stats['total_revenue'] > 0 else 200000
                        values = np.random.normal(base_value, base_value * 0.2, months_ahead)
                        title = "Monthly Revenue Forecast (Demo)"
                        y_title = "Revenue (₹)"
                        format_str = "₹{:,.0f}"
                    elif forecast_type == "Patient Count":
                        base_value = stats['total_patients'] / 12 if stats['total_patients'] > 0 else 500
                        values = np.random.normal(base_value, base_value * 0.15, months_ahead)
                        title = "Monthly Patient Count Forecast (Demo)"
                        y_title = "Number of Patients"
                        format_str = "{:,.0f}"
                    # else:
                    #     values = np.random.normal(1000, 200, months_ahead)
                    #     title = "Inventory Stock Forecast (Demo)"
                    #     y_title = "Stock Level"
                    #     format_str = "{:,.0f}"
                    
                    # Ensure positive values
                    values = np.abs(values)
                    
                    # Create forecast chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:,.0f}<extra></extra>'
                    ))
                    
                    # Add confidence intervals (mock)
                    upper_bound = values * 1.1
                    lower_bound = values * 0.9
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=upper_bound,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=lower_bound,
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        line=dict(width=0),
                        name=f'{confidence_level} Confidence Interval',
                        hoverinfo='skip'
                    ))
                    
                    fig.update_layout(
                        title=title,
                        xaxis_title="Date",
                        yaxis_title=y_title,
                        template="plotly_white",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show mock insights
                    st.markdown("### 💡 Forecast Insights (Demo)")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.info(f"**Average Forecast:** {format_str.format(values.mean())}")
                    with col2:
                        st.info(f"**Minimum Expected:** {format_str.format(values.min())}")
                    with col3:
                        st.info(f"**Maximum Expected:** {format_str.format(values.max())}")
                    
                    st.info("💡 **Note:** This is a demo forecast. Implement `forecasting.py` with Prophet for real predictions.")
                
            except Exception as e:
                st.error(f"❌ Error generating forecast: {e}")
                st.info("Please check that your forecasting module is properly configured.")

def main():
    """Main Streamlit application."""
    
    # Main header
    st.markdown('<div class="main-header">🏥 Medical Center AI Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=Medical+AI", width=200)
        
        st.markdown("### 🚀 Dashboard Features")
        st.markdown("- 🩺 **Patient Analytics:** AI-powered patient data insights")
        st.markdown("- 📦 **Smart Inventory:** Intelligent stock management") 
        st.markdown("- 📈 **Future Forecasting:** Predictive analytics with Prophet")
        st.markdown("- 🤖 **Natural Language:** Ask questions in plain English")
        
        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        
        # Check database connection
        is_db_connected, db_message = check_database_connection()
        if is_db_connected:
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Issue")
        
        # Check if OpenAI key is set
        if os.getenv('OPENAI_API_KEY'):
            st.success("✅ OpenAI API Ready")
        else:
            st.error("❌ OpenAI Key Missing")
            st.info("Add OPENAI_API_KEY to .env file")
        
        # System info
        st.markdown("---")
        st.markdown("### 📋 Quick Stats")
        try:
            stats = get_database_stats()
            st.metric("Total Patients", f"{stats['total_patients']:,}")
            st.metric("Database Tables", stats['total_tables'])
        except:
            st.warning("Stats unavailable")
        
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
    

    # main tabs:
    tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Patient Chatbot", 
    "📦 Inventory Chatbot", 
    "📈 Forecasting Dashboard",
    "📤 Data Upload"
])

    
    with tab1:
        render_patient_chatbot()
    
    with tab2:
        render_inventory_chatbot()
    
    with tab3:
        render_forecasting_dashboard()
    
    with tab4:
        render_data_upload_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
        "🏥 Medical Center AI Dashboard v1.0 | Built with Streamlit, OpenAI & Prophet | "
        f"Database: {len(get_database_stats())} tables"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()