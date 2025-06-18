import os
import pandas as pd
from sqlalchemy import create_engine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_connections():
    """Test different database paths to find where patient_data exists."""
    
    possible_paths = [
        '../db/diagnostics.db',
        './db/diagnostics.db', 
        'db/diagnostics.db',
        '/diagnostics_ai/db/diagnostics.db',
        'diagnostics.db'
    ]
    
    print("=== TESTING DATABASE PATHS ===")
    
    for db_path in possible_paths:
        print(f"\nTesting path: {db_path}")
        print(f"  File exists: {os.path.exists(db_path)}")
        print(f"  Absolute path: {os.path.abspath(db_path)}")
        
        if os.path.exists(db_path):
            try:
                engine = create_engine(f'sqlite:///{db_path}')
                
                # Check for patient_data table
                tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)
                all_tables = tables_df['name'].tolist()
                
                print(f"  ✅ Database readable")
                print(f"  Tables found: {len(all_tables)}")
                
                if 'patient_data' in all_tables:
                    count_df = pd.read_sql("SELECT COUNT(*) as count FROM patient_data", engine)
                    print(f"  🎯 FOUND patient_data table with {count_df['count'].iloc[0]} rows!")
                    return db_path
                else:
                    print(f"  ❌ No patient_data table found")
                    print(f"  Available tables: {all_tables}")
                    
            except Exception as e:
                print(f"  ❌ Error reading database: {e}")
        else:
            print(f"  ❌ File does not exist")
    
    return None

def create_patient_data_table_directly():
    """Create patient_data table by combining existing tables."""
    
    # Try to find the correct database path
    db_path = test_database_connections()
    
    if not db_path:
        print("❌ Could not find database with tables!")
        return False
    
    print(f"\n=== CREATING patient_data TABLE ===")
    print(f"Using database: {db_path}")
    
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Get all patient tables
        patient_tables = []
        tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)
        
        for table_name in tables_df['name']:
            if table_name.startswith('table_20'):  # table_2018, table_2019, etc.
                patient_tables.append(table_name)
        
        print(f"Found patient tables: {patient_tables}")
        
        if not patient_tables:
            print("❌ No patient tables found!")
            return False
        
        # Combine all patient tables
        combined_df = pd.DataFrame()
        
        for table_name in patient_tables:
            print(f"Reading {table_name}...")
            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
            print(f"  Rows: {len(df)}")
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        print(f"Total combined rows: {len(combined_df)}")
        
        # Create patient_data table
        combined_df.to_sql('patient_data', engine, if_exists='replace', index=False)
        print("✅ Created patient_data table")
        
        # Verify
        count_df = pd.read_sql("SELECT COUNT(*) as count FROM patient_data", engine)
        print(f"✅ Verification: patient_data has {count_df['count'].iloc[0]} rows")
        
        return db_path
        
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        return False

def test_chatbot_with_correct_path(db_path):
    """Test the chatbot with the correct database path."""
    
    print(f"\n=== TESTING CHATBOT ===")
    print(f"Using database: {db_path}")
    
    try:
        # Test a simple query
        engine = create_engine(f'sqlite:///{db_path}')
        
        test_queries = [
            "SELECT COUNT(*) as total_patients FROM patient_data",
            "SELECT doctor, COUNT(*) as patient_count FROM patient_data GROUP BY doctor ORDER BY patient_count DESC LIMIT 5",
            "SELECT SUM(total_amt) as total_revenue FROM patient_data"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: {query}")
            result_df = pd.read_sql(query, engine)
            print(f"✅ Query worked! Result:\n{result_df}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 FIXING CHATBOT DATABASE ISSUES...")
    
    # Step 1: Find correct database path and create patient_data table
    db_path = create_patient_data_table_directly()
    
    if db_path:
        # Step 2: Test chatbot queries
        success = test_chatbot_with_correct_path(db_path)
        
        if success:
            print(f"\n" + "="*60)
            print("🎉 SUCCESS! Here's what to do:")
            print(f"1. Update your chatbot to use this path: {db_path}")
            print(f"2. Full absolute path: {os.path.abspath(db_path)}")
            print("3. Your chatbot should now work!")
            print("="*60)
        else:
            print("\n❌ Chatbot still has issues")
    else:
        print("\n❌ Could not fix the database issue")