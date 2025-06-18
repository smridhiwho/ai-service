import os
import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
import logging
from pathlib import Path
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, db_path='../db/diagnostics.db'):
        """Initialize the data ingestion class with database path."""
        self.db_path = db_path
        # Create db directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Create SQLAlchemy engine
        self.engine = create_engine(f'sqlite:///{db_path}')
        logger.info(f"Database connection established at: {db_path}")
    
    def filename_to_table_name(self, filename):
        """Convert filename to snake_case table name."""
        # Remove file extension
        name = os.path.splitext(filename)[0]
        # Replace spaces and hyphens with underscores
        name = re.sub(r'[-\s]+', '_', name)
        # Convert to lowercase
        name = name.lower()
        # Remove any non-alphanumeric characters except underscores
        name = re.sub(r'[^a-z0-9_]', '', name)
        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = 'table_' + name
        return name
    
    def clean_column_names(self, df):
        """Clean column names to be SQL-friendly."""
        df.columns = df.columns.astype(str)  # Convert to string
        # Strip whitespace and replace problematic characters
        df.columns = [col.strip().replace(' ', '_').replace('.', '_').replace('-', '_').lower() 
                     for col in df.columns]
        # Remove any non-alphanumeric characters except underscores
        df.columns = [re.sub(r'[^a-z0-9_]', '', col) for col in df.columns]
        # Ensure column names don't start with numbers
        df.columns = [f'col_{col}' if col and col[0].isdigit() else col for col in df.columns]
        # Handle empty column names
        df.columns = [f'unnamed_{i}' if not col else col for i, col in enumerate(df.columns)]
        return df
    
    def check_table_exists(self, table_name):
        """Check if table exists in database."""
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"
            ), {"table_name": table_name})
            return result.fetchone() is not None
    
    def get_existing_data_hash(self, table_name, df):
        """Get hash of existing data to identify duplicates."""
        if not self.check_table_exists(table_name):
            return set()
        
        try:
            existing_df = pd.read_sql_table(table_name, self.engine)
            # Create hash based on all columns to identify duplicates
            existing_hashes = set()
            for _, row in existing_df.iterrows():
                row_hash = hash(tuple(str(val) for val in row.values))
                existing_hashes.add(row_hash)
            return existing_hashes
        except Exception as e:
            logger.warning(f"Could not read existing data from {table_name}: {e}")
            return set()
    
    def remove_duplicates(self, df, existing_hashes):
        """Remove rows that already exist in the database."""
        if not existing_hashes:
            return df
        
        new_rows = []
        for _, row in df.iterrows():
            row_hash = hash(tuple(str(val) for val in row.values))
            if row_hash not in existing_hashes:
                new_rows.append(row)
        
        if new_rows:
            return pd.DataFrame(new_rows).reset_index(drop=True)
        else:
            return pd.DataFrame(columns=df.columns)
    
    def standardize_columns(self, dataframes):
        """Standardize columns across multiple dataframes for combining."""
        if not dataframes:
            return []
        
        # Get all unique columns from all dataframes
        all_columns = set()
        for df in dataframes:
            all_columns.update(df.columns)
        
        all_columns = sorted(list(all_columns))
        
        # Ensure all dataframes have the same columns
        standardized_dfs = []
        for df in dataframes:
            # Add missing columns with NaN values
            for col in all_columns:
                if col not in df.columns:
                    df[col] = pd.NA
            
            # Reorder columns to match the standard order
            df = df[all_columns]
            standardized_dfs.append(df)
        
        return standardized_dfs
    
    def ingest_csv_file(self, file_path, folder_type):
        """Ingest a single CSV file into the database."""
        try:
            filename = os.path.basename(file_path)
            
            logger.info(f"Processing file: {filename}")
            
            # Read CSV file with first row as column names
            df = pd.read_csv(file_path, header=0)
            
            if df.empty:
                logger.warning(f"File {filename} is empty, skipping...")
                return None
            
            # Clean column names
            df = self.clean_column_names(df)
            
            # Add source file column for patient data to track which file data came from
            if folder_type == 'patient_data':
                df['source_file'] = filename
            
            logger.info(f"Successfully read {len(df)} rows from {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    def ingest_patient_data_folder(self, folder_path):
        """Ingest all CSV files from patient_data folder into a single patient_data table."""
        if not os.path.exists(folder_path):
            logger.warning(f"Folder {folder_path} does not exist, skipping...")
            return
        
        csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"No CSV files found in {folder_path}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files in patient_data folder")
        
        # Read all CSV files
        all_dataframes = []
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            df = self.ingest_csv_file(file_path, 'patient_data')
            if df is not None:
                all_dataframes.append(df)
        
        if not all_dataframes:
            logger.warning("No valid CSV files found in patient_data folder")
            return
        
        # Standardize columns across all dataframes
        standardized_dfs = self.standardize_columns(all_dataframes)
        
        # Combine all dataframes
        combined_df = pd.concat(standardized_dfs, ignore_index=True)
        
        logger.info(f"Combined {len(all_dataframes)} files into {len(combined_df)} total rows")
        
        # Get existing data hashes to avoid duplicates
        table_name = 'patient_data'
        existing_hashes = self.get_existing_data_hash(table_name, combined_df)
        
        # Remove duplicates
        df_new = self.remove_duplicates(combined_df, existing_hashes)
        
        if df_new.empty:
            logger.info(f"No new data found in patient_data files, skipping...")
            return
        
        # Insert data into database
        df_new.to_sql(table_name, self.engine, if_exists='append', index=False)
        
        logger.info(f"Successfully ingested {len(df_new)} new rows into patient_data table")
    
    def ingest_stock_data_folder(self, folder_path):
        """Ingest all CSV files from stock_data folder as separate tables."""
        if not os.path.exists(folder_path):
            logger.warning(f"Folder {folder_path} does not exist, skipping...")
            return
        
        csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"No CSV files found in {folder_path}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files in stock_data folder")
        
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            df = self.ingest_csv_file(file_path, 'stock_data')
            
            if df is None:
                continue
            
            # Create table name from filename
            table_name = self.filename_to_table_name(csv_file)
            
            # Get existing data hashes to avoid duplicates
            existing_hashes = self.get_existing_data_hash(table_name, df)
            
            # Remove duplicates
            df_new = self.remove_duplicates(df, existing_hashes)
            
            if df_new.empty:
                logger.info(f"No new data found in {csv_file}, skipping...")
                continue
            
            # Insert data into database
            df_new.to_sql(table_name, self.engine, if_exists='append', index=False)
            
            logger.info(f"Successfully ingested {len(df_new)} new rows from {csv_file} into {table_name}")
    
    def ingest_all_data(self):
        """Ingest data from both patient_data and stock_data folders."""
        logger.info("Starting data ingestion process...")
        
        # Define folder paths relative to project root
        patient_data_path = '../data/patient_data/'
        stock_data_path = '../data/stock_data/'
        
        # Process patient_data folder (combine into single table)
        logger.info("Processing patient_data folder...")
        self.ingest_patient_data_folder(patient_data_path)
        
        # Process stock_data folder (separate tables)
        logger.info("Processing stock_data folder...")
        self.ingest_stock_data_folder(stock_data_path)
        
        logger.info("Data ingestion process completed!")
    
    def get_table_info(self):
        """Get information about all tables in the database."""
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ))
            tables = [row[0] for row in result.fetchall()]
            
            table_info = {}
            for table in tables:
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = count_result.fetchone()[0]
                table_info[table] = count
            
            return table_info
    
    def cleanup_old_patient_tables(self):
        """Remove old individual patient data tables that are now combined into patient_data."""
        logger.info("Checking for old patient data tables to cleanup...")
        
        # Get all table names
        table_info = self.get_table_info()
        
        # Identify tables that look like they might be old patient data tables
        # (tables that start with 'table_' followed by a year)
        old_patient_tables = []
        for table_name in table_info.keys():
            # Check if table name matches pattern like 'table_2020', 'table_2021', etc.
            if re.match(r'^table_\d{4}$', table_name):
                old_patient_tables.append(table_name)
        
        if old_patient_tables:
            print(f"\nFound potential old patient data tables: {old_patient_tables}")
            response = input("Do you want to delete these old tables? (y/N): ").strip().lower()
            
            if response == 'y':
                with self.engine.connect() as conn:
                    for table_name in old_patient_tables:
                        try:
                            conn.execute(text(f"DROP TABLE {table_name}"))
                            conn.commit()
                            logger.info(f"Deleted old table: {table_name}")
                        except Exception as e:
                            logger.error(f"Error deleting table {table_name}: {e}")
                
                print("Old patient data tables have been cleaned up.")
            else:
                print("Old tables kept. You can manually delete them later if needed.")
        else:
            logger.info("No old patient data tables found to cleanup.")

# Main execution
if __name__ == "__main__":
    # Initialize data ingestion with correct paths
    # This assumes the script is run from /diagnostics_ai/src/ directory
    ingestion = DataIngestion()
    
    # Ingest all data
    ingestion.ingest_all_data()
    
    # Display table information
    table_info = ingestion.get_table_info()
    print("\n" + "="*50)
    print("DATABASE SUMMARY")
    print("="*50)
    for table_name, row_count in table_info.items():
        print(f"{table_name}: {row_count} rows")
    print("="*50)
    
    # Offer to cleanup old patient data tables
    ingestion.cleanup_old_patient_tables()