import pandas as pd
import sqlite3
import streamlit as st
import os
from datetime import datetime
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataUploader:
    def __init__(self, db_path="db/diagnostics.db"):
        """Initialize the CSV data uploader."""
        self.db_path = db_path
        
        # Define table configurations based on your existing database structure
        self.table_configs = {
            "Patient Data (2018)": {
                "table_name": "table_2018",
                "required_columns": ["case_no", "opd_dt", "patient", "doctor", "rpt_name", "city", "total_amt"],
                "optional_columns": ["discount", "net_amt", "paid_amt"],
                "description": "Patient appointments and consultations for 2018",
                "sample_format": "case_no,opd_dt,patient,doctor,rpt_name,city,total_amt,discount,net_amt,paid_amt"
            },
            "Patient Data (2019)": {
                "table_name": "table_2019",
                "required_columns": ["case_no", "opd_dt", "patient", "doctor", "rpt_name", "city", "total_amt"],
                "optional_columns": ["discount", "net_amt", "paid_amt"],
                "description": "Patient appointments and consultations for 2019",
                "sample_format": "case_no,opd_dt,patient,doctor,rpt_name,city,total_amt,discount,net_amt,paid_amt"
            },
            "Patient Data (2020)": {
                "table_name": "table_2020",
                "required_columns": ["case_no", "opd_dt", "patient", "doctor", "rpt_name", "city", "total_amt"],
                "optional_columns": ["discount", "net_amt", "paid_amt"],
                "description": "Patient appointments and consultations for 2020",
                "sample_format": "case_no,opd_dt,patient,doctor,rpt_name,city,total_amt,discount,net_amt,paid_amt"
            },
            "Patient Data (2021)": {
                "table_name": "table_2021",
                "required_columns": ["case_no", "opd_dt", "patient", "doctor", "rpt_name", "city", "total_amt"],
                "optional_columns": ["discount", "net_amt", "paid_amt"],
                "description": "Patient appointments and consultations for 2021",
                "sample_format": "case_no,opd_dt,patient,doctor,rpt_name,city,total_amt,discount,net_amt,paid_amt"
            },
            "Patient Data (2022)": {
                "table_name": "table_2022",
                "required_columns": ["case_no", "opd_dt", "patient", "doctor", "rpt_name", "city", "total_amt"],
                "optional_columns": ["discount", "net_amt", "paid_amt"],
                "description": "Patient appointments and consultations for 2022",
                "sample_format": "case_no,opd_dt,patient,doctor,rpt_name,city,total_amt,discount,net_amt,paid_amt"
            },
            "Patient Data (2023)": {
                "table_name": "table_2023",
                "required_columns": ["case_no", "opd_dt", "patient", "doctor", "rpt_name", "city", "total_amt"],
                "optional_columns": ["discount", "net_amt", "paid_amt"],
                "description": "Patient appointments and consultations for 2023",
                "sample_format": "case_no,opd_dt,patient,doctor,rpt_name,city,total_amt,discount,net_amt,paid_amt"
            },
            "Patient Data (2024)": {
                "table_name": "table_2024",
                "required_columns": ["case_no", "opd_dt", "patient", "doctor", "rpt_name", "city", "total_amt"],
                "optional_columns": ["discount", "net_amt", "paid_amt"],
                "description": "Patient appointments and consultations for 2024",
                "sample_format": "case_no,opd_dt,patient,doctor,rpt_name,city,total_amt,discount,net_amt,paid_amt"
            },
            "Patient Data (2025)": {
                "table_name": "table_2025",
                "required_columns": ["case_no", "opd_dt", "patient", "doctor", "rpt_name", "city", "total_amt"],
                "optional_columns": ["discount", "net_amt", "paid_amt"],
                "description": "Patient appointments and consultations for 2025",
                "sample_format": "case_no,opd_dt,patient,doctor,rpt_name,city,total_amt,discount,net_amt,paid_amt"
            },
            "Stock Update": {
                "table_name": "stock_update",
                "required_columns": ["item", "unit", "closing_stock"],
                "optional_columns": ["opening", "receipt", "total", "issue"],
                "description": "Current inventory stock levels",
                "sample_format": "item,unit,opening,receipt,total,issue,closing_stock"
            },
            "Item Stock (One Page)": {
                "table_name": "item_stock_one_page",
                "required_columns": ["item", "unit"],
                "optional_columns": ["opening", "receipt", "issue", "closing_stock"],
                "description": "Item stock summary page",
                "sample_format": "item,unit,opening,receipt,issue,closing_stock"
            },
            "Purchase Order Register": {
                "table_name": "purchase_order_register_sum",
                "required_columns": ["v_no", "date", "party", "city", "total_amt"],
                "optional_columns": ["description", "items"],
                "description": "Purchase order register summary",
                "sample_format": "v_no,date,party,city,total_amt,description"
            },
            "Purchase Challan Register": {
                "table_name": "purchase_challan_register_d", 
                "required_columns": ["chln_no", "chln_date", "party_name", "item_name", "tr_qty"],
                "optional_columns": ["document_name", "po_no", "po_date", "party_chln_no", "party_chln_date", "sr_no", "uom", "batch_no", "tr_rate"],
                "description": "Purchase challan register details",
                "sample_format": "document_name,chln_no,chln_date,po_no,po_date,party_chln_no,party_chln_date,party_name,sr_no,item_name,uom,batch_no,tr_qty,tr_rate"
            },
            "Purchase Register Summary": {
                "table_name": "purchase_register_summary",
                "required_columns": ["v_no", "date", "party", "city", "total_amt"],
                "optional_columns": ["items", "description"],
                "description": "Purchase register summary",
                "sample_format": "v_no,date,party,city,total_amt,items"
            },
            "Requisition Detail Report": {
                "table_name": "requitition_detail_report",
                "required_columns": ["item", "quantity"],
                "optional_columns": ["date", "department", "status"],
                "description": "Requisition detail report",
                "sample_format": "item,quantity,date,department,status"
            }
        }
    
    def get_existing_tables(self):
        """Get list of existing tables in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            return tables
        except Exception as e:
            logger.error(f"Error getting existing tables: {e}")
            return []
    
    def validate_csv_data(self, df, table_type):
        """Validate uploaded CSV data against table requirements."""
        config = self.table_configs[table_type]
        required_cols = config["required_columns"]
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check if DataFrame is empty
        if df.empty:
            validation_results["is_valid"] = False
            validation_results["errors"].append("CSV file is empty")
            return validation_results
        
        # Clean column names for comparison
        df_columns = [col.lower().strip().replace(' ', '_').replace('.', '_').replace('-', '_') 
                     for col in df.columns]
        df_columns = [re.sub(r'[^a-z0-9_]', '', col) for col in df_columns]
        
        # Check required columns
        missing_cols = []
        for req_col in required_cols:
            if req_col not in df_columns:
                missing_cols.append(req_col)
        
        if missing_cols:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Missing required columns: {missing_cols}")
        
        # Data quality checks
        validation_results["stats"]["total_rows"] = len(df)
        validation_results["stats"]["columns_found"] = len(df.columns)
        validation_results["stats"]["null_percentage"] = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            validation_results["warnings"].append(f"Found {empty_rows} completely empty rows (will be removed)")
        
        # Specific validations based on table type
        if "Patient Data" in table_type:
            # Check date format
            if 'opd_dt' in df.columns:
                invalid_dates = 0
                for date_val in df['opd_dt'].dropna():
                    try:
                        # Try different date formats
                        pd.to_datetime(str(date_val), format='%d/%m/%Y', errors='raise')
                    except:
                        try:
                            pd.to_datetime(str(date_val), format='%Y-%m-%d', errors='raise')
                        except:
                            invalid_dates += 1
                
                if invalid_dates > 0:
                    validation_results["warnings"].append(f"Found {invalid_dates} rows with invalid date formats")
            
            # Check amount columns
            amount_cols = ['total_amt', 'net_amt', 'paid_amt']
            for col in amount_cols:
                if col in df.columns:
                    non_numeric = pd.to_numeric(df[col], errors='coerce').isnull().sum()
                    if non_numeric > 0:
                        validation_results["warnings"].append(f"Found {non_numeric} non-numeric values in {col}")
        
        elif "Stock" in table_type:
            # Check stock quantities
            stock_cols = ['closing_stock', 'opening', 'receipt', 'total', 'issue']
            for col in stock_cols:
                if col in df.columns:
                    negative_stock = (pd.to_numeric(df[col], errors='coerce') < 0).sum()
                    if negative_stock > 0:
                        validation_results["warnings"].append(f"Found {negative_stock} items with negative values in {col}")
        
        return validation_results
    
    def clean_csv_data(self, df, table_type):
        """Clean and prepare CSV data for database insertion."""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Standardize column names to match database schema
        df.columns = [col.lower().strip().replace(' ', '_').replace('.', '_').replace('-', '_') 
                     for col in df.columns]
        df.columns = [re.sub(r'[^a-z0-9_]', '', col) for col in df.columns]
        
        # Handle empty column names
        df.columns = [f'unnamed_{i}' if not col else col for i, col in enumerate(df.columns)]
        
        # Table-specific cleaning
        if "Patient Data" in table_type:
            # Standardize date format
            if 'opd_dt' in df.columns:
                df['opd_dt'] = df['opd_dt'].apply(self._standardize_date)
            
            # Clean amount columns
            amount_cols = ['total_amt', 'discount', 'net_amt', 'paid_amt']
            for col in amount_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        elif "Stock" in table_type:
            # Clean numeric columns
            numeric_cols = ['opening', 'receipt', 'total', 'issue', 'closing_stock']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        elif "Purchase" in table_type:
            # Clean amount column
            if 'total_amt' in df.columns:
                df['total_amt'] = pd.to_numeric(df['total_amt'], errors='coerce').fillna(0)
            
            # Clean date columns
            date_cols = ['date', 'chln_date', 'po_date', 'party_chln_date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = df[col].apply(self._standardize_date)
        
        elif "Challan" in table_type:
            # Clean quantity and rate columns
            if 'tr_qty' in df.columns:
                df['tr_qty'] = pd.to_numeric(df['tr_qty'], errors='coerce').fillna(0)
            if 'tr_rate' in df.columns:
                df['tr_rate'] = pd.to_numeric(df['tr_rate'], errors='coerce').fillna(0)
        
        return df
    
    def _standardize_date(self, date_value):
        """Standardize date format to YYYY-MM-DD."""
        if pd.isna(date_value):
            return None
        
        date_str = str(date_value).strip()
        
        # Try different date formats
        formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return date_str  # Return original if no format matches
    
    def check_table_exists(self, table_name):
        """Check if table exists in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def get_existing_data_hash(self, table_name, df):
        """Get hash of existing data to identify duplicates."""
        if not self.check_table_exists(table_name):
            return set()
        
        try:
            conn = sqlite3.connect(self.db_path)
            existing_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            
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
    
    def insert_data_to_db(self, df, table_type):
        """Insert cleaned data into the database."""
        try:
            config = self.table_configs[table_type]
            table_name = config["table_name"]
            
            # Check if table exists
            if not self.check_table_exists(table_name):
                logger.warning(f"Table {table_name} does not exist. Creating new table.")
            
            # Get existing data hashes to avoid duplicates
            existing_hashes = self.get_existing_data_hash(table_name, df)
            
            # Remove duplicates
            df_new = self.remove_duplicates(df, existing_hashes)
            
            if df_new.empty:
                logger.info(f"No new data found, all rows already exist in {table_name}")
                return True, f"No new data to upload - all {len(df)} rows already exist in {table_name}"
            
            conn = sqlite3.connect(self.db_path)
            
            # Insert data
            df_new.to_sql(table_name, conn, if_exists='append', index=False)
            
            conn.close()
            
            logger.info(f"Successfully inserted {len(df_new)} new rows into {table_name}")
            duplicate_count = len(df) - len(df_new)
            message = f"Successfully uploaded {len(df_new)} new rows to {table_name}"
            if duplicate_count > 0:
                message += f" (skipped {duplicate_count} duplicates)"
            
            return True, message
        
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return False, f"Error uploading data: {str(e)}"
    
    def get_table_info(self, table_type):
        """Get information about a specific table type."""
        return self.table_configs.get(table_type, {})
    
    def get_available_tables(self):
        """Get list of available table types for upload."""
        existing_tables = self.get_existing_tables()
        available_configs = []
        
        for table_type, config in self.table_configs.items():
            if config["table_name"] in existing_tables:
                available_configs.append(table_type)
        
        return available_configs

def render_data_upload_page():
    """Main function to render the data upload page."""
    st.markdown('<div class="tab-header">📤 Data Upload Manager</div>', unsafe_allow_html=True)
    
    # Initialize uploader
    uploader = CSVDataUploader()
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    📋 <strong>Upload CSV Data:</strong> Select the type of data you want to upload and follow the format requirements.
    Data will be automatically validated and added to the appropriate database table.
    </div>
    """, unsafe_allow_html=True)
    
    # Check database connection
    existing_tables = uploader.get_existing_tables()
    if not existing_tables:
        st.error("❌ Cannot connect to database or no tables found. Please check your database setup.")
        return
    
    # Step 1: Select data type
    st.markdown("### 1️⃣ Select Data Type")
    
    available_tables = uploader.get_available_tables()
    if not available_tables:
        st.warning("⚠️ No matching table configurations found for existing database tables.")
        st.info(f"Available database tables: {', '.join(existing_tables)}")
        return
    
    selected_table = st.selectbox(
        "Choose the type of data you want to upload:",
        available_tables,
        help="Select the appropriate data type based on your CSV content"
    )
    
    # Show table information
    table_info = uploader.get_table_info(selected_table)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Description:** {table_info['description']}")
        st.info(f"**Database Table:** {table_info['table_name']}")
    with col2:
        with st.expander("📋 Required Format"):
            st.code(table_info['sample_format'], language='csv')
            st.markdown(f"**Required columns:** {', '.join(table_info['required_columns'])}")
            st.markdown(f"**Optional columns:** {', '.join(table_info['optional_columns'])}")
    
    st.markdown("---")
    
    # Step 2: File upload
    st.markdown("### 2️⃣ Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help=f"Upload a CSV file containing {selected_table.lower()} data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Step 3: Data preview
            st.markdown("### 3️⃣ Data Preview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Show data preview
            st.markdown("**Data Preview (first 5 rows):**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Step 4: Validation
            st.markdown("### 4️⃣ Data Validation")
            
            with st.spinner("Validating data..."):
                validation = uploader.validate_csv_data(df, selected_table)
            
            if validation["is_valid"]:
                st.success("✅ Data validation passed!")
                
                # Show stats
                stats = validation["stats"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Valid Rows", stats["total_rows"])
                with col2:
                    st.metric("Columns Found", stats["columns_found"])
                with col3:
                    st.metric("Data Quality", f"{100 - stats['null_percentage']:.1f}%")
                
                # Show warnings if any
                if validation["warnings"]:
                    for warning in validation["warnings"]:
                        st.warning(f"⚠️ {warning}")
                
                # Step 5: Upload confirmation
                st.markdown("### 5️⃣ Upload to Database")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    upload_button = st.button("🚀 Upload Data", type="primary")
                with col2:
                    st.info("This will add the data to your existing database table (duplicates will be skipped).")
                
                if upload_button:
                    with st.spinner("Uploading data to database..."):
                        # Clean data
                        cleaned_df = uploader.clean_csv_data(df, selected_table)
                        
                        # Insert to database
                        success, message = uploader.insert_data_to_db(cleaned_df, selected_table)
                        
                        if success:
                            st.success(f"🎉 {message}")
                            st.balloons()
                            
                            # Show what was uploaded
                            with st.expander("📊 Upload Summary"):
                                st.markdown(f"**Table:** {table_info['table_name']}")
                                st.markdown(f"**Original Rows:** {len(df)}")
                                st.markdown(f"**Processed Rows:** {len(cleaned_df)}")
                                st.markdown(f"**Upload Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                
                                if len(cleaned_df) <= 10:
                                    st.markdown("**Processed Data:**")
                                    st.dataframe(cleaned_df, use_container_width=True)
                        else:
                            st.error(f"❌ {message}")
            
            else:
                st.error("❌ Data validation failed!")
                for error in validation["errors"]:
                    st.error(f"🚫 {error}")
                
                st.markdown("**💡 Fix these issues and try uploading again:**")
                st.markdown("1. Check that all required columns are present")
                st.markdown("2. Verify column names match the expected format")
                st.markdown("3. Ensure data types are correct (dates, numbers)")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please check that your file is a valid CSV format.")
    
    else:
        st.info("👆 Please upload a CSV file to continue")
    
    # Database info section
    with st.expander("🗄️ Database Information"):
        st.markdown("**Available Tables in Database:**")
        for table in existing_tables:
            try:
                conn = sqlite3.connect(uploader.db_path)
                count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
                conn.close()
                st.markdown(f"- **{table}**: {count} rows")
            except:
                st.markdown(f"- **{table}**: Unable to count rows")
    
    # Help section
    with st.expander("❓ Need Help?"):
        st.markdown("""
        **Common Issues:**
        - **Column names don't match:** Ensure your CSV headers match the required format
        - **Date format errors:** Use DD/MM/YYYY or YYYY-MM-DD format for dates
        - **Missing data:** Required columns cannot be empty
        - **File too large:** Consider splitting large files into smaller chunks
        
        **Tips:**
        - Use the sample format as a template for your CSV files
        - Test with a small file first before uploading large datasets
        - Check your data in Excel/Google Sheets before uploading
        - Duplicate data will be automatically skipped
        """)

if __name__ == "__main__":
    # For testing
    render_data_upload_page()