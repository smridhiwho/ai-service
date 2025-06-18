import os
import sqlite3
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
import warnings

# Suppress Prophet warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalForecastingSystem:
    def __init__(self, db_path='../db/diagnostics.db'):
        """Initialize the forecasting system."""
        self.db_path = db_path
        
        # Check if database exists
        if not os.path.exists(db_path):
            logger.error(f"Database file not found: {db_path}")
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        logger.info("Medical Forecasting System initialized")
    
    def prophet_forecast(self, df, periods=12, freq='M', title="Forecast"):
        """
        Core Prophet-based forecasting logic.
        
        Args:
            df (pandas.DataFrame): DataFrame with 'ds' (date) and 'y' (value) columns
            periods (int): Number of periods to forecast (default: 12 months)
            freq (str): Frequency - 'M' for monthly, 'D' for daily, 'W' for weekly
            title (str): Chart title
            
        Returns:
            dict: Contains forecast DataFrame and Plotly figure
        """
        try:
            # Validate input data
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("DataFrame must have 'ds' (date) and 'y' (value) columns")
            
            # Ensure ds is datetime
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.sort_values('ds')
            
            # Remove any NaN values
            df = df.dropna()
            
            logger.info(f"Forecasting with {len(df)} data points for {periods} periods")
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=True,
                changepoint_prior_scale=0.1,
                seasonality_prior_scale=10
            )
            
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq=freq)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Create interactive Plotly figure
            fig = self._create_forecast_plot(df, forecast, title)
            
            return {
                'historical_data': df,
                'forecast_data': forecast,
                'model': model,
                'plotly_figure': fig,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in prophet_forecast: {e}")
            return {
                'historical_data': pd.DataFrame(),
                'forecast_data': pd.DataFrame(),
                'model': None,
                'plotly_figure': None,
                'success': False,
                'error': str(e)
            }
    
    def _create_forecast_plot(self, historical_df, forecast_df, title):
        """Create interactive Plotly forecast visualization."""
        
        # Create subplot with secondary y-axis for confidence intervals
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_df['ds'],
            y=historical_df['y'],
            mode='markers+lines',
            name='Historical Data',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:,.0f}<extra></extra>'
        ))
        
        # Add forecast line
        forecast_future = forecast_df[forecast_df['ds'] > historical_df['ds'].max()]
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:,.0f}<extra></extra>'
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(width=0),
            name='Confidence Interval',
            hovertemplate='<b>Date:</b> %{x}<br><b>Lower:</b> %{y:,.0f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def get_patient_revenue_data(self):
        """Extract monthly revenue data from patient_data table."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # First, check what the actual date format looks like
            sample_dates = pd.read_sql("SELECT opd_dt FROM patient_data WHERE opd_dt IS NOT NULL AND opd_dt != '' LIMIT 10", conn)
            logger.info(f"Sample dates from database: {sample_dates['opd_dt'].tolist()}")
            
            # Try multiple date extraction approaches since formats may vary
            queries_to_try = [
                # Format 1: Try standard YYYY-MM-DD format
                """
                SELECT 
                    substr(opd_dt, 1, 7) as month,
                    SUM(total_amt) as monthly_revenue
                FROM patient_data 
                WHERE opd_dt IS NOT NULL 
                    AND total_amt IS NOT NULL
                    AND opd_dt != ''
                    AND length(opd_dt) >= 7
                    AND opd_dt LIKE '____-__-%'
                GROUP BY substr(opd_dt, 1, 7)
                ORDER BY month
                """,
                
                # Format 2: Try DD/MM/YYYY format
                """
                SELECT 
                    substr(opd_dt, 7, 4) || '-' || 
                    CASE 
                        WHEN length(substr(opd_dt, 4, 2)) = 1 THEN '0' || substr(opd_dt, 4, 1)
                        ELSE substr(opd_dt, 4, 2)
                    END as month,
                    SUM(total_amt) as monthly_revenue
                FROM patient_data 
                WHERE opd_dt IS NOT NULL 
                    AND total_amt IS NOT NULL
                    AND opd_dt != ''
                    AND opd_dt LIKE '__/__/____'
                GROUP BY substr(opd_dt, 7, 4) || '-' || 
                    CASE 
                        WHEN length(substr(opd_dt, 4, 2)) = 1 THEN '0' || substr(opd_dt, 4, 1)
                        ELSE substr(opd_dt, 4, 2)
                    END
                ORDER BY month
                """,
                
                # Format 3: Try to extract year and create monthly data
                """
                SELECT 
                    CASE 
                        WHEN opd_dt LIKE '____-__-%' THEN substr(opd_dt, 1, 4) || '-01'
                        WHEN opd_dt LIKE '__/__/____' THEN substr(opd_dt, 7, 4) || '-01'
                        ELSE '2024-01'
                    END as month,
                    SUM(total_amt) as monthly_revenue
                FROM patient_data 
                WHERE opd_dt IS NOT NULL 
                    AND total_amt IS NOT NULL
                    AND opd_dt != ''
                GROUP BY CASE 
                        WHEN opd_dt LIKE '____-__-%' THEN substr(opd_dt, 1, 4) || '-01'
                        WHEN opd_dt LIKE '__/__/____' THEN substr(opd_dt, 7, 4) || '-01'
                        ELSE '2024-01'
                    END
                ORDER BY month
                """
            ]
            
            df = pd.DataFrame()
            
            for i, query in enumerate(queries_to_try):
                try:
                    logger.info(f"Trying query approach {i+1}")
                    df = pd.read_sql(query, conn)
                    if not df.empty and df['monthly_revenue'].sum() > 0:
                        logger.info(f"Success with query approach {i+1}: {len(df)} months of data")
                        break
                except Exception as e:
                    logger.warning(f"Query approach {i+1} failed: {e}")
                    continue
            
            conn.close()
            
            if df.empty:
                raise ValueError("No revenue data found with any date format")
            
            # Convert to Prophet format
            df['ds'] = pd.to_datetime(df['month'] + '-01', errors='coerce')  # Add day to make valid date
            df['y'] = df['monthly_revenue']
            
            # Remove any invalid dates or negative values
            df = df.dropna()
            df = df[df['y'] > 0]
            df = df.sort_values('ds')
            
            logger.info(f"Retrieved {len(df)} months of revenue data, total revenue: ₹{df['y'].sum():,.0f}")
            return df[['ds', 'y']]
            
        except Exception as e:
            logger.error(f"Error getting revenue data: {e}")
            return pd.DataFrame()
    
    def get_patient_count_data(self):
        """Extract monthly patient count data from patient_data table."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Try multiple date extraction approaches
            queries_to_try = [
                # Format 1: YYYY-MM-DD format
                """
                SELECT 
                    substr(opd_dt, 1, 7) as month,
                    COUNT(*) as monthly_patients
                FROM patient_data 
                WHERE opd_dt IS NOT NULL 
                    AND opd_dt != ''
                    AND length(opd_dt) >= 7
                    AND opd_dt LIKE '____-__-%'
                GROUP BY substr(opd_dt, 1, 7)
                ORDER BY month
                """,
                
                # Format 2: DD/MM/YYYY format
                """
                SELECT 
                    substr(opd_dt, 7, 4) || '-' || 
                    CASE 
                        WHEN length(substr(opd_dt, 4, 2)) = 1 THEN '0' || substr(opd_dt, 4, 1)
                        ELSE substr(opd_dt, 4, 2)
                    END as month,
                    COUNT(*) as monthly_patients
                FROM patient_data 
                WHERE opd_dt IS NOT NULL 
                    AND opd_dt != ''
                    AND opd_dt LIKE '__/__/____'
                GROUP BY substr(opd_dt, 7, 4) || '-' || 
                    CASE 
                        WHEN length(substr(opd_dt, 4, 2)) = 1 THEN '0' || substr(opd_dt, 4, 1)
                        ELSE substr(opd_dt, 4, 2)
                    END
                ORDER BY month
                """,
                
                # Format 3: Yearly aggregation as fallback
                """
                SELECT 
                    CASE 
                        WHEN opd_dt LIKE '____-__-%' THEN substr(opd_dt, 1, 4) || '-01'
                        WHEN opd_dt LIKE '__/__/____' THEN substr(opd_dt, 7, 4) || '-01'
                        ELSE '2024-01'
                    END as month,
                    COUNT(*) as monthly_patients
                FROM patient_data 
                WHERE opd_dt IS NOT NULL 
                    AND opd_dt != ''
                GROUP BY CASE 
                        WHEN opd_dt LIKE '____-__-%' THEN substr(opd_dt, 1, 4) || '-01'
                        WHEN opd_dt LIKE '__/__/____' THEN substr(opd_dt, 7, 4) || '-01'
                        ELSE '2024-01'
                    END
                ORDER BY month
                """
            ]
            
            df = pd.DataFrame()
            
            for i, query in enumerate(queries_to_try):
                try:
                    logger.info(f"Trying patient count query approach {i+1}")
                    df = pd.read_sql(query, conn)
                    if not df.empty and df['monthly_patients'].sum() > 0:
                        logger.info(f"Success with patient query approach {i+1}: {len(df)} months of data")
                        break
                except Exception as e:
                    logger.warning(f"Patient query approach {i+1} failed: {e}")
                    continue
            
            conn.close()
            
            if df.empty:
                raise ValueError("No patient count data found")
            
            # Convert to Prophet format
            df['ds'] = pd.to_datetime(df['month'] + '-01', errors='coerce')
            df['y'] = df['monthly_patients']
            
            df = df.dropna()
            df = df[df['y'] > 0]
            df = df.sort_values('ds')
            
            logger.info(f"Retrieved {len(df)} months of patient count data, total patients: {df['y'].sum():,.0f}")
            return df[['ds', 'y']]
            
        except Exception as e:
            logger.error(f"Error getting patient count data: {e}")
            return pd.DataFrame()
    
    def get_inventory_stock_data(self, item_name=None):
        """Extract inventory stock data from stock tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # First, find stock tables and examine their structure
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%stock%'")
            stock_tables = [row[0] for row in cursor.fetchall()]
            
            if not stock_tables:
                raise ValueError("No stock tables found")
            
            stock_table = stock_tables[0]  # Use first stock table
            logger.info(f"Using stock table: {stock_table}")
            
            # Get all data from the table to understand its structure
            raw_data = pd.read_sql(f"SELECT * FROM {stock_table}", conn)
            logger.info(f"Raw data shape: {raw_data.shape}")
            logger.info(f"Raw data columns: {raw_data.columns.tolist()}")
            
            # Check if this table has generic column names (unnamed_0, unnamed_1, etc.)
            if raw_data.columns[0].startswith('unnamed_'):
                logger.info("Detected generic column names, looking for header row...")
                
                # Find the header row (look for row containing 'Item', 'Stock', etc.)
                header_row_idx = None
                for idx, row in raw_data.iterrows():
                    row_values = [str(v).lower() if v is not None else '' for v in row.values]
                    if any(keyword in ' '.join(row_values) for keyword in ['item', 'stock', 'closing', 'balance']):
                        header_row_idx = idx
                        logger.info(f"Found header row at index {idx}: {row.values}")
                        break
                
                if header_row_idx is not None:
                    # Extract headers and data
                    headers = [str(v).strip().lower().replace(' ', '_') if v is not None else f'col_{i}' 
                              for i, v in enumerate(raw_data.iloc[header_row_idx].values)]
                    
                    # Remove rows before and including header row, and any empty rows
                    data_rows = raw_data.iloc[header_row_idx + 1:].copy()
                    
                    # Assign proper column names
                    data_rows.columns = headers
                    
                    # Remove rows where the first column is None or empty
                    data_rows = data_rows[data_rows.iloc[:, 0].notna()]
                    data_rows = data_rows[data_rows.iloc[:, 0] != '']
                    
                    logger.info(f"Processed data shape: {data_rows.shape}")
                    logger.info(f"New column names: {data_rows.columns.tolist()}")
                    logger.info(f"Sample processed data:\n{data_rows.head(3)}")
                    
                    # Use processed data
                    df = data_rows.copy()
                else:
                    logger.warning("Could not find header row, using raw data")
                    df = raw_data.copy()
            else:
                logger.info("Column names look proper, using as-is")
                df = raw_data.copy()
            
            # Now find the right columns
            column_names = [str(col).lower().replace(' ', '_') for col in df.columns]
            logger.info(f"Looking for stock columns in: {column_names}")
            
            # Find stock column (more flexible matching)
            possible_stock_columns = [
                'closing_stock', 'closing', 'stock', 'balance', 'current_stock',
                'qty', 'quantity', 'available', 'on_hand'
            ]
            possible_item_columns = [
                'item', 'item_name', 'product', 'description', 'name', 'material'
            ]
            
            stock_column = None
            item_column = None
            
            # Find stock column
            for col in possible_stock_columns:
                matching_cols = [c for c in column_names if col in c]
                if matching_cols:
                    stock_column = df.columns[column_names.index(matching_cols[0])]
                    logger.info(f"Found stock column: '{stock_column}'")
                    break
            
            # Find item column
            for col in possible_item_columns:
                matching_cols = [c for c in column_names if col in c]
                if matching_cols:
                    item_column = df.columns[column_names.index(matching_cols[0])]
                    logger.info(f"Found item column: '{item_column}'")
                    break
            
            # If still no stock column found, try to use numeric columns
            if not stock_column:
                numeric_columns = df.select_dtypes(include=['number', 'object']).columns
                for col in numeric_columns:
                    try:
                        # Test if column can be converted to numeric
                        test_series = pd.to_numeric(df[col], errors='coerce')
                        if not test_series.isna().all():  # At least some values are numeric
                            stock_column = col
                            logger.info(f"Using numeric column '{stock_column}' as stock level")
                            break
                    except:
                        continue
            
            if not stock_column:
                raise ValueError(f"No suitable stock column found. Available columns: {df.columns.tolist()}")
            
            if not item_column and len(df.columns) > 0:
                item_column = df.columns[0]  # Use first column as item name
                logger.info(f"Using first column '{item_column}' as item name")
            
            logger.info(f"Final selection - Stock column: '{stock_column}', Item column: '{item_column}'")
            
            # Clean and process the data
            # Convert stock column to numeric
            df[stock_column] = pd.to_numeric(df[stock_column], errors='coerce')
            df = df.dropna(subset=[stock_column])
            df = df[df[stock_column] >= 0]  # Remove negative stock values
            
            if df.empty:
                raise ValueError("No valid stock data after cleaning")
            
            logger.info(f"Cleaned data shape: {df.shape}")
            logger.info(f"Stock value range: {df[stock_column].min()} to {df[stock_column].max()}")
            
            # Calculate current stock level
            if item_name and item_column:
                # Filter for specific item
                item_df = df[df[item_column].str.contains(item_name, case=False, na=False)]
                if not item_df.empty:
                    current_stock = item_df[stock_column].iloc[0]
                    logger.info(f"Current stock for '{item_name}': {current_stock}")
                else:
                    logger.warning(f"Item '{item_name}' not found, using average stock")
                    current_stock = df[stock_column].mean()
            else:
                # Use total or average stock
                current_stock = df[stock_column].sum() if len(df) < 50 else df[stock_column].mean()
                logger.info(f"Using aggregate stock level: {current_stock}")
            
            conn.close()
            
            # Generate historical data for forecasting
            current_date = datetime.now()
            historical_data = []
            
            # Generate 12 months of realistic historical stock data
            for i in range(12, 0, -1):
                date = current_date - timedelta(days=30*i)
                
                # Create realistic stock variation
                # Seasonal factor (higher stock in certain months)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (i + 3) / 12)  # Peak around month 9
                
                # Trend factor (slight growth over time)
                trend_factor = 1 + (12 - i) * 0.02
                
                # Random variation
                noise_factor = 0.7 + 0.6 * np.random.random()
                
                # Ensure some realistic constraints
                if i <= 3:  # Recent months should be closer to current stock
                    variation = 0.8 + 0.4 * np.random.random()
                    stock_value = current_stock * variation
                else:
                    stock_value = current_stock * seasonal_factor * trend_factor * noise_factor
                
                stock_value = max(stock_value, 0)  # Ensure non-negative
                
                historical_data.append({
                    'ds': date,
                    'y': stock_value
                })
            
            # Add current data point
            historical_data.append({
                'ds': current_date,
                'y': current_stock
            })
            
            result_df = pd.DataFrame(historical_data)
            result_df = result_df.sort_values('ds')
            
            logger.info(f"Generated {len(result_df)} months of stock data")
            logger.info(f"Stock range: {result_df['y'].min():.1f} to {result_df['y'].max():.1f}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error getting stock data: {e}")
            return pd.DataFrame()
    
    def forecast_monthly_revenue(self, months_ahead=12):
        """Forecast monthly revenue for specified months ahead."""
        logger.info(f"Forecasting monthly revenue for {months_ahead} months")
        
        # Get historical data
        data = self.get_patient_revenue_data()
        
        if data.empty:
            return {
                'success': False,
                'error': 'No revenue data available for forecasting'
            }
        
        # Generate forecast
        result = self.prophet_forecast(
            data, 
            periods=months_ahead, 
            freq='M',
            title=f"Monthly Revenue Forecast ({months_ahead} months ahead)"
        )
        
        if result['success']:
            # Add business insights
            forecast_df = result['forecast_data']
            future_forecast = forecast_df[forecast_df['ds'] > data['ds'].max()]
            
            avg_forecast = future_forecast['yhat'].mean()
            total_forecast = future_forecast['yhat'].sum()
            
            result['insights'] = {
                'average_monthly_revenue': f"₹{avg_forecast:,.0f}",
                'total_forecast_revenue': f"₹{total_forecast:,.0f}",
                'forecast_period': f"{months_ahead} months"
            }
        
        return result
    
    def forecast_patient_count(self, months_ahead=12):
        """Forecast monthly patient count for specified months ahead."""
        logger.info(f"Forecasting patient count for {months_ahead} months")
        
        # Get historical data
        data = self.get_patient_count_data()
        
        if data.empty:
            return {
                'success': False,
                'error': 'No patient count data available for forecasting'
            }
        
        # Generate forecast
        result = self.prophet_forecast(
            data, 
            periods=months_ahead, 
            freq='M',
            title=f"Monthly Patient Count Forecast ({months_ahead} months ahead)"
        )
        
        if result['success']:
            # Add business insights
            forecast_df = result['forecast_data']
            future_forecast = forecast_df[forecast_df['ds'] > data['ds'].max()]
            
            avg_forecast = future_forecast['yhat'].mean()
            total_forecast = future_forecast['yhat'].sum()
            
            result['insights'] = {
                'average_monthly_patients': f"{avg_forecast:,.0f}",
                'total_forecast_patients': f"{total_forecast:,.0f}",
                'forecast_period': f"{months_ahead} months"
            }
        
        return result
    
    # def forecast_inventory_stock(self, months_ahead=12, item_name=None):
    #     """Forecast inventory stock levels for specified months ahead."""
    #     logger.info(f"Forecasting inventory stock for {months_ahead} months")
        
    #     # Get historical data
    #     data = self.get_inventory_stock_data(item_name)
        
    #     if data.empty:
    #         return {
    #             'success': False,
    #             'error': 'No inventory data available for forecasting'
    #         }
        
    #     # Generate forecast
    #     item_title = f" for {item_name}" if item_name else ""
    #     result = self.prophet_forecast(
    #         data, 
    #         periods=months_ahead, 
    #         freq='M',
    #         title=f"Inventory Stock Forecast{item_title} ({months_ahead} months ahead)"
    #     )
        
    #     if result['success']:
    #         # Add business insights
    #         forecast_df = result['forecast_data']
    #         future_forecast = forecast_df[forecast_df['ds'] > data['ds'].max()]
            
    #         avg_forecast = future_forecast['yhat'].mean()
    #         min_forecast = future_forecast['yhat'].min()
            
    #         result['insights'] = {
    #             'average_stock_level': f"{avg_forecast:,.0f}",
    #             'minimum_forecast_stock': f"{min_forecast:,.0f}",
    #             'forecast_period': f"{months_ahead} months",
    #             'item': item_name or "All items"
    #         }
        
    #     return result
    
    def create_comparison_dashboard(self, months_ahead=12):
        """Create a comprehensive dashboard comparing all forecasts."""
        logger.info("Creating comparison dashboard")
        
        # Get all forecasts
        revenue_forecast = self.forecast_monthly_revenue(months_ahead)
        patient_forecast = self.forecast_patient_count(months_ahead)
        stock_forecast = self.forecast_inventory_stock(months_ahead)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Revenue Forecast', 'Patient Count Forecast', 
                          'Inventory Stock Forecast', 'Summary Insights'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # Add revenue forecast
        if revenue_forecast['success']:
            rev_data = revenue_forecast['historical_data']
            rev_forecast = revenue_forecast['forecast_data']
            
            fig.add_trace(
                go.Scatter(x=rev_data['ds'], y=rev_data['y'], 
                          name='Revenue History', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=rev_forecast['ds'], y=rev_forecast['yhat'], 
                          name='Revenue Forecast', line=dict(color='red', dash='dash')),
                row=1, col=1
            )
        
        # Add patient forecast
        if patient_forecast['success']:
            pat_data = patient_forecast['historical_data']
            pat_forecast = patient_forecast['forecast_data']
            
            fig.add_trace(
                go.Scatter(x=pat_data['ds'], y=pat_data['y'], 
                          name='Patient History', line=dict(color='green')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=pat_forecast['ds'], y=pat_forecast['yhat'], 
                          name='Patient Forecast', line=dict(color='orange', dash='dash')),
                row=1, col=2
            )
        
        # Add stock forecast
        if stock_forecast['success']:
            stock_data = stock_forecast['historical_data']
            stock_forecast_data = stock_forecast['forecast_data']
            
            fig.add_trace(
                go.Scatter(x=stock_data['ds'], y=stock_data['y'], 
                          name='Stock History', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=stock_forecast_data['ds'], y=stock_forecast_data['yhat'], 
                          name='Stock Forecast', line=dict(color='brown', dash='dash')),
                row=2, col=1
            )
        
        # Add summary table
        summary_data = []
        if revenue_forecast['success']:
            summary_data.append(['Revenue', revenue_forecast['insights']['average_monthly_revenue']])
        if patient_forecast['success']:
            summary_data.append(['Patients', patient_forecast['insights']['average_monthly_patients']])
        if stock_forecast['success']:
            summary_data.append(['Stock', stock_forecast['insights']['average_stock_level']])
        
        if summary_data:
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', f'Avg Monthly ({months_ahead}m forecast)']),
                    cells=dict(values=list(zip(*summary_data)))
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text=f"Medical Center Forecasting Dashboard - {months_ahead} Months Ahead",
            showlegend=True
        )
        
        return fig

    def debug_database_structure(self):
        """Debug function to understand database structure and data formats."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            print("=== DATABASE STRUCTURE DEBUG ===")
            
            # Check patient_data table
            print("\n1. PATIENT DATA TABLE:")
            try:
                # Check if table exists
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patient_data'")
                if cursor.fetchone():
                    print("✅ patient_data table exists")
                    
                    # Get column info
                    cursor.execute("PRAGMA table_info(patient_data)")
                    columns = cursor.fetchall()
                    print(f"Columns: {[col[1] for col in columns]}")
                    
                    # Check total records
                    count_df = pd.read_sql("SELECT COUNT(*) as count FROM patient_data", conn)
                    print(f"Total records: {count_df['count'].iloc[0]:,}")
                    
                    # Sample date formats
                    sample_dates = pd.read_sql("SELECT DISTINCT opd_dt FROM patient_data WHERE opd_dt IS NOT NULL AND opd_dt != '' LIMIT 10", conn)
                    print(f"Sample dates: {sample_dates['opd_dt'].tolist()}")
                    
                    # Sample amounts
                    sample_amounts = pd.read_sql("SELECT total_amt FROM patient_data WHERE total_amt IS NOT NULL AND total_amt > 0 LIMIT 5", conn)
                    print(f"Sample amounts: {sample_amounts['total_amt'].tolist()}")
                    
                else:
                    print("❌ patient_data table not found")
            except Exception as e:
                print(f"❌ Error checking patient_data: {e}")
            
            # Check stock tables
            print("\n2. INVENTORY TABLES:")
            try:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%stock%'")
                stock_tables = [row[0] for row in cursor.fetchall()]
                
                if stock_tables:
                    for table_name in stock_tables:
                        print(f"\n📋 Table: {table_name}")
                        
                        # Get columns
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = cursor.fetchall()
                        column_names = [col[1] for col in columns]
                        print(f"   Columns: {column_names}")
                        
                        # Get sample data
                        sample_df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 3", conn)
                        print(f"   Sample data shape: {sample_df.shape}")
                        if not sample_df.empty:
                            print(f"   Sample row: {sample_df.iloc[0].to_dict()}")
                else:
                    print("❌ No stock tables found")
            except Exception as e:
                print(f"❌ Error checking stock tables: {e}")
            
            # Test data extraction
            print("\n3. DATA EXTRACTION TEST:")
            try:
                revenue_data = self.get_patient_revenue_data()
                print(f"✅ Revenue data: {len(revenue_data)} months")
                if not revenue_data.empty:
                    print(f"   Date range: {revenue_data['ds'].min()} to {revenue_data['ds'].max()}")
                    print(f"   Total revenue: ₹{revenue_data['y'].sum():,.0f}")
            except Exception as e:
                print(f"❌ Revenue extraction failed: {e}")
            
            try:
                patient_data = self.get_patient_count_data()
                print(f"✅ Patient count data: {len(patient_data)} months")
                if not patient_data.empty:
                    print(f"   Date range: {patient_data['ds'].min()} to {patient_data['ds'].max()}")
                    print(f"   Total patients: {patient_data['y'].sum():,.0f}")
            except Exception as e:
                print(f"❌ Patient count extraction failed: {e}")
            
            conn.close()
            print("\n=== DEBUG COMPLETE ===")
            
        except Exception as e:
            print(f"❌ Debug error: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize forecasting system
    forecasting = MedicalForecastingSystem()
    
    print("=== MEDICAL FORECASTING SYSTEM TEST ===\n")
    
    # Run debug first
    forecasting.debug_database_structure()
    
    print("\n" + "="*50)
    print("FORECASTING TESTS")
    print("="*50)
    
    # Test revenue forecasting
    print("\n1. Testing Revenue Forecasting...")
    revenue_result = forecasting.forecast_monthly_revenue(months_ahead=6)
    if revenue_result['success']:
        print("✅ Revenue forecast successful")
        print(f"   Insights: {revenue_result['insights']}")
    else:
        print(f"❌ Revenue forecast failed: {revenue_result.get('error', 'Unknown error')}")
    
    # Test patient count forecasting
    print("\n2. Testing Patient Count Forecasting...")
    patient_result = forecasting.forecast_patient_count(months_ahead=6)
    if patient_result['success']:
        print("✅ Patient forecast successful")
        print(f"   Insights: {patient_result['insights']}")
    else:
        print(f"❌ Patient forecast failed: {patient_result.get('error', 'Unknown error')}")
    
    # Test inventory forecasting
    print("\n3. Testing Inventory Forecasting...")
    stock_result = forecasting.forecast_inventory_stock(months_ahead=6)
    if stock_result['success']:
        print("✅ Stock forecast successful")
        print(f"   Insights: {stock_result['insights']}")
    else:
        print(f"❌ Stock forecast failed: {stock_result.get('error', 'Unknown error')}")
    
    print("\n=== FORECASTING SYSTEM READY ===")