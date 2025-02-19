import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from io import BytesIO, StringIO

# Initial configuration
st.set_page_config(page_title="SEO Traffic Forecasting Tool", layout="wide")

def create_prophet_model(data_source="GSC"):
    """
    Creates a new Prophet model instance with parameters
    optimized for the data source, consistent with monthly data.
    """
    if data_source == "GSC":
        return Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # used only if the metric is not "Position"
            changepoint_prior_scale=0.05,  # more conservative in variations (ADDED)
        )
    else:  # GA4
        return Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,  # disabled for monthly data
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,  # more conservative in variations
            seasonality_prior_scale=10,    # emphasizes seasonality
            interval_width=0.95            # confidence interval
        )

@st.cache_data
def load_gsc_data(uploaded_file, min_months=14):
    """Loads and validates data from Google Search Console"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check essential columns
        required_columns = ['Date', 'Clicks', 'Impressions', 'Position', 'CTR']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
        
        # Date conversion
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check date range with flexibility
        date_range = (df['Date'].max() - df['Date'].min()).days / 30
        if date_range < min_months:
            if date_range >= min_months - 2:
                st.warning(f"The dataset covers {date_range:.1f} months. For optimal results, {min_months} months of data are recommended.")
            else:
                raise ValueError(f"The dataset only covers {date_range:.1f} months. At least {min_months-2} months of data are required.")
        
        # Convert and validate numeric data
        numeric_columns = ['Clicks', 'Impressions', 'Position']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing or invalid data
        df = df.dropna(subset=numeric_columns)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading GSC data: {str(e)}")
        return None

@st.cache_data
def load_ga4_data(uploaded_file):
    """Universal CSV loader for analytics data with extreme flexibility"""
    try:
        # Reset upload file position to start
        uploaded_file.seek(0)
        
        # Try direct pandas read first
        try:
            df = pd.read_csv(uploaded_file)
            st.info("Direct CSV read successful")
        except Exception as e:
            st.warning(f"Standard CSV read failed: {e}. Trying alternative parsing...")
            # Reset file position
            uploaded_file.seek(0)
            content = uploaded_file.read().decode('utf8', errors='replace')
            
            # Try to determine if there's metadata section to skip
            lines = content.split('\n')
            
            # Look for the first line that could be a header
            data_start = 0
            for i, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # If line doesn't start with # or other comment indicators
                if not line.strip().startswith(('#', '//', '/*', '<!--')):
                    # Check if it has comma or tab (likely a header)
                    if ',' in line or '\t' in line:
                        data_start = i
                        break
            
            # If we found a potential header
            if data_start > 0:
                try:
                    data_content = '\n'.join(lines[data_start:])
                    df = pd.read_csv(StringIO(data_content))
                    st.info(f"Skipped {data_start} lines and successfully loaded data")
                except:
                    # Try tab delimiter
                    try:
                        df = pd.read_csv(StringIO(data_content), delimiter='\t')
                        st.info(f"Loaded as tab-delimited data")
                    except Exception as e:
                        raise ValueError(f"Failed to parse file after skipping metadata: {e}")
            else:
                raise ValueError("Could not identify data section in the file")
        
        # Display column names for debugging
        st.write("Found columns:", df.columns.tolist())
        
        # Identify date column
        date_columns = []
        for col in df.columns:
            col_lower = col.lower()
            # Check for date-related names
            if any(date_term in col_lower for date_term in ['date', 'day', 'week', 'month', 'year', 'time']):
                date_columns.append(col)
                
        if not date_columns:
            # If no obvious date column, look for columns that might contain dates
            for col in df.columns:
                # Sample the first 10 non-null values
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    # Check if values look like dates or numbers that could be weeks
                    try:
                        sample_str = sample.astype(str)
                        # Check for date patterns: YYYY-MM-DD, DD/MM/YYYY, etc.
                        if any(re.search(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', val) for val in sample_str):
                            date_columns.append(col)
                        # Check for standalone numbers that could be weeks/days
                        elif all(re.match(r'^\d+$', val) for val in sample_str):
                            if all(int(val) < 53 for val in sample_str):  # Max weeks in a year
                                date_columns.append(col)
                    except:
                        continue
        
        if not date_columns:
            raise ValueError("Could not identify a date column. Available columns: " + ", ".join(df.columns))
        
        date_col = date_columns[0]  # Use the first identified date column
        st.info(f"Using '{date_col}' as date column")
        
        # Identify user/metric column
        metric_col = None
        metric_terms = ['user', 'visit', 'session', 'traffic', 'view']
        for col in df.columns:
            if col != date_col and any(term in col.lower() for term in metric_terms):
                metric_col = col
                break
                
        # If no user column found, try to find any numeric column
        if metric_col is None:
            for col in df.columns:
                if col != date_col:
                    try:
                        # Check if column has numeric values
                        if pd.to_numeric(df[col], errors='coerce').notna().any():
                            metric_col = col
                            break
                    except:
                        continue
                        
        if metric_col is None:
            raise ValueError("Could not identify a metrics column (users, sessions, etc.)")
            
        st.info(f"Using '{metric_col}' as metrics column")
        
        # Process date column
        try:
            # Try direct conversion
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            
            # If conversion failed for most values
            if df['Date'].isna().mean() > 0.5:
                # Check if it looks like week numbers
                if df[date_col].astype(str).str.match(r'^\d+$').all():
                    # Assume weeks starting from beginning of current year
                    current_year = datetime.now().year
                    df['Date'] = pd.to_datetime(f'{current_year}-01-01') + pd.to_timedelta(df[date_col].astype(int) * 7, unit='D')
                    st.warning("Interpreted numeric values as week numbers")
            
            # Check that dates are valid
            if df['Date'].isna().all():
                raise ValueError("Could not convert date column to valid dates")
                
        except Exception as e:
            raise ValueError(f"Failed to process date column: {e}")
            
        # Process metrics column
        try:
            # Handle potential non-numeric formatting (commas, etc)
            df['Users'] = pd.to_numeric(df[metric_col].astype(str).str.replace(',', ''), errors='coerce')
            
            # Check if conversion was successful
            if df['Users'].isna().all():
                raise ValueError("Could not convert metrics to numeric values")
                
        except Exception as e:
            raise ValueError(f"Failed to process metrics column: {e}")
            
        # Final cleanup
        df = df[['Date', 'Users']].dropna()
        
        # Sort by date
        df = df.sort_values('Date')
        
        if len(df) < 4:
            raise ValueError(f"Not enough valid data points after processing. Found only {len(df)} valid rows.")
            
        st.success(f"Successfully loaded {len(df)} data points from {df['Date'].min()} to {df['Date'].max()}")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading GA4 data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def prepare_search_console_data(df):
    """
    Prepares Google Search Console data for analysis by aggregating at monthly level
    and using the first day of the month as the date.
    """
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_data = df.groupby('Month').agg({
        'Clicks': 'sum',
        'Impressions': 'sum',
        'Position': 'mean'
    }).reset_index()
    
    # Convert Period to datetime (first day of month)
    monthly_data['Date'] = monthly_data['Month'].apply(lambda r: r.to_timestamp(how='S'))
    monthly_data.drop(columns='Month', inplace=True)
    
    return monthly_data.sort_values('Date')

def prepare_analytics_data(df):
    """
    Prepares Google Analytics data for analysis by aggregating at monthly level
    and using the first day of the month as the date.
    """
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_data = df.groupby('Month').agg({
        'Users': 'sum'
    }).reset_index()
    
    monthly_data['Date'] = monthly_data['Month'].apply(lambda r: r.to_timestamp(how='S'))
    monthly_data.drop(columns='Month', inplace=True)
    
    return monthly_data.sort_values('Date')


def forecast_with_prophet(df, metric, forecast_months, data_source="GSC"):
    """Performs forecasting using Facebook Prophet, with log transform for specific metrics only."""
    try:
        prophet_df = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'})
        
        # Apply log transform ONLY for Impressions and Users, NOT for Clicks
        apply_log_transform = metric in ["Impressions", "Users"]
        
        # If Position, use specialized model parameters
        if metric == "Position":
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False, 
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.03,  # More flexible for position
                seasonality_prior_scale=5      # Less emphasis on seasonality for position
            )
        # Special case for Clicks - no log transform
        elif metric == "Clicks":
            model = create_prophet_model(data_source)
        # For Impressions and Users - use log transform
        else:
            model = create_prophet_model(data_source)
            if apply_log_transform:
                prophet_df['y'] = np.log1p(prophet_df['y'])  # log(y+1)
        
        # Fit model
        model.fit(prophet_df)
        
        # Predict future
        future = model.make_future_dataframe(periods=forecast_months, freq='MS')
        forecast = model.predict(future)
        
        # If we applied log transform, convert back to original scale
        if apply_log_transform:
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] = np.expm1(forecast[['yhat', 'yhat_lower', 'yhat_upper']])
        
        # Rename columns
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        result = result.rename(columns={
            'ds': 'Date',
            'yhat': f'Forecast_{metric}',
            'yhat_lower': f'{metric}_Lower_Bound',
            'yhat_upper': f'{metric}_Upper_Bound'
        })
        
        return result
        
    except Exception as e:
        st.error(f"Error in forecast for {metric}: {str(e)}")
        return None

def calculate_gsc_forecast(df, forecast_months=24):
    """Calculates forecasts for Search Console metrics"""
    try:
        clicks_forecast = forecast_with_prophet(df, 'Clicks', forecast_months, data_source="GSC")
        impressions_forecast = forecast_with_prophet(df, 'Impressions', forecast_months, data_source="GSC")
        position_forecast = forecast_with_prophet(df, 'Position', forecast_months, data_source="GSC")
        
        if clicks_forecast is None or impressions_forecast is None or position_forecast is None:
            raise ValueError("Error in forecast calculation")
        
        forecast_df = clicks_forecast.merge(
            impressions_forecast[['Date', 'Forecast_Impressions', 'Impressions_Lower_Bound', 'Impressions_Upper_Bound']],
            on='Date'
        ).merge(
            position_forecast[['Date', 'Forecast_Position', 'Position_Lower_Bound', 'Position_Upper_Bound']],
            on='Date'
        )
        
        historical_dates = df['Date']
        historical_df = df.copy()
        forecast_df = forecast_df[~forecast_df['Date'].isin(historical_dates)].copy()
        
        return historical_df, forecast_df
        
    except Exception as e:
        st.error(f"Error in GSC forecast calculation: {str(e)}")
        return None, None

def calculate_ga4_forecast(df, forecast_months=24):
    """Calculates forecasts for Analytics users"""
    try:
        users_forecast = forecast_with_prophet(df, 'Users', forecast_months, data_source="GA4")
        
        if users_forecast is None:
            raise ValueError("Error in forecast calculation")
        
        historical_dates = df['Date']
        historical_df = df.copy()
        forecast_df = users_forecast[~users_forecast['Date'].isin(historical_dates)].copy()
        
        return historical_df, forecast_df
        
    except Exception as e:
        st.error(f"Error in GA4 forecast calculation: {str(e)}")
        return None, None

def display_gsc_summary_metrics(historical_df, forecast_df):
    """Shows summary of Search Console metrics"""
    metrics = [
        {
            'name': 'Clicks',
            'metric': 'Clicks',
            'format': '{:,.0f}',
            'description': 'Number of times users clicked on your search results'
        },
        {
            'name': 'Impressions',
            'metric': 'Impressions',
            'format': '{:,.0f}',
            'description': 'Number of times your results appeared in search results'
        },
        {
            'name': 'Average Position',
            'metric': 'Position',
            'format': '{:.1f}',
            'inverse': True,
            'description': 'Average position of your results in SERPs (lower is better)'
        }
    ]
    
    for i, metric in enumerate(metrics):
        cols = st.columns([20, 1])
        with cols[0]:
            st.subheader(metric['name'])
        with cols[1]:
            st.markdown(f"<div title='{metric['description']}'>❔</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        current_avg = historical_df[metric['metric']].mean()
        forecast_avg = forecast_df[f"Forecast_{metric['metric']}"].mean()
        
        if metric.get('inverse', False):
            # Metrica "inversa" (posizione più bassa è meglio)
            change_pct = ((current_avg - forecast_avg) / current_avg * 100)
        else:
            change_pct = ((forecast_avg - current_avg) / current_avg * 100)
        
        with col1:
            st.metric(
                f"Current Monthly Average {metric['name']}",
                metric['format'].format(current_avg)
            )
        with col2:
            # Nota: per la Position invertiamo segno e colore
            display_pct = -change_pct if metric['metric'] == 'Position' else change_pct
            delta_color = "inverse" if metric['metric'] == 'Position' else "normal"
                
            st.metric(
                f"Forecasted Monthly Average {metric['name']}",
                metric['format'].format(forecast_avg),
                f"{display_pct:+.1f}%",
                delta_color=delta_color
            )
        with col3:
            difference = forecast_avg - current_avg
            is_improvement = difference > 0
            
            # Se la metrica è inversa, un decremento è un miglioramento
            if metric.get('inverse', False):
                is_improvement = not is_improvement
                
            display_difference = f"{'+' if difference > 0 else '-'}{metric['format'].format(abs(difference))}/month"
            
            st.metric(
                f"{metric['name']} Change",
                display_difference,
                delta_color="normal" if is_improvement else "inverse"
            )
        
        if i < len(metrics) - 1:
            st.markdown("---")

def display_ga4_summary_metrics(historical_df, forecast_df):
    """Shows summary of Google Analytics metrics"""
    metric = {
        'name': 'Users',
        'metric': 'Users',
        'format': '{:,.0f}',
        'description': 'Number of active users on the site'
    }
    
    cols = st.columns([20, 1])
    with cols[0]:
        st.subheader(metric['name'])
    with cols[1]:
        st.markdown(f"<div title='{metric['description']}'>❔</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    current_avg = historical_df[metric['metric']].mean()
    forecast_avg = forecast_df[f"Forecast_{metric['metric']}"].mean()
    change_pct = ((forecast_avg - current_avg) / current_avg * 100)
    
    with col1:
        st.metric(
            f"Current Monthly Average {metric['name']}",
            metric['format'].format(current_avg)
        )
    with col2:
        st.metric(
            f"Forecasted Monthly Average {metric['name']}",
            metric['format'].format(forecast_avg),
            f"{change_pct:+.1f}%"
        )
    with col3:
        difference = forecast_avg - current_avg
        display_difference = f"{'+' if difference > 0 else '-'}{metric['format'].format(abs(difference))}/month"
        
        st.metric(
            f"{metric['name']} Change",
            display_difference,
            delta_color="normal" if difference > 0 else "inverse"
        )

def create_gsc_plots(historical_df, forecast_df, plot_type='line'):
    """Creates plots for Search Console metrics"""
    metrics = [
        ('Clicks', 'Clicks'),
        ('Impressions', 'Impressions'),
        ('Position', 'Average Position')
    ]
    
    figs = []
    for metric, title in metrics:
        fig = go.Figure()
        
        # Historical and forecast series
        if plot_type == 'line':
            fig.add_trace(go.Scatter(
                x=historical_df['Date'],
                y=historical_df[metric],
                name=f"Historical {title}",
                line=dict(color="#2962FF", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df[f'Forecast_{metric}'],
                name=f"Forecasted {title}",
                line=dict(color="#FF6D00", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df[f'{metric}_Upper_Bound'],
                name="Upper Bound",
                line=dict(color="#2ca02c", width=1, dash="dash"),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df[f'{metric}_Lower_Bound'],
                name="Lower Bound",
                line=dict(color="#d62728", width=1, dash="dash"),
                fill='tonexty',
                showlegend=False
            ))
        else:
            fig.add_trace(go.Bar(
                x=historical_df['Date'],
                y=historical_df[metric],
                name=f"Historical {title}",
                marker_color="#2962FF"
            ))
            fig.add_trace(go.Bar(
                x=forecast_df['Date'],
                y=forecast_df[f'Forecast_{metric}'],
                name=f"Forecasted {title}",
                marker_color="#FF6D00"
            ))
        
        # Dashed line between last historical and first forecast
        if not forecast_df.empty:
            fig.add_shape(
                type="line",
                x0=historical_df['Date'].iloc[-1],
                y0=historical_df[metric].iloc[-1],
                x1=forecast_df['Date'].iloc[0],
                y1=forecast_df[f'Forecast_{metric}'].iloc[0],
                line=dict(color="grey", width=2, dash="dot")
            )
        
        fig.update_layout(
            title=f"{title} Trend",
            xaxis_title="Date",
            yaxis_title=title,
            height=400,
            showlegend=True,
            yaxis_autorange="reversed" if metric == "Position" else None
        )
        
        figs.append(fig)
    
    return figs

def create_ga4_plots(historical_df, forecast_df, plot_type='line'):
    """Creates plots for Google Analytics metrics"""
    fig = go.Figure()
    
    if plot_type == 'line':
        fig.add_trace(go.Scatter(
            x=historical_df['Date'],
            y=historical_df['Users'],
            name="Historical Users",
            line=dict(color="#2962FF", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast_Users'],
            name="Forecasted Users",
            line=dict(color="#FF6D00", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Users_Upper_Bound'],
            name="Upper Bound",
            line=dict(color="#2ca02c", width=1, dash="dash"),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Users_Lower_Bound'],
            name="Lower Bound",
            line=dict(color="#d62728", width=1, dash="dash"),
            fill='tonexty',
            showlegend=False
        ))
    else:
        fig.add_trace(go.Bar(
            x=historical_df['Date'],
            y=historical_df['Users'],
            name="Historical Users",
            marker_color="#2962FF"
        ))
        fig.add_trace(go.Bar(
            x=forecast_df['Date'],
            y=forecast_df['Forecast_Users'],
            name="Forecasted Users",
            marker_color="#FF6D00"
        ))
    
    # Dashed line between last historical and first forecast
    if not forecast_df.empty:
        fig.add_shape(
            type="line",
            x0=historical_df['Date'].iloc[-1],
            y0=historical_df['Users'].iloc[-1],
            x1=forecast_df['Date'].iloc[0],
            y1=forecast_df['Forecast_Users'].iloc[0],
            line=dict(color="grey", width=2, dash="dot")
        )
    
    fig.update_layout(
        title="Users Trend",
        xaxis_title="Date",
        yaxis_title="Users",
        height=400,
        showlegend=True
    )
    
    return [fig]

def export_gsc_data(historical_df, forecast_df):
    """Exports Search Console data to Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        historical_df = historical_df.sort_values('Date').copy()
        forecast_df = forecast_df.sort_values('Date').copy()
        
        historical_df['Date'] = pd.to_datetime(historical_df['Date']).dt.strftime('%d/%m/%Y')
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.strftime('%d/%m/%Y')
        
        historical_df.to_excel(writer, sheet_name='Historical Data', index=False)
        forecast_df.to_excel(writer, sheet_name='Forecasts', index=False)
    
    return output.getvalue()

def export_ga4_data(historical_df, forecast_df):
    """Exports Google Analytics data to Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        historical_df = historical_df.sort_values('Date').copy()
        forecast_df = forecast_df.sort_values('Date').copy()
        
        historical_df['Date'] = pd.to_datetime(historical_df['Date']).dt.strftime('%d/%m/%Y')
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.strftime('%d/%m/%Y')
        
        historical_df.to_excel(writer, sheet_name='Historical Data', index=False)
        forecast_df.to_excel(writer, sheet_name='Forecasts', index=False)
    
    return output.getvalue()

def main():
    st.title("SEO Traffic Forecasting Tool")
    
    # Data source selection
    data_source = st.selectbox(
        "Select data source",
        ["Google Search Console", "Google Analytics 4"],
        format_func=lambda x: x  # Show full names
    )
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        forecast_months = st.slider(
            "Months to forecast",
            min_value=1,
            max_value=24,
            value=12
        )
        
        plot_type = st.selectbox(
            "Chart type",
            ["line", "bar"],
            format_func=lambda x: "Lines" if x == "line" else "Bars"
        )
    
    # Data source specific instructions
    if data_source == "Google Search Console":
        st.info("""
        **How to export data from Google Search Console:**
        1. Access Google Search Console
        2. Go to "Performance" > "Search results"
        3. Set the time filter to the **last 16 months**
        4. Click the "Export" button at the top
        5. Select "CSV" as format
        
        ⚠️ **Important**: 
        - Don't modify column names in the CSV file
        - At least 14 months of data are recommended for accurate forecasts
        - Don't modify the exported file format
        """)
    else:  # Google Analytics 4
        st.info("""
        **How to export data from Google Analytics 4:**
        1. Access Google Analytics 4
        2. Go to "Reports" > "Acquisition" > "Acquisition Overview"
        3. Set the time filter to the **last 12 months** 
        4. Click the "Share this report" button at the top right
        5. Select "Download file"
        
        ⚠️ **Important**: 
        - If possible, export monthly data in CSV (or weekly if you want a weekly forecast)
        - At least 8 weeks of data are recommended for accurate forecasts
        - Don't modify the exported file format
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        f"Upload CSV file with {data_source} data",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        if data_source == "Google Search Console":
            df = load_gsc_data(uploaded_file)
            if df is not None:
                df = prepare_search_console_data(df)
                
                if st.button("Generate forecast"):
                    with st.spinner("Calculating forecasts..."):
                        historical_df, forecast_df = calculate_gsc_forecast(
                            df, 
                            forecast_months=forecast_months
                        )
                    
                    if historical_df is not None and forecast_df is not None:
                        st.header("Forecast Results")
                        
                        display_gsc_summary_metrics(historical_df, forecast_df)
                        
                        figs = create_gsc_plots(
                            historical_df, 
                            forecast_df,
                            plot_type=plot_type
                        )
                        
                        tab1, tab2, tab3 = st.tabs(["Clicks", "Impressions", "Average Position"])
                        with tab1:
                            st.plotly_chart(figs[0], use_container_width=True)
                        with tab2:
                            st.plotly_chart(figs[1], use_container_width=True)
                        with tab3:
                            st.plotly_chart(figs[2], use_container_width=True)
                        
                        excel_data = export_gsc_data(historical_df, forecast_df)
                        st.download_button(
                            label="Download complete report (Excel)",
                            data=excel_data,
                            file_name="seo_traffic_forecast.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        else:  # Google Analytics 4
            df = load_ga4_data(uploaded_file)
            if df is not None:
                df = prepare_analytics_data(df)
                
                if st.button("Generate forecast"):
                    with st.spinner("Calculating forecasts..."):
                        historical_df, forecast_df = calculate_ga4_forecast(
                            df,
                            forecast_months=forecast_months
                        )
                    
                    if historical_df is not None and forecast_df is not None:
                        st.header("Forecast Results")
                        
                        display_ga4_summary_metrics(historical_df, forecast_df)
                        
                        figs = create_ga4_plots(
                            historical_df,
                            forecast_df,
                            plot_type=plot_type
                        )
                        
                        st.plotly_chart(figs[0], use_container_width=True)
                        
                        excel_data = export_ga4_data(historical_df, forecast_df)
                        st.download_button(
                            label="Download complete report (Excel)",
                            data=excel_data,
                            file_name="analytics_traffic_forecast.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == "__main__":
    main()
