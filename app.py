import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from io import BytesIO, StringIO

# Initial configuration
st.set_page_config(page_title="SEO Traffic Forecasting Tool", layout="wide")

def create_prophet_model(data_source="GSC", data_frequency="monthly"):
    """
    Creates a new Prophet model instance with parameters
    optimized for the data source and frequency.
    """
    if data_source == "GSC":
        return Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # used only if the metric is not "Position"
            changepoint_prior_scale=0.05,  # more conservative in variations
        )
    else:  # GA4
        # Per dati giornalieri, abilita seasonality settimanale
        if data_frequency == "daily":
            return Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,  # Enabled for daily data
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                interval_width=0.95
            )
        else:  # monthly
            return Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,  # disabled for monthly data
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                interval_width=0.95
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
    """
    Loads and validates data from Google Analytics 4 with improved support
    for simple CSV format with 'Data' and 'Utenti totali' columns
    """
    try:
        # Prima tentiamo di caricare il file come CSV semplice
        try:
            df = pd.read_csv(uploaded_file)
            
            # Verifica se è il formato semplice (Data, Utenti totali)
            if set(df.columns) == set(['Data', 'Utenti totali']):
                df = df.rename(columns={
                    'Data': 'Date',
                    'Utenti totali': 'Users'
                })
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Converte Users in numerico
                df['Users'] = pd.to_numeric(df['Users'], errors='coerce')
                
                # Verifica validità dei dati
                if df['Date'].isna().any():
                    raise ValueError("Some dates could not be converted to a valid date format")
                    
                if df['Users'].isna().any():
                    st.warning("Some user values are missing or invalid and will be ignored")
                    df = df.dropna(subset=['Users'])
                
                # Controlla il periodo di dati disponibile
                if len(df) < 30:  # Minimo 30 giorni di dati
                    st.warning(f"Dataset contains only {len(df)} days. At least 30 days of data are recommended for accurate forecasts.")
                
                return df
            
            # Se non è il formato semplice, prova con il formato standard
            required_cols = ['Date', 'Users']
            if all(col in df.columns for col in required_cols):
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Users'] = pd.to_numeric(df['Users'], errors='coerce')
                return df[['Date', 'Users']].dropna()
                
        except Exception as e:
            st.warning(f"Couldn't parse as simple CSV: {str(e)}. Trying standard GA4 export format...")
        
        # Se fallisce, fallback al parser originale
        content = uploaded_file.read().decode('utf8')
        lines = content.split('\n')
        
        # Extract metadata with better error handling
        metadata = {
            'start_date': None,
            'end_date': None
        }
        data_start_index = 0
        
        # Try to find metadata and data start
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('# Start date:'):
                try:
                    date_str = line.split(':')[1].strip()
                    metadata['start_date'] = datetime.strptime(date_str, '%Y%m%d')
                except (IndexError, ValueError):
                    st.warning(f"Could not parse start date from: {line}")
            elif line.startswith('# End date:'):
                try:
                    date_str = line.split(':')[1].strip()
                    metadata['end_date'] = datetime.strptime(date_str, '%Y%m%d')
                except (IndexError, ValueError):
                    st.warning(f"Could not parse end date from: {line}")
            elif 'Week,Active Users' in line or 'Day,Active Users' in line or 'Month,Active Users' in line:
                data_start_index = i
                break
            # Alternative format detection
            elif any(x in line for x in ['Users', 'Sessions', 'Pageviews']) and i > 5:
                data_start_index = i
                break
        
        # If no data section found, look for other possible headers
        if data_start_index == 0:
            for i, line in enumerate(lines):
                if any(x in line for x in ['Date', 'Week', 'Day', 'Month']) and any(x in line for x in ['Users', 'Visitors', 'Sessions']):
                    data_start_index = i
                    break
        
        if data_start_index == 0:
            raise ValueError("Could not find data section in the file. Please check the file format.")
            
        # Verify minimum data period if dates are available
        if metadata['start_date'] is not None and metadata['end_date'] is not None:
            if (metadata['end_date'] - metadata['start_date']).days < 56:  # 8 weeks
                st.warning("At least 8 weeks of data are recommended for accurate forecasts.")
        else:
            st.warning("Could not determine data period. Make sure to have at least 8 weeks of data.")
        
        # Find end of temporal data
        data_end_index = data_start_index + 1
        while data_end_index < len(lines):
            line = lines[data_end_index].strip()
            if not line or not (line[0].isdigit() or line.startswith('20')):  # Checking for date format
                break
            data_end_index += 1
        
        # Create DataFrame from the data section
        data_csv = '\n'.join(lines[data_start_index:data_end_index])
        df = pd.read_csv(StringIO(data_csv))
        
        # Try to extract date information
        date_column = None
        for col in df.columns:
            if any(x in col.lower() for x in ['date', 'day', 'week', 'month']):
                date_column = col
                break
        
        if date_column is None:
            raise ValueError("Could not find date column in the file")
            
        # Try to find users column
        users_column = None
        for col in df.columns:
            if any(x in col.lower() for x in ['users', 'visitors', 'active users']):
                users_column = col
                break
                
        if users_column is None:
            raise ValueError("Could not find users column in the file")
        
        # Handle different date formats
        try:
            df['Date'] = pd.to_datetime(df[date_column])
        except:
            if 'Week' in df.columns:
                # If using Week column and we have start_date
                if metadata['start_date'] is not None:
                    df['Date'] = metadata['start_date'] + pd.to_timedelta(df['Week'] * 7, unit='D')
                else:
                    # Assume weeks are numbered from start of current year
                    current_year = datetime.now().year
                    df['Date'] = pd.to_datetime(f'{current_year}-01-01') + pd.to_timedelta(df['Week'] * 7, unit='D')
                    st.warning("Could not determine start date. Assuming weeks are from beginning of current year.")
            else:
                raise ValueError(f"Could not convert {date_column} to datetime format")
        
        # Rename users column for uniformity
        df = df.rename(columns={users_column: 'Users'})
        
        # Convert Users to numeric, handling potential formatting issues
        df['Users'] = pd.to_numeric(df['Users'].astype(str).str.replace(',', ''), errors='coerce')
        
        # Drop rows with missing values
        df = df.dropna(subset=['Date', 'Users'])
        
        if df.empty:
            raise ValueError("No valid data found after processing")
            
        return df[['Date', 'Users']]
        
    except Exception as e:
        st.error(f"Error loading GA4 data: {str(e)}")
        return None

def detect_data_frequency(df):
    """
    Detects the frequency of the time series data
    Returns: 'daily', 'weekly', or 'monthly'
    """
    if len(df) < 2:
        return 'daily'  # Default to daily if not enough data
        
    df = df.sort_values('Date')
    # Calcola la differenza mediana in giorni tra date consecutive
    date_diffs = df['Date'].diff().dt.days.median()
    
    if date_diffs is None or pd.isna(date_diffs):
        return 'daily'  # Default
    
    if date_diffs <= 1.5:
        return 'daily'
    elif date_diffs <= 7.5:
        return 'weekly'
    else:
        return 'monthly'

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

def prepare_analytics_data(df, frequency='auto'):
    """
    Prepares Google Analytics data for analysis by aggregating at the appropriate level
    based on detected frequency or user selection.
    """
    if frequency == 'auto':
        frequency = detect_data_frequency(df)
    
    if frequency == 'monthly':
        df['Period'] = df['Date'].dt.to_period('M')
        period_label = 'Month'
    elif frequency == 'weekly':
        # Create weekly periods starting on Mondays
        df['Period'] = df['Date'].dt.to_period('W-MON')
        period_label = 'Week'
    else:  # 'daily' - no aggregation needed
        return df.sort_values('Date')
    
    # Aggregate data at the chosen frequency level
    aggregated_data = df.groupby('Period').agg({
        'Users': 'sum'
    }).reset_index()
    
    # Convert Period to datetime (first day of period)
    aggregated_data['Date'] = aggregated_data['Period'].apply(lambda r: r.to_timestamp(how='S'))
    aggregated_data.drop(columns='Period', inplace=True)
    
    return aggregated_data.sort_values('Date')

def forecast_with_prophet(df, metric, forecast_periods, frequency='auto', data_source="GSC"):
    """Performs forecasting using Facebook Prophet with appropriate frequency."""
    try:
        if frequency == 'auto':
            frequency = detect_data_frequency(df)
            
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'MS'  # Month Start
        }
        prophet_freq = freq_map.get(frequency, 'D')
        
        prophet_df = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'})
        
        # Apply log transform ONLY for Impressions and Users, NOT for Clicks
        apply_log_transform = metric in ["Impressions", "Users"]
        
        # If Position, use specialized model parameters
        if metric == "Position":
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=(frequency == 'daily'),
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.03,
                seasonality_prior_scale=5
            )
        # Special case for Clicks - no log transform
        elif metric == "Clicks":
            model = create_prophet_model(data_source, frequency)
        # For Impressions and Users - use log transform
        else:
            model = create_prophet_model(data_source, frequency)
            if apply_log_transform:
                prophet_df['y'] = np.log1p(prophet_df['y'])  # log(y+1)
        
        # Fit model
        model.fit(prophet_df)
        
        # Predict future
        future = model.make_future_dataframe(periods=forecast_periods, freq=prophet_freq)
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
        clicks_forecast = forecast_with_prophet(df, 'Clicks', forecast_months, 'monthly', data_source="GSC")
        impressions_forecast = forecast_with_prophet(df, 'Impressions', forecast_months, 'monthly', data_source="GSC")
        position_forecast = forecast_with_prophet(df, 'Position', forecast_months, 'monthly', data_source="GSC")
        
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

def calculate_ga4_forecast(df, forecast_periods=24, frequency='auto'):
    """
    Calculates forecasts for Analytics users with support for different frequencies
    """
    try:
        if frequency == 'auto':
            frequency = detect_data_frequency(df)
            
        # Adjust forecast periods based on frequency
        if frequency == 'monthly':
            periods = forecast_periods  # as is
        elif frequency == 'weekly':
            periods = forecast_periods * 4  # approximately 4 weeks per month
        else:  # daily
            periods = forecast_periods * 30  # approximately 30 days per month
        
        users_forecast = forecast_with_prophet(df, 'Users', periods, frequency, data_source="GA4")
        
        if users_forecast is None:
            raise ValueError("Error in forecast calculation")
        
        historical_dates = df['Date']
        historical_df = df.copy()
        forecast_df = users_forecast[~users_forecast['Date'].isin(historical_dates)].copy()
        
        return historical_df, forecast_df, frequency
        
    except Exception as e:
        st.error(f"Error in GA4 forecast calculation: {str(e)}")
        return None, None, None

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

def display_ga4_summary_metrics(historical_df, forecast_df, frequency='monthly'):
    """Shows summary of Google Analytics metrics with appropriate time period label"""
    metric = {
        'name': 'Users',
        'metric': 'Users',
        'format': '{:,.0f}',
        'description': 'Number of active users on the site'
    }
    
    # Scegli l'etichetta appropriata basata sulla frequenza
    period_label = {
        'monthly': 'Monthly',
        'weekly': 'Weekly',
        'daily': 'Daily'
    }.get(frequency, 'Monthly')
    
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
            f"Current {period_label} Average {metric['name']}",
            metric['format'].format(current_avg)
        )
    with col2:
        st.metric(
            f"Forecasted {period_label} Average {metric['name']}",
            metric['format'].format(forecast_avg),
            f"{change_pct:+.1f}%"
        )
    with col3:
        difference = forecast_avg - current_avg
        display_difference = f"{'+' if difference > 0 else '-'}{metric['format'].format(abs(difference))}/{frequency[:-2]}"
        
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

def create_ga4_plots(historical_df, forecast_df, plot_type='line', frequency='monthly'):
    """Creates plots for Google Analytics metrics with appropriate frequency labels"""
    period_label = {
        'monthly': 'Monthly',
        'weekly': 'Weekly',
        'daily': 'Daily'
    }.get(frequency, 'Monthly')
    
    fig = go.Figure()
    
    if plot_type == 'line':
        fig.add_trace(go.Scatter(
            x=historical_df['Date'],
            y=historical_df['Users'],
            name=f"Historical {period_label} Users",
            line=dict(color="#2962FF", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast_Users'],
            name=f"Forecasted {period_label} Users",
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
            name=f"Historical {period_label} Users",
            marker_color="#2962FF"
        ))
        fig.add_trace(go.Bar(
            x=forecast_df['Date'],
            y=forecast_df['Forecast_Users'],
            name=f"Forecasted {period_label} Users",
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
        title=f"{period_label} Users Trend",
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
        
        # Nuova opzione per Analytics 4: scegliere la frequenza 
        # o lasciarla rilevare automaticamente
        if data_source == "Google Analytics 4":
            data_frequency = st.selectbox(
                "Data frequency",
                ["auto", "daily", "weekly", "monthly"],
                format_func=lambda x: "Auto-detect" if x == "auto" else x.capitalize(),
                index=0
            )
        else:
            data_frequency = "monthly"  # Default for GSC
    
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
        1. Access the SEO forecast template: https://lookerstudio.google.com/reporting/12aaee27-8de8-4a87-a62e-deb8a8c4d8f0
        2. Select your account from the dropdown at the top
        3. Set the timeframe to the last 12 months
        4. In the table on the left, click on the three dots (⋮) menu
        5. Select "Export" and choose CSV format
        
        ⚠️ **Important**: 
        - The CSV file should contain "Data" and "Utenti totali" columns
        - At least 12 months of data are recommended for accurate forecasts
        - Do not modify the column names in the exported file
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
                # Verifica e mostra la frequenza rilevata
                detected_frequency = detect_data_frequency(df)
                if data_frequency == 'auto':
                    st.info(f"Detected data frequency: {detected_frequency}")
                    selected_frequency = detected_frequency
                else:
                    selected_frequency = data_frequency
                    
                # Prepara i dati in base alla frequenza selezionata
                df = prepare_analytics_data(df, frequency=selected_frequency)
                
                if st.button("Generate forecast"):
                    with st.spinner("Calculating forecasts..."):
                        historical_df, forecast_df, used_frequency = calculate_ga4_forecast(
                            df,
                            forecast_periods=forecast_months,
                            frequency=selected_frequency
                        )
                    
                    if historical_df is not None and forecast_df is not None:
                        st.header("Forecast Results")
                        
                        display_ga4_summary_metrics(historical_df, forecast_df, frequency=used_frequency)
                        
                        figs = create_ga4_plots(
                            historical_df,
                            forecast_df,
                            plot_type=plot_type,
                            frequency=used_frequency
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
