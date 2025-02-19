import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from io import BytesIO, StringIO
import matplotlib.pyplot as plt

# Initial configuration
st.set_page_config(page_title="SEO Traffic Analysis & Forecasting Tool", layout="wide")

def extract_update_name(text):
    """Extracts update name from the full text"""
    if "update" in text.lower():
        idx = text.lower().index("update")
        return text[:idx + len("update")]
    return text

def get_predefined_updates():
    """Returns predefined list of Google updates"""
    update_dates_list = [
        '26 Dec 2024', '12 Dec 2024', '11 Nov 2024', '15 Aug 2024', '20 Jun 2024', 
        '5 Mar 2024', '8 Nov 2023', '2 Nov 2023', '5 Oct 2023', '14 Sep 2023', 
        '22 Aug 2023', '12 Apr 2023', '15 Mar 2023', '21 Feb 2023',
        '14 Dec 2022', '5 Dec 2022', '19 Oct 2022', '20 Sep 2022', '12 Sep 2022',
        '25 Aug 2022', '27 Jul 2022', '25 May 2022', '23 Mar 2022', '22 Feb 2022'
    ]
    
    update_names_list = [
        extract_update_name(name) for name in [
            'Released the December 2024 spam update',
            'Released the December 2024 core update',
            'Released the November 2024 core update',
            'Released the August 2024 core update',
            'Released the June 2024 spam update',
            'Released the March 2024 core update',
            'Released the November 2023 reviews update',
            'Released the November 2023 core update',
            'Released the October 2023 core update',
            'Released the September 2023 helpful content update',
            'Released the August 2023 core update',
            'Released the April 2023 reviews update',
            'Released the March 2023 core update',
            'Released the February 2023 product reviews update',
            'Released the December 2022 link spam update',
            'Released the December 2022 helpful content update',
            'Released the October 2022 spam update',
            'Released the September 2022 product reviews update',
            'Released the September 2022 core update',
            'Released the August 2022 helpful content update',
            'Released the July 2022 product reviews update',
            'Released the May 2022 core update',
            'Released the March 2022 product reviews update',
            'Released the page experience update for desktop'
        ]
    ]
    
    # Add custom updates if present
    if hasattr(st.session_state, 'custom_updates'):
        update_dates_list.extend(st.session_state.custom_updates['dates'])
        update_names_list.extend([extract_update_name(name) for name in st.session_state.custom_updates['names']])
    
    return update_dates_list, update_names_list

def create_prophet_model(data_source="GSC"):
    """Creates a new Prophet model instance with optimized parameters"""
    if data_source == "GSC":
        return Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
    else:  # GA4
        return Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            interval_width=0.95
        )

def create_traffic_trend_plot(df, show_updates=True):
    """Creates a plot showing only traffic data with Google updates markers"""
    update_dates_list, update_names_list = get_predefined_updates()
    update_dates = [datetime.strptime(date, "%d %b %Y") for date in update_dates_list]
    
    fig = go.Figure()
    
    # Add traffic data line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Organic Traffic'],
        name="Organic Traffic",
        line=dict(color="#1E88E5", width=2),
        mode='lines'
    ))
    
    if show_updates:
        # Separate core and spam updates
        core_updates = [(date, name) for date, name in zip(update_dates, update_names_list) 
                       if 'core' in name.lower()]
        spam_updates = [(date, name) for date, name in zip(update_dates, update_names_list) 
                       if 'spam' in name.lower()]
        
        y_max = df['Organic Traffic'].max()
        y_min = df['Organic Traffic'].min()
        y_range = y_max - y_min
        y_position = y_max - (y_range * 0.1)
        
        # Add vertical lines for core updates (purple)
        for date, name in core_updates:
            if (date >= df['Date'].min()) and (date <= df['Date'].max()):
                fig.add_vline(
                    x=date,
                    line=dict(color="#9C27B0", width=1, dash="dash"),
                    opacity=0.5
                )
                fig.add_annotation(
                    x=date,
                    y=y_position,
                    text=name,
                    textangle=90,
                    showarrow=False,
                    font=dict(color="#9C27B0", size=10),
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
        
        # Add vertical lines for spam updates (orange)
        for date, name in spam_updates:
            if (date >= df['Date'].min()) and (date <= df['Date'].max()):
                fig.add_vline(
                    x=date,
                    line=dict(color="#FF5722", width=1, dash="dash"),
                    opacity=0.5
                )
                fig.add_annotation(
                    x=date,
                    y=y_position,
                    text=name,
                    textangle=90,
                    showarrow=False,
                    font=dict(color="#FF5722", size=10),
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
        
        # Add legend for update types
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color="#9C27B0", dash="dash"),
            name="Core Updates"
        ))
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color="#FF5722", dash="dash"),
            name="Spam Updates"
        ))
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(
            text="Current Traffic Trend with Google Updates",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickangle=45
        ),
        yaxis=dict(
            title="Organic Traffic",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            rangemode='tozero'
        ),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100)
    )
    
    return fig

@st.cache_data
def load_gsc_data(uploaded_file, min_months=14):
    """Loads and validates data from Google Search Console"""
    try:
        content = uploaded_file.read().decode('utf8')
        
        # Check if the file uses semicolons as separators
        if ';' in content.split('\n')[0]:
            content = content.replace(';;', '')
            content = content.replace(';', ',')
        
        df = pd.read_csv(StringIO(content))
        
        # Check essential columns
        required_columns = ['Date', 'Clicks', 'Impressions', 'Position', 'CTR']
        italian_columns = ['Data', 'Clic', 'Impressioni', 'Posizione', 'CTR']
        
        # Check if Italian columns are present
        if all(col in df.columns for col in italian_columns):
            # Rename Italian columns to English
            column_mapping = {
                'Data': 'Date',
                'Clic': 'Clicks',
                'Impressioni': 'Impressions',
                'Posizione': 'Position'
            }
            df = df.rename(columns=column_mapping)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
        
        # Date conversion
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check date range
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
        
        # Remove rows with missing data
        df = df.dropna(subset=numeric_columns)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading GSC data: {str(e)}")
        return None

@st.cache_data
def load_ga4_data(uploaded_file):
    """Loads and validates data from Google Analytics 4"""
    try:
        content = uploaded_file.read().decode('utf8')
        
        # Fix double semicolons and convert to comma-separated if needed
        if ';;' in content:
            content = content.replace(';;', '')
            content = content.replace(';', ',')
            
        lines = content.split('\n')
        
        # Extract metadata (will be used if available)
        metadata = {'start_date': None, 'end_date': None}
        
        # Find the first non-comment line (should be header)
        data_start_index = 0
        has_header = False
        
        # Show file preview for debugging
        st.subheader("File Preview")
        st.code("\n".join(lines[:min(10, len(lines))]))
        
        # Process metadata while skipping comment lines
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                # Try to extract dates from comments
                if 'start date:' in line.lower():
                    try:
                        date_text = line.split(':')[1].strip()
                        metadata['start_date'] = datetime.strptime(date_text, '%Y%m%d')
                    except:
                        pass
                        
                elif 'end date:' in line.lower():
                    try:
                        date_text = line.split(':')[1].strip()
                        metadata['end_date'] = datetime.strptime(date_text, '%Y%m%d')
                    except:
                        pass
            elif line.strip():
                # This is the first non-comment line with content - should be the header
                data_start_index = i
                has_header = True
                break
        
        if not has_header:
            st.error("No data found in file after comment lines. Please check file format.")
            return None
        
        # Extract data section (header + data)
        data_csv = '\n'.join(lines[data_start_index:])
        
        # Read the CSV
        try:
            df = pd.read_csv(StringIO(data_csv))
        except Exception as csv_err:
            st.error(f"Error parsing CSV data: {str(csv_err)}")
            return None
            
        # Show the actual loaded data
        st.subheader("Loaded Data Preview")
        st.dataframe(df.head())
        
        # Try to identify columns based on headers (English or Italian)
        date_col = None
        users_col = None
        
        # First try exact matches
        for col in df.columns:
            if col.lower() in ['week', 'settimana', 'ennesima settimana', 'date', 'data']:
                date_col = col
            if col.lower() in ['active users', 'utenti attivi', 'users', 'utenti']:
                users_col = col
        
        # If not found, try partial matches
        if date_col is None:
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['week', 'settimana', 'date', 'data', 'day', 'giorno']):
                    date_col = col
                    break
                    
        if users_col is None:
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['user', 'utent', 'visit', 'active', 'attiv']):
                    users_col = col
                    break
        
        # If still not found and we have at least 2 columns, use the first and second columns
        if (date_col is None or users_col is None) and len(df.columns) >= 2:
            if date_col is None:
                date_col = df.columns[0]
                st.info(f"Using '{date_col}' as date column")
            
            if users_col is None:
                # Use the first non-date column
                for col in df.columns:
                    if col != date_col:
                        users_col = col
                        st.info(f"Using '{users_col}' as users column")
                        break
                
        # If still can't identify columns, let user select
        if date_col is None or users_col is None:
            st.warning("Could not automatically identify required columns.")
            
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Select date/time column:", options=df.columns)
            
            with col2:
                users_col = st.selectbox("Select users/traffic column:", 
                                       options=[c for c in df.columns if c != date_col])
                
        # If we couldn't find the metadata dates, try to infer from data
        if metadata['start_date'] is None or metadata['end_date'] is None:
            # Try to process the date column to infer date range
            try:
                # For week format
                if 'week' in date_col.lower() or 'settimana' in date_col.lower():
                    # If we have numeric weeks, estimate dates
                    if pd.to_numeric(df[date_col], errors='coerce').notna().any():
                        metadata['start_date'] = datetime.now() - timedelta(days=90)
                        metadata['end_date'] = datetime.now()
                else:
                    # Try to parse as dates
                    temp_dates = pd.to_datetime(df[date_col], errors='coerce')
                    if not temp_dates.isna().all():
                        metadata['start_date'] = temp_dates.min().to_pydatetime()
                        metadata['end_date'] = temp_dates.max().to_pydatetime()
            except:
                pass
                
        # If still missing dates, use defaults
        if metadata['start_date'] is None:
            metadata['start_date'] = datetime.now() - timedelta(days=90)
            st.info("Using 90 days ago as start date")
            
        if metadata['end_date'] is None:
            metadata['end_date'] = datetime.now()
            st.info("Using today as end date")
        
        # Create standardized dataframe
        result_df = pd.DataFrame()
        
        # Handle Week vs Date format
        if 'week' in date_col.lower() or 'settimana' in date_col.lower():
            # For week format, calculate dates from start date
            try:
                weeks = pd.to_numeric(df[date_col], errors='coerce')
                result_df['Date'] = metadata['start_date'] + pd.to_timedelta(weeks * 7, unit='D')
            except Exception as week_err:
                st.error(f"Error converting weeks to dates: {str(week_err)}")
                return None
        else:
            # For date format, parse the date
            try:
                result_df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            except:
                try:
                    # Try European format
                    result_df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
                except Exception as date_err:
                    st.error(f"Could not convert '{date_col}' column to dates: {str(date_err)}")
                    return None
        
        # Convert users column to numeric
        result_df['Users'] = pd.to_numeric(df[users_col], errors='coerce')
        
        # Drop rows with missing data
        result_df = result_df.dropna(subset=['Users', 'Date'])
        
        if len(result_df) == 0:
            st.error("No valid data found after processing.")
            return None
            
        st.success(f"Successfully loaded {len(result_df)} data points")
        return result_df[['Date', 'Users']]
        
    except Exception as e:
        st.error(f"Error loading GA4 data: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

def prepare_search_console_data(df):
    """Prepares Google Search Console data for analysis"""
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_data = df.groupby('Month').agg({
        'Clicks': 'sum',
        'Impressions': 'sum',
        'Position': 'mean'
    }).reset_index()
    
    monthly_data['Date'] = monthly_data['Month'].apply(lambda r: r.to_timestamp(how='S'))
    monthly_data.drop(columns='Month', inplace=True)
    
    return monthly_data.sort_values('Date')

def prepare_analytics_data(df):
    """Prepares Google Analytics data for analysis"""
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_data = df.groupby('Month').agg({
        'Users': 'sum'
    }).reset_index()
    
    monthly_data['Date'] = monthly_data['Month'].apply(lambda r: r.to_timestamp(how='S'))
    monthly_data.drop(columns='Month', inplace=True)
    
    return monthly_data.sort_values('Date')

def forecast_with_prophet(df, metric, forecast_months, data_source="GSC"):
    """Performs forecasting using Facebook Prophet"""
    try:
        prophet_df = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'})
        
        apply_log_transform = metric in ["Clicks", "Impressions", "Users"]
        
        if metric == "Position":
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive'
            )
        else:
            model = create_prophet_model(data_source)
        
        if apply_log_transform:
            prophet_df['y'] = np.log1p(prophet_df['y'])
        
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_months, freq='MS')
        forecast = model.predict(future)
        
        if apply_log_transform:
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] = np.expm1(forecast[['yhat', 'yhat_lower', 'yhat_upper']])
        
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
            change_pct = ((current_avg - forecast_avg) / current_avg * 100)
        else:
            change_pct = ((forecast_avg - current_avg) / current_avg * 100)
        
        with col1:
            st.metric(
                f"Current Monthly Average {metric['name']}",
                metric['format'].format(current_avg)
            )
        with col2:
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

def export_traffic_data(df):
    """Exports current traffic data to Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export = df.copy()
        df_export['Date'] = pd.to_datetime(df_export['Date']).dt.strftime('%d/%m/%Y')
        df_export.to_excel(writer, sheet_name='Traffic Data', index=False)
    
    return output.getvalue()

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
    st.title("SEO Traffic Analysis & Forecasting Tool")
    
    # Data source selection with the new option
    data_source = st.selectbox(
        "Select data source",
        ["Google Search Console", "Google Analytics 4", "Current traffic trend"],
        format_func=lambda x: x
    )
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Show forecast settings only for GSC and GA4
        if data_source != "Current traffic trend":
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
            
            show_updates = st.checkbox("Show Google Updates", value=True)
        else:
            # Options specific for Current traffic trend
            show_updates = st.checkbox("Show Google updates in the graph", value=True)
            
            # Add new Google updates
            st.subheader("Add a new Google Update")
            new_update_date = st.date_input(
                "Update date",
                value=None,
                format="DD/MM/YYYY"
            )
            new_update_name = st.text_input("Update name")
            new_update_type = st.selectbox(
                "Update type",
                ["core", "spam"],
                format_func=lambda x: "Core Update" if x == "core" else "Spam Update"
            )
            
            if st.button("Add an update"):
                if new_update_date and new_update_name:
                    # Convert date to required format
                    formatted_date = new_update_date.strftime("%d %b %Y")
                    
                    # Update the lists of updates
                    if 'custom_updates' not in st.session_state:
                        st.session_state.custom_updates = {
                            'dates': [],
                            'names': []
                        }
                    
                    st.session_state.custom_updates['dates'].append(formatted_date)
                    full_name = f"Released the {new_update_name} {new_update_type} update"
                    st.session_state.custom_updates['names'].append(full_name)
                    st.success(f"Update '{full_name}' successfully added!")
    
    # Data source specific instructions
    if data_source == "Current traffic trend":
        st.info("""
        **How to prepare traffic trend data:**
        1. You can use the same GA4 export file - the app will automatically detect relevant columns
        2. The tool will identify date/time columns and traffic/users columns automatically
        3. Comment lines (starting with #) will be automatically skipped
        
        ⚠️ **Important**: 
        - Include data for at least 6 months for meaningful analysis
        - The file can use either commas or semicolons as separators
        - Both English and Italian exports are fully supported
        """)
    elif data_source == "Google Search Console":
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
        - Both English and Italian exports are supported (Clicks/Clic, Impressions/Impressioni)
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
        - If possible, export monthly data in CSV format
        - At least 8 weeks of data are recommended for accurate forecasts
        - Both English and Italian exports are supported
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        f"Upload CSV file with {data_source} data",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        if data_source == "Current traffic trend":
            try:
                content = uploaded_file.read().decode('utf8')
                
                # Check if the file uses semicolons as separators
                if ';' in content.split('\n')[0]:
                    content = content.replace(';;', '')
                    content = content.replace(';', ',')
                
                # Skip comment lines (lines starting with #)
                lines = content.split('\n')
                data_start_index = 0
                has_header = False
                
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith('#'):
                        # This is the first non-comment line - should be the header
                        data_start_index = i
                        has_header = True
                        break
                
                if not has_header:
                    st.error("No data found in file. Please check the file format.")
                    st.stop()
                
                # Extract data section as CSV
                data_csv = '\n'.join(lines[data_start_index:])
                df = pd.read_csv(StringIO(data_csv))
                
                # For traffic trend, we need to detect or create Date and Organic Traffic columns
                # from the actual columns in the GA4 file
                
                # First, show preview of loaded columns
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Try to identify date column
                date_col = None
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['date', 'data', 'week', 'settimana', 'giorno', 'day']):
                        date_col = col
                        break
                        
                # If no date column found, use the first column as date
                if date_col is None and len(df.columns) > 0:
                    date_col = df.columns[0]
                    st.info(f"Using '{date_col}' as the date column")
                
                # Try to identify traffic column
                traffic_col = None
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in 
                           ['user', 'utent', 'traffic', 'traffico', 'visit', 'session', 'active', 'attiv']):
                        traffic_col = col
                        break
                        
                # If no traffic column found, use the second column
                if traffic_col is None and len(df.columns) > 1:
                    traffic_col = df.columns[1]
                    st.info(f"Using '{traffic_col}' as the traffic column")
                
                # If columns couldn't be identified, let user select
                if date_col is None or traffic_col is None:
                    st.error("Could not identify required columns automatically.")
                    if len(df.columns) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            date_col = st.selectbox("Select date column", options=df.columns)
                        with col2:
                            traffic_col = st.selectbox("Select traffic column", 
                                                     options=[c for c in df.columns if c != date_col])
                    else:
                        st.error("File must contain at least two columns.")
                        st.stop()
                
                # Create standardized dataframe with Date and Organic Traffic columns
                standardized_df = pd.DataFrame()
                
                # Convert date column
                try:
                    standardized_df['Date'] = pd.to_datetime(df[date_col])
                except:
                    try:
                        # Try European format
                        standardized_df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
                    except:
                        st.error(f"Could not convert '{date_col}' to dates. Please ensure it contains valid dates.")
                        st.stop()
                
                # Convert traffic column to numeric
                standardized_df['Organic Traffic'] = pd.to_numeric(df[traffic_col], errors='coerce')
                
                # Drop rows with missing data
                standardized_df = standardized_df.dropna()
                
                if len(standardized_df) == 0:
                    st.error("No valid data found after processing.")
                    st.stop()
                    
                # Use the standardized dataframe for analysis
                df = standardized_df
                
                if st.button("Generate traffic analysis"):
                    with st.spinner("Analyzing data..."):
                        # Show the traffic trend with Google updates
                        fig = create_traffic_trend_plot(df, show_updates=show_updates)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add summary statistics
                        st.subheader("Traffic Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Average daily traffic",
                                f"{df['Organic Traffic'].mean():,.0f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Peak traffic",
                                f"{df['Organic Traffic'].max():,.0f}"
                            )
                        
                        with col3:
                            latest = df.iloc[-1]['Organic Traffic']
                            previous = df.iloc[-2]['Organic Traffic']
                            change = ((latest - previous) / previous) * 100
                            st.metric(
                                "Most recent traffic",
                                f"{latest:,.0f}",
                                f"{change:+.1f}%"
                            )
                        
                        # Show raw data
                        st.subheader("Raw data")
                        st.dataframe(df)
                        
                        # Export functionality
                        excel_data = export_traffic_data(df)
                        st.download_button(
                            label="Download traffic analysis (Excel)",
                            data=excel_data,
                            file_name="traffic_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            except Exception as e:
                st.error(f"Error loading traffic data: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
        
        elif data_source == "Google Search Console":
            df = load_gsc_data(uploaded_file)
            if df is not None:
                df = prepare_search_console_data(df)
                
                if st.button("Generate Analysis"):
                    with st.spinner("Analyzing data..."):
                        historical_df, forecast_df = calculate_gsc_forecast(
                            df, 
                            forecast_months=forecast_months
                        )
                    
                    if historical_df is not None and forecast_df is not None:
                        st.header("Analysis Results")
                        
                        # Display metrics
                        display_gsc_summary_metrics(historical_df, forecast_df)
                        
                        # Create tabs for different views
                        tab1, tab2 = st.tabs([
                            "Forecast Charts", 
                            "Raw Data"
                        ])
                        
                        with tab1:
                            # Original forecast charts
                            figs = create_gsc_plots(
                                historical_df, 
                                forecast_df,
                                plot_type=plot_type
                            )
                            st.plotly_chart(figs[0], use_container_width=True)
                            st.plotly_chart(figs[1], use_container_width=True)
                            st.plotly_chart(figs[2], use_container_width=True)
                            
                        with tab2:
                            # Raw data view
                            st.dataframe(historical_df)
                            
                        # Export functionality
                        excel_data = export_gsc_data(historical_df, forecast_df)
                        st.download_button(
                            label="Download complete report (Excel)",
                            data=excel_data,
                            file_name="seo_traffic_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        else:  # Google Analytics 4
            df = load_ga4_data(uploaded_file)
            if df is not None:
                df = prepare_analytics_data(df)
                
                if st.button("Generate Analysis"):
                    with st.spinner("Analyzing data..."):
                        historical_df, forecast_df = calculate_ga4_forecast(
                            df,
                            forecast_months=forecast_months
                        )
                    
                    if historical_df is not None and forecast_df is not None:
                        st.header("Analysis Results")
                        
                        display_ga4_summary_metrics(historical_df, forecast_df)
                        
                        # Create tabs for different views
                        tab1, tab2 = st.tabs([
                            "Forecast Charts",
                            "Raw Data"
                        ])
                        
                        with tab1:
                            figs = create_ga4_plots(
                                historical_df,
                                forecast_df,
                                plot_type=plot_type
                            )
                            st.plotly_chart(figs[0], use_container_width=True)
                        
                        with tab2:
                            st.dataframe(historical_df)
                        
                        excel_data = export_ga4_data(historical_df, forecast_df)
                        st.download_button(
                            label="Download complete report (Excel)",
                            data=excel_data,
                            file_name="analytics_traffic_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == "__main__":
    main()
