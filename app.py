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
    
    # Aggiungi gli aggiornamenti personalizzati se presenti
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
        df = pd.read_csv(uploaded_file)
        
        # Check essential columns
        required_columns = ['Date', 'Clicks', 'Impressions', 'Position', 'CTR']
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
        lines = content.split('\n')
        
        # Extract metadata
        metadata = {}
        data_start_index = 0
        for i, line in enumerate(lines):
            if line.startswith('# Start date:'):
                metadata['start_date'] = datetime.strptime(line.split(':')[1].strip(), '%Y%m%d')
            elif line.startswith('# End date:'):
                metadata['end_date'] = datetime.strptime(line.split(':')[1].strip(), '%Y%m%d')
            elif 'Week,Active Users' in line:
                data_start_index = i
                break
        
        # Verify minimum data period
        if (metadata['end_date'] - metadata['start_date']).days < 56:
            st.warning("At least 8 weeks of data are recommended for accurate forecasts.")
        
        # Process data
        data_end_index = data_start_index + 1
        while data_end_index < len(lines):
            line = lines[data_end_index].strip()
            if not line or not line[0].isdigit():
                break
            data_end_index += 1
        
        data_csv = '\n'.join(lines[data_start_index:data_end_index])
        df = pd.read_csv(StringIO(data_csv))
        
        df['Date'] = metadata['start_date'] + pd.to_timedelta(df['Week'] * 7, unit='D')
        df = df.rename(columns={'Active Users': 'Users'})
        
        return df[['Date', 'Users']]
        
    except Exception as e:
        st.error(f"Error loading GA4 data: {str(e)}")
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
            # Opzioni specifiche per Current traffic trend
            show_updates = st.checkbox("Show Google updates in the graph", value=True)
            
            # Aggiungi nuovi aggiornamenti Google
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
                    # Convertiamo la data nel formato richiesto
                    formatted_date = new_update_date.strftime("%d %b %Y")
                    
                    # Aggiorniamo le liste degli aggiornamenti
                    if 'custom_updates' not in st.session_state:
                        st.session_state.custom_updates = {
                            'dates': [],
                            'names': []
                        }
                    
                    st.session_state.custom_updates['dates'].append(formatted_date)
                    full_name = f"Released the {new_update_name} {new_update_type} update"
                    st.session_state.custom_updates['names'].append(full_name)
                    st.success(f"Aggiornamento '{full_name}' aggiunto con successo!")
    
    # Data source specific instructions
    if data_source == "Current traffic trend":
        st.info("""
        **How to export data from Google Analytics 4:**
        1. Access Google Analytics 4
        2. Go to "Reports" > "Acquisition" > "Acquisition Overview"
        3. Set the time filter to the **last 12 months** 
        4. Click the "Share this report" button at the top right
        5. Select "Download file"
        
        ⚠️ **Important**: 
        - If possible, export monthly data in CSV
        - At least 8 weeks of data are recommended for accurate forecasts
        - Don't modify the exported file format
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
        - If possible, export monthly data in CSV
        - At least 8 weeks of data are recommended for accurate forecasts
        - Don't modify the exported file format
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        f"Upload CSV file with {data_source} data",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        if data_source == "Current traffic trend":
            try:
                df = pd.read_csv(uploaded_file)
                df['Date'] = pd.to_datetime(df['Date'])
                
                if st.button("Genera analisi del traffico"):
                    with st.spinner("Analisi dei dati in corso..."):
                        # Show the traffic trend with Google updates
                        fig = create_traffic_trend_plot(df, show_updates=show_updates)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add summary statistics
                        st.subheader("Riepilogo del traffico")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Traffico medio giornaliero",
                                f"{df['Organic Traffic'].mean():,.0f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Picco di traffico",
                                f"{df['Organic Traffic'].max():,.0f}"
                            )
                        
                        with col3:
                            latest = df.iloc[-1]['Organic Traffic']
                            previous = df.iloc[-2]['Organic Traffic']
                            change = ((latest - previous) / previous) * 100
                            st.metric(
                                "Traffico più recente",
                                f"{latest:,.0f}",
                                f"{change:+.1f}%"
                            )
                        
                        # Show raw data
                        st.subheader("Dati grezzi")
                        st.dataframe(df)
                        
                        # Export functionality
                        excel_data = export_traffic_data(df)
                        st.download_button(
                            label="Scarica analisi del traffico (Excel)",
                            data=excel_data,
                            file_name="analisi_traffico.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            except Exception as e:
                st.error(f"Errore nel caricamento dei dati di traffico: {str(e)}")
        
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