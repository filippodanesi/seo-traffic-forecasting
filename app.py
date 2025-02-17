import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from io import BytesIO, StringIO

# Configurazione iniziale
st.set_page_config(page_title="SEO & Analytics Traffic Forecasting", layout="wide")

def create_prophet_model(data_source="GSC"):
    """
    Crea una nuova istanza del modello Prophet con parametri
    ottimizzati per la fonte dati, coerenti con dati mensili.
    """
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
            weekly_seasonality=False,  # disattivato per dati mensili
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,  # Più conservativo nelle variazioni
            seasonality_prior_scale=10,    # Enfatizza la stagionalità
            interval_width=0.95            # Intervallo di confidenza
        )

@st.cache_data
def load_gsc_data(uploaded_file, min_months=14):
    """Carica e valida i dati da Google Search Console"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Controllo colonne essenziali
        required_columns = ['Date', 'Clicks', 'Impressions', 'Position', 'CTR']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonne mancanti: {', '.join(missing_columns)}")
        
        # Conversione delle date
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Controllo range date con flessibilità
        date_range = (df['Date'].max() - df['Date'].min()).days / 30
        if date_range < min_months:
            if date_range >= min_months - 2:
                st.warning(f"Il dataset copre {date_range:.1f} mesi. Per risultati ottimali si consigliano {min_months} mesi di dati.")
            else:
                raise ValueError(f"Il dataset copre solo {date_range:.1f} mesi. Sono necessari almeno {min_months-2} mesi di dati.")
        
        # Conversione e validazione dati numerici
        numeric_columns = ['Clicks', 'Impressions', 'Position']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rimozione righe con dati mancanti o invalidi
        df = df.dropna(subset=numeric_columns)
        
        return df
        
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati GSC: {str(e)}")
        return None

@st.cache_data
def load_ga4_data(uploaded_file):
    """Carica e valida i dati da Google Analytics 4"""
    try:
        # Leggi il contenuto del file
        content = uploaded_file.read().decode('utf8')
        lines = content.split('\n')
        
        # Estrai metadati
        metadata = {}
        data_start_index = 0
        for i, line in enumerate(lines):
            if line.startswith('# Data di inizio:'):
                metadata['start_date'] = datetime.strptime(line.split(':')[1].strip(), '%Y%m%d')
            elif line.startswith('# Data di fine:'):
                metadata['end_date'] = datetime.strptime(line.split(':')[1].strip(), '%Y%m%d')
            elif 'Ennesima settimana,Utenti attivi' in line:
                data_start_index = i
                break
        
        # Verifica periodo minimo di dati (8 settimane)
        if (metadata['end_date'] - metadata['start_date']).days < 56:  # 8 settimane
            st.warning("Si consigliano almeno 8 settimane di dati per previsioni accurate.")
        
        # Trova la fine dei dati temporali
        data_end_index = data_start_index + 1
        while data_end_index < len(lines):
            line = lines[data_end_index].strip()
            if not line or not line[0].isdigit():  # Se la riga non inizia con un numero
                break
            data_end_index += 1
        
        # Crea DataFrame solo con i dati temporali
        data_csv = '\n'.join(lines[data_start_index:data_end_index])
        df = pd.read_csv(StringIO(data_csv))
        
        # Converti l'ennesima settimana in date effettive
        df['Date'] = metadata['start_date'] + pd.to_timedelta(df['Ennesima settimana'] * 7, unit='D')
        
        # Rinomina la colonna degli utenti per uniformità
        df = df.rename(columns={'Utenti attivi': 'Users'})
        
        return df[['Date', 'Users']]
        
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati GA4: {str(e)}")
        return None

def prepare_search_console_data(df):
    """
    Prepara i dati da Google Search Console per l'analisi aggregando a livello mensile
    e usando come data il primo giorno del mese.
    """
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_data = df.groupby('Month').agg({
        'Clicks': 'sum',
        'Impressions': 'sum',
        'Position': 'mean'
    }).reset_index()
    
    # Convertiamo il Period in datetime (primo giorno del mese)
    monthly_data['Date'] = monthly_data['Month'].apply(lambda r: r.to_timestamp(how='S'))
    monthly_data.drop(columns='Month', inplace=True)
    
    return monthly_data.sort_values('Date')

def prepare_analytics_data(df):
    """
    Prepara i dati da Google Analytics per l'analisi, aggregando a livello mensile
    e usando come data il primo giorno del mese.
    """
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_data = df.groupby('Month').agg({
        'Users': 'sum'
    }).reset_index()
    
    monthly_data['Date'] = monthly_data['Month'].apply(lambda r: r.to_timestamp(how='S'))
    monthly_data.drop(columns='Month', inplace=True)
    
    return monthly_data.sort_values('Date')

def forecast_with_prophet(df, metric, forecast_months, data_source="GSC"):
    """Esegue la previsione usando Facebook Prophet"""
    try:
        prophet_df = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'})
        
        model = create_prophet_model(data_source)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=forecast_months, freq='MS')
        forecast = model.predict(future)
        
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        result = result.rename(columns={
            'ds': 'Date',
            'yhat': f'Forecast_{metric}',
            'yhat_lower': f'{metric}_Lower_Bound',
            'yhat_upper': f'{metric}_Upper_Bound'
        })
        
        return result
        
    except Exception as e:
        st.error(f"Errore nella previsione per {metric}: {str(e)}")
        return None

def calculate_gsc_forecast(df, forecast_months=24):
    """Calcola le previsioni per le metriche di Search Console"""
    try:
        clicks_forecast = forecast_with_prophet(df, 'Clicks', forecast_months)
        impressions_forecast = forecast_with_prophet(df, 'Impressions', forecast_months)
        position_forecast = forecast_with_prophet(df, 'Position', forecast_months)
        
        if clicks_forecast is None or impressions_forecast is None or position_forecast is None:
            raise ValueError("Errore nel calcolo delle previsioni")
        
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
        st.error(f"Errore nel calcolo delle previsioni GSC: {str(e)}")
        return None, None

def calculate_ga4_forecast(df, forecast_months=24):
    """Calcola le previsioni per gli utenti di Analytics"""
    try:
        users_forecast = forecast_with_prophet(df, 'Users', forecast_months, data_source="GA4")
        
        if users_forecast is None:
            raise ValueError("Errore nel calcolo delle previsioni")
        
        historical_dates = df['Date']
        historical_df = df.copy()
        forecast_df = users_forecast[~users_forecast['Date'].isin(historical_dates)].copy()
        
        return historical_df, forecast_df
        
    except Exception as e:
        st.error(f"Errore nel calcolo delle previsioni GA4: {str(e)}")
        return None, None

def display_gsc_summary_metrics(historical_df, forecast_df):
    """Mostra il riepilogo delle metriche di Search Console"""
    metrics = [
        {
            'name': 'Click',
            'metric': 'Clicks',
            'format': '{:,.0f}',
            'description': 'Numero di volte che gli utenti hanno cliccato sui tuoi risultati di ricerca'
        },
        {
            'name': 'Impressioni',
            'metric': 'Impressions',
            'format': '{:,.0f}',
            'description': 'Numero di volte che i tuoi risultati sono apparsi nei risultati di ricerca'
        },
        {
            'name': 'Posizione media',
            'metric': 'Position',
            'format': '{:.1f}',
            'inverse': True,
            'description': 'Posizione media dei tuoi risultati nelle SERP (più basso è meglio)'
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
                f"Media {metric['name']} mensile attuale",
                metric['format'].format(current_avg)
            )
        with col2:
            display_pct = -change_pct if metric['metric'] == 'Position' else change_pct
            delta_color = "inverse" if metric['metric'] == 'Position' else "normal"
                
            st.metric(
                f"Media {metric['name']} mensile prevista",
                metric['format'].format(forecast_avg),
                f"{display_pct:+.1f}%",
                delta_color=delta_color
            )
        with col3:
            difference = forecast_avg - current_avg
            is_improvement = difference > 0
            
            if metric.get('inverse', False):
                is_improvement = not is_improvement
                
            display_difference = f"{'+' if difference > 0 else '-'}{metric['format'].format(abs(difference))}/mese"
            
            st.metric(
                f"Variazione {metric['name']}",
                display_difference,
                delta_color="normal" if is_improvement else "inverse"
            )
        
        if i < len(metrics) - 1:
            st.markdown("---")

def display_ga4_summary_metrics(historical_df, forecast_df):
    """Mostra il riepilogo delle metriche di Google Analytics"""
    metric = {
        'name': 'Utenti',
        'metric': 'Users',
        'format': '{:,.0f}',
        'description': 'Numero di utenti attivi sul sito'
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
            f"Media {metric['name']} mensile attuale",
            metric['format'].format(current_avg)
        )
    with col2:
        st.metric(
            f"Media {metric['name']} mensile prevista",
            metric['format'].format(forecast_avg),
            f"{change_pct:+.1f}%"
        )
    with col3:
        difference = forecast_avg - current_avg
        display_difference = f"{'+' if difference > 0 else '-'}{metric['format'].format(abs(difference))}/mese"
        
        st.metric(
            f"Variazione {metric['name']}",
            display_difference,
            delta_color="normal" if difference > 0 else "inverse"
        )

def create_gsc_plots(historical_df, forecast_df, plot_type='line'):
    """Crea grafici per le metriche di Search Console"""
    metrics = [
        ('Clicks', 'Click'),
        ('Impressions', 'Impressioni'),
        ('Position', 'Posizione media')
    ]
    
    figs = []
    for metric, title in metrics:
        fig = go.Figure()
        
        # Serie storica e previsionale
        if plot_type == 'line':
            fig.add_trace(go.Scatter(
                x=historical_df['Date'],
                y=historical_df[metric],
                name=f"{title} Storici",
                line=dict(color="#2962FF", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df[f'Forecast_{metric}'],
                name=f"{title} Previsti",
                line=dict(color="#FF6D00", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df[f'{metric}_Upper_Bound'],
                name="Limite superiore",
                line=dict(color="#2ca02c", width=1, dash="dash"),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df[f'{metric}_Lower_Bound'],
                name="Limite inferiore",
                line=dict(color="#d62728", width=1, dash="dash"),
                fill='tonexty',
                showlegend=False
            ))
        else:
            fig.add_trace(go.Bar(
                x=historical_df['Date'],
                y=historical_df[metric],
                name=f"{title} Storici",
                marker_color="#2962FF"
            ))
            fig.add_trace(go.Bar(
                x=forecast_df['Date'],
                y=forecast_df[f'Forecast_{metric}'],
                name=f"{title} Previsti",
                marker_color="#FF6D00"
            ))
        
        # Linea tratteggiata tra ultimo storico e primo forecast
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
            title=f"Andamento {title}",
            xaxis_title="Data",
            yaxis_title=title,
            height=400,
            showlegend=True,
            yaxis_autorange="reversed" if metric == "Position" else None
        )
        
        figs.append(fig)
    
    return figs

def create_ga4_plots(historical_df, forecast_df, plot_type='line'):
    """Crea grafici per le metriche di Google Analytics"""
    fig = go.Figure()
    
    if plot_type == 'line':
        fig.add_trace(go.Scatter(
            x=historical_df['Date'],
            y=historical_df['Users'],
            name="Utenti Storici",
            line=dict(color="#2962FF", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast_Users'],
            name="Utenti Previsti",
            line=dict(color="#FF6D00", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Users_Upper_Bound'],
            name="Limite superiore",
            line=dict(color="#2ca02c", width=1, dash="dash"),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Users_Lower_Bound'],
            name="Limite inferiore",
            line=dict(color="#d62728", width=1, dash="dash"),
            fill='tonexty',
            showlegend=False
        ))
    else:
        fig.add_trace(go.Bar(
            x=historical_df['Date'],
            y=historical_df['Users'],
            name="Utenti Storici",
            marker_color="#2962FF"
        ))
        fig.add_trace(go.Bar(
            x=forecast_df['Date'],
            y=forecast_df['Forecast_Users'],
            name="Utenti Previsti",
            marker_color="#FF6D00"
        ))
    
    # Linea tratteggiata tra ultimo storico e primo forecast
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
        title="Andamento Utenti",
        xaxis_title="Data",
        yaxis_title="Utenti",
        height=400,
        showlegend=True
    )
    
    return [fig]

def export_gsc_data(historical_df, forecast_df):
    """Esporta i dati di Search Console in Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        historical_df = historical_df.sort_values('Date').copy()
        forecast_df = forecast_df.sort_values('Date').copy()
        
        historical_df['Date'] = pd.to_datetime(historical_df['Date']).dt.strftime('%d/%m/%Y')
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.strftime('%d/%m/%Y')
        
        historical_df.to_excel(writer, sheet_name='Dati storici', index=False)
        forecast_df.to_excel(writer, sheet_name='Previsioni', index=False)
    
    return output.getvalue()

def export_ga4_data(historical_df, forecast_df):
    """Esporta i dati di Google Analytics in Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        historical_df = historical_df.sort_values('Date').copy()
        forecast_df = forecast_df.sort_values('Date').copy()
        
        historical_df['Date'] = pd.to_datetime(historical_df['Date']).dt.strftime('%d/%m/%Y')
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.strftime('%d/%m/%Y')
        
        historical_df.to_excel(writer, sheet_name='Dati storici', index=False)
        forecast_df.to_excel(writer, sheet_name='Previsioni', index=False)
    
    return output.getvalue()

def main():
    st.title("SEO & Analytics Traffic Forecasting Tool")
    
    # Selezione fonte dati
    data_source = st.selectbox(
        "Seleziona la fonte dati",
        ["Google Search Console", "Google Analytics 4"],
        format_func=lambda x: "GSC" if x == "Google Search Console" else "GA4"
    )
    
    # Sidebar per le impostazioni
    with st.sidebar:
        st.header("Impostazioni")
        forecast_months = st.slider(
            "Mesi da prevedere",
            min_value=1,
            max_value=24,
            value=12
        )
        
        plot_type = st.selectbox(
            "Tipo di grafico",
            ["line", "bar"],
            format_func=lambda x: "Linee" if x == "line" else "Barre"
        )
    
    # Istruzioni specifiche per fonte dati
    if data_source == "Google Search Console":
        st.info("""
        **Come esportare i dati da Google Search Console:**
        1. Accedi a Google Search Console
        2. Vai in "Performance" (Prestazioni)
        3. Imposta il filtro temporale sugli **ultimi 16 mesi**
        4. Clicca sul pulsante "Export" (Esporta) in alto
        5. Seleziona "CSV" come formato
        
        ⚠️ **Importante**: 
        - Non modificare il nome delle colonne nel file CSV
        - Si consigliano almeno 14 mesi di dati per previsioni accurate
        - Non modificare il formato del file esportato
        """)
    else:  # Google Analytics 4
        st.info("""
        **Come esportare i dati da Google Analytics 4:**
        1. Accedi a Google Analytics 4
        2. Vai in "Reports" > "Acquisizione" > "Panoramica dell'acquisizione"
        4. Imposta il filtro temporale sugli **ultimi 12 mesi**
        5. Clicca sul pulsante "Condividi questo report" in alto a destra
        6. Seleziona "Scarica file"
        
        ⚠️ **Importante**: 
        - Se possibile, esporta i dati mensili in CSV (o settimanali se vuoi un forecast settimanale)
        - Si consigliano almeno 8 settimane di dati per previsioni accurate
        - Non modificare il formato del file esportato
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        f"Carica il file CSV con i dati di {data_source}",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        if data_source == "Google Search Console":
            df = load_gsc_data(uploaded_file)
            if df is not None:
                df = prepare_search_console_data(df)
                
                if st.button("Genera previsione"):
                    with st.spinner("Calcolo previsioni in corso..."):
                        historical_df, forecast_df = calculate_gsc_forecast(
                            df, 
                            forecast_months=forecast_months
                        )
                    
                    if historical_df is not None and forecast_df is not None:
                        st.header("Risultati previsione")
                        
                        display_gsc_summary_metrics(historical_df, forecast_df)
                        
                        figs = create_gsc_plots(
                            historical_df, 
                            forecast_df,
                            plot_type=plot_type
                        )
                        
                        tab1, tab2, tab3 = st.tabs(["Click", "Impressioni", "Posizione media"])
                        with tab1:
                            st.plotly_chart(figs[0], use_container_width=True)
                        with tab2:
                            st.plotly_chart(figs[1], use_container_width=True)
                        with tab3:
                            st.plotly_chart(figs[2], use_container_width=True)
                        
                        excel_data = export_gsc_data(historical_df, forecast_df)
                        st.download_button(
                            label="Scarica report completo (Excel)",
                            data=excel_data,
                            file_name="previsione_traffico_seo.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        else:  # Google Analytics 4
            df = load_ga4_data(uploaded_file)
            if df is not None:
                df = prepare_analytics_data(df)
                
                if st.button("Genera previsione"):
                    with st.spinner("Calcolo previsioni in corso..."):
                        historical_df, forecast_df = calculate_ga4_forecast(
                            df,
                            forecast_months=forecast_months
                        )
                    
                    if historical_df is not None and forecast_df is not None:
                        st.header("Risultati previsione")
                        
                        display_ga4_summary_metrics(historical_df, forecast_df)
                        
                        figs = create_ga4_plots(
                            historical_df,
                            forecast_df,
                            plot_type=plot_type
                        )
                        
                        st.plotly_chart(figs[0], use_container_width=True)
                        
                        excel_data = export_ga4_data(historical_df, forecast_df)
                        st.download_button(
                            label="Scarica report completo (Excel)",
                            data=excel_data,
                            file_name="previsione_traffico_analytics.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == "__main__":
    main()
