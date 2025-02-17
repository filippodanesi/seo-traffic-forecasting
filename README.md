# SEO & Analytics Traffic Forecasting

A Streamlit-based tool for forecasting SEO traffic and analytics metrics using [Facebook's Prophet algorithm](https://facebook.github.io/prophet/).

## Features

- Support for both Google Search Console and Google Analytics 4 data
- Monthly traffic forecasting up to 24 months ahead
- Interactive visualizations with confidence intervals
- Exportable reports in Excel format
- Customizable visualization options

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/filippodanesi/seo-traffic-forecasting.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Upload your data:
   - For GSC: Export Performance data (last 16 months recommended)
   - For GA4: Export Acquisition Overview data (minimum 8 weeks recommended)

3. Configure forecast settings in the sidebar
4. Generate and explore forecasts

## Data Format Requirements

### Google Search Console
- Required columns: Date, Clicks, Impressions, Position, CTR
- Minimum 14 months of historical data recommended

### Google Analytics 4
- Required columns: **Nth week**, **Active users**
- Minimum 8 weeks of historical data recommended

## License

MIT