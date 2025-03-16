# vehicle_price_analyzer.py
"""
Vehicle Price Analyzer - A tool to scrape, analyze, and visualize vehicle pricing data.

This application:
1. Scrapes vehicle listings from an online marketplace
2. Processes the data into a structured format
3. Visualizes pricing trends in an interactive dashboard
4. Allows filtering and analysis of vehicle prices

Usage:
    python vehicle_price_analyzer.py [options]

Options:
    --output-dir DIR      Directory to save scraped data (default: 'scraped_vehicles')
    --manufacturer ID     Manufacturer ID to scrape (default: 38)
    --model ID            Model ID to scrape (default: 10514)
    --max-pages NUM       Maximum number of pages to scrape (default: 25)
    --skip-scrape         Skip scraping and use existing data
    --port NUM            Port to run the web server on (default: 8050)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
from scipy import optimize

# Web dashboard
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# Import scraper modules
from scraper import VehicleScraper
import yad2_parser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vehicle_analyzer')

# Constants
DEFAULT_OUTPUT_DIR = 'scraped_vehicles'
DEFAULT_MANUFACTURER_ID = 38
DEFAULT_MODEL_ID = 10514
DEFAULT_MAX_PAGES = 25
DEFAULT_PORT = 8050

# Dashboard styling constants
COLORS = {
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'success': '#2ecc71',
    'danger': '#e74c3c',
    'warning': '#f39c12',
    'info': '#3498db',
    'light': '#f9f9f9',
    'dark': '#2c3e50',
    'background': '#f9f9f9',
    'text': '#2c3e50',
    'border': '#ddd'
}

EXTERNAL_STYLESHEETS = [
    {
        'href': 'https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap',
        'rel': 'stylesheet'
    }
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vehicle Price Analyzer')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save scraped data (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--manufacturer', type=int, default=DEFAULT_MANUFACTURER_ID,
                        help=f'Manufacturer ID to scrape (default: {DEFAULT_MANUFACTURER_ID})')
    parser.add_argument('--model', type=int, default=DEFAULT_MODEL_ID,
                        help=f'Model ID to scrape (default: {DEFAULT_MODEL_ID})')
    parser.add_argument('--max-pages', type=int, default=DEFAULT_MAX_PAGES,
                        help=f'Maximum number of pages to scrape (default: {DEFAULT_MAX_PAGES})')
    parser.add_argument('--skip-scrape', action='store_true',
                        help='Skip scraping and use existing data')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'Port to run the web server on (default: {DEFAULT_PORT})')
    return parser.parse_args()


class VehicleDataManager:
    """Class to handle all data operations for the vehicle analysis"""
    
    def __init__(self, output_dir):
        """Initialize with output directory path"""
        self.output_dir = Path(output_dir)
        self.data = None
    
    def scrape_data(self, manufacturer, model, max_pages):
        """Run the scraper to collect vehicle data with rate limiting"""
        logger.info(f"Scraping data for manufacturer={manufacturer}, model={model}...")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scraper
        scraper = VehicleScraper(str(self.output_dir), manufacturer, model)
        
        # Scrape with rate limiting
        pages_scraped = 0
        max_retries = 3
        
        for page_num in range(1, max_pages + 1):
            retries = 0
            success = False
            
            while retries < max_retries and not success:
                try:
                    logger.info(f"Scraping page {page_num} of {max_pages}")
                    scraper.scrape_page(page_num)
                    success = True
                    pages_scraped += 1
                    
                    # Implement rate limiting
                    if page_num < max_pages:
                        delay = 2 + np.random.random() * 3  # Random delay between 2-5 seconds
                        logger.info(f"Waiting {delay:.1f} seconds before next request")
                        time.sleep(delay)
                        
                except Exception as e:
                    retries += 1
                    logger.warning(f"Error scraping page {page_num}: {str(e)}. Retry {retries}/{max_retries}")
                    time.sleep(5)  # Wait longer after an error
            
            if not success:
                logger.error(f"Failed to scrape page {page_num} after {max_retries} attempts")
        
        logger.info(f"Scraping complete. Scraped {pages_scraped} pages.")
    
    def process_data(self):
        """Process the scraped HTML files into a CSV"""
        logger.info("Processing scraped HTML files...")
        dir_name = self.output_dir.name
        
        try:
            yad2_parser.process_directory(str(self.output_dir))
            output_file = f"{dir_name}_summary.csv"
            output_path = self.output_dir / output_file
            
            # Check if the CSV file exists
            if not output_path.exists():
                logger.error(f"Could not find processed data at {output_path}")
                raise FileNotFoundError(f"Processed data file not found: {output_path}")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def load_data(self, csv_path):
        """Load and prepare the CSV data for visualization"""
        try:
            logger.info(f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Basic data cleaning
            logger.info("Cleaning and preparing data")
            
            # Filter out cars with no price or price = 0
            df = df[df['price'] > 0].copy()
            
            # Convert date strings to datetime objects
            df['productionDate'] = pd.to_datetime(df['productionDate'])
            
            # Extract year from production date for easier filtering
            df['productionYear'] = df['productionDate'].dt.year
            
            # Add any additional data preparation steps here
            
            self.data = df
            logger.info(f"Loaded {len(df)} vehicle records")
            
            # Clean up the CSV file after loading
            try:
                os.unlink(csv_path)
                logger.info(f"Removed temporary CSV file: {csv_path}")
            except Exception as e:
                logger.warning(f"Could not remove temporary CSV file: {str(e)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


class VehicleDashboard:
    """Class to handle the interactive Dash dashboard for vehicle price analysis"""
    
    def __init__(self, data_df, port=DEFAULT_PORT):
        """Initialize with DataFrame and port number"""
        self.df = data_df
        self.port = port
        self.app = None
        self.styles = self._define_styles()
    
    def _define_styles(self):
        """Define CSS styles for the dashboard"""
        return {
            'container': {
                'font-family': 'Roboto, sans-serif',
                'max-width': '1200px',
                'margin': '0 auto',
                'padding': '20px',
                'background-color': COLORS['light'],
                'border-radius': '8px',
                'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'
            },
            'header': {
                'background-color': COLORS['primary'],
                'color': 'white',
                'padding': '15px 20px',
                'margin-bottom': '20px',
                'border-radius': '5px',
                'text-align': 'center'
            },
            'filter_container': {
                'display': 'flex',
                'flex-wrap': 'wrap',
                'gap': '15px',
                'background-color': 'white',
                'padding': '15px',
                'border-radius': '5px',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.05)',
                'margin-bottom': '20px'
            },
            'filter': {
                'width': '23%',
                'min-width': '200px',
                'padding': '10px'
            },
            'label': {
                'font-weight': 'bold',
                'margin-bottom': '5px',
                'color': COLORS['primary']
            },
            'graph': {
                'background-color': 'white',
                'padding': '15px',
                'border-radius': '5px',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.05)',
                'margin-bottom': '20px'
            },
            'summary': {
                'background-color': 'white',
                'padding': '15px',
                'border-radius': '5px',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.05)'
            },
            'summary_header': {
                'color': COLORS['primary'],
                'border-bottom': f'2px solid {COLORS["secondary"]}',
                'padding-bottom': '10px',
                'margin-bottom': '15px'
            },
            'button': {
                'background-color': COLORS['primary'],
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'border-radius': '5px',
                'cursor': 'pointer',
                'font-weight': 'bold',
                'margin-top': '10px',
                'width': '100%'
            },
            'clear_button': {
                'background-color': COLORS['danger'],
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'border-radius': '5px',
                'cursor': 'pointer',
                'font-weight': 'bold',
                'margin-top': '10px',
                'width': '100%'
            },
            'click_instruction': {
                'text-align': 'center',
                'font-style': 'italic',
                'color': COLORS['secondary'],
                'margin': '10px 0',
                'padding': '8px',
                'background-color': '#f0f7ff',
                'border-radius': '5px',
                'border-left': f'3px solid {COLORS["secondary"]}'
            },
            'summary_card': {
                'container': {
                    'display': 'flex',
                    'flex-wrap': 'wrap',
                    'gap': '20px'
                },
                'card': {
                    'flex': '1',
                    'min-width': '180px',
                    'padding': '15px',
                    'border-radius': '5px',
                    'background-color': '#f5f9ff',
                    'box-shadow': '0 2px 4px rgba(0,0,0,0.05)',
                    'text-align': 'center'
                },
                'value': {
                    'font-size': '20px',
                    'font-weight': 'bold',
                    'color': COLORS['primary'],
                    'margin': '10px 0'
                },
                'label': {
                    'font-size': '14px',
                    'color': '#7f8c8d',
                    'margin': '0'
                }
            }
        }
    
    def _get_filter_options(self):
        """Generate options for dashboard filters"""
        # Kilometer range options
        km_ranges = [
            {'label': 'All', 'value': 'all'},
            {'label': '≤ 10,000 km/year', 'value': '0-10000'},
            {'label': '≤ 15,000 km/year', 'value': '0-15000'},
            {'label': '≤ 20,000 km/year', 'value': '0-20000'},
            {'label': '≤ 25,000 km/year', 'value': '0-25000'},
            {'label': '> 25,000 km/year', 'value': '25000-999999'}
        ]
        
        # Hand options (filter by previous owners)
        hands = [{'label': 'All Hands', 'value': 'all'}]
        for h in sorted(self.df['hand'].unique()):
            if h > 0:
                hands.append({'label': f'Hand ≤ {h}', 'value': f'0-{h}'})
        
        # Model options
        models = [{'label': m, 'value': m} for m in sorted(self.df['model'].unique())]
        
        # Ad type options
        ad_types = [{'label': 'All', 'value': 'all'}]
        for at in sorted(self.df['listingType'].unique()):
            ad_types.append({'label': at, 'value': at})
        
        return {
            'km_ranges': km_ranges,
            'hands': hands,
            'models': models,
            'ad_types': ad_types
        }
    
    def _create_app_layout(self, filter_options):
        """Create the Dash app layout"""
        return html.Div([
            # Header
            html.Div([
                html.H1("Vehicle Price Analysis Dashboard", style={'margin': '0'})
            ], style=self.styles['header']),
            
            # Filter section
            html.Div([
                # KM per year filter
                html.Div([
                    html.Label("Filter by km/year:", style=self.styles['label']),
                    dcc.Dropdown(
                        id='km-filter',
                        options=filter_options['km_ranges'],
                        value='all',
                        clearable=False
                    ),
                ], style=self.styles['filter']),
                
                # Owner hand filter
                html.Div([
                    html.Label("Filter by owner hand:", style=self.styles['label']),
                    dcc.Dropdown(
                        id='hand-filter',
                        options=filter_options['hands'],
                        value='all',
                        clearable=False
                    ),
                ], style=self.styles['filter']),
                
                # Model filter
                html.Div([
                    html.Label("Filter by model:", style=self.styles['label']),
                    dcc.Dropdown(
                        id='model-filter',
                        options=filter_options['models'],
                        value=[],
                        multi=True,
                        placeholder="Select model(s)"
                    ),
                ], style=self.styles['filter']),
                
                # Listing type filter
                html.Div([
                    html.Label("Filter by listing type:", style=self.styles['label']),
                    dcc.Dropdown(
                        id='adtype-filter',
                        options=filter_options['ad_types'],
                        value='all',
                        clearable=False
                    ),
                ], style=self.styles['filter']),

                # Submodel filter section
                html.Div([
                    html.Label("Filter by sub-model:", style=self.styles['label']),
                    html.Div([
                        dcc.Checklist(
                            id='submodel-checklist',
                            options=[],  # Will be populated dynamically
                            value=[],
                            labelStyle={'display': 'block', 'margin-bottom': '8px', 'cursor': 'pointer'},
                            style={'max-height': '200px', 'overflow-y': 'auto', 'padding': '10px', 
                                  'background-color': '#f5f9ff', 'border-radius': '5px'}
                        ),
                    ]),
                    html.Div([
                        html.Button(
                            'Apply Filters', 
                            id='apply-submodel-button', 
                            style=self.styles['button']
                        ),
                        html.Button(
                            'Clear Selection', 
                            id='clear-submodel-button', 
                            style=self.styles['clear_button']
                        ),
                    ], style={'display': 'flex', 'gap': '10px'}),
                ], style={'width': '23%', 'min-width': '200px', 'padding': '10px', 'flex-grow': '1'}),
                
            ], style=self.styles['filter_container']),
            
            # Click instruction
            html.Div([
                html.P("👆 Click on any point in the graph to open the vehicle ad in a new tab")
            ], style=self.styles['click_instruction']),
            
            # Graph section
            html.Div([
                dcc.Graph(id='price-date-scatter')
            ], style=self.styles['graph']),
            
            # Summary section
            html.Div([
                html.H3("Data Summary", style=self.styles['summary_header']),
                html.Div(id='summary-stats')
            ], style=self.styles['summary']),
            
            # Store for clicked links
            dcc.Store(id='clicked-link', storage_type='memory'),
        ], style=self.styles['container'])
    
    def _fit_trend_curve(self, x_data, y_data):
        """Fit an exponential trend curve to the data with fallbacks
        
        Args:
            x_data (array): The x values (days since newest)
            y_data (array): The y values (prices)
            
        Returns:
            tuple: (x_curve, y_curve, curve_type) for plotting
        """
        # Ensure we have valid numeric data
        valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
        x = x_data[valid_indices]
        y = y_data[valid_indices]
        
        if len(x) <= 1:
            logger.warning("Not enough data points for curve fitting")
            return None, None, None
        
        # Try different curve fitting approaches with fallbacks
        try:
            # 1. First try: Full exponential decay model with residual value
            # Price(t) = Base_Price * exp(-decay_rate * t) + Residual_Value
            def exp_decay_with_offset(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            # Calculate initial parameters and bounds
            max_price = np.max(y)
            mean_price = np.mean(y)
            min_price = np.min(y)
            
            p0 = [max_price - min_price, 0.001, min_price]
            bounds = ([0, 0.0001, 0], [2 * max_price, 0.1, mean_price])
            
            params, _ = optimize.curve_fit(
                exp_decay_with_offset, x, y, 
                p0=p0, bounds=bounds, 
                method='trf', maxfev=10000
            )
            
            a, b, c = params
            x_curve = np.linspace(0, x.max(), 200)
            y_curve = exp_decay_with_offset(x_curve, a, b, c)
            curve_type = 'exponential'
            
            logger.info("Successfully fit exponential decay curve with offset")
            return x_curve, y_curve, curve_type
            
        except Exception as e1:
            logger.warning(f"Primary curve fitting failed: {str(e1)}")
            
            try:
                # 2. Second try: Simpler exponential model without offset
                def exp_decay(x, a, b):
                    return a * np.exp(-b * x)
                
                p0_simple = [max_price, 0.001]
                bounds_simple = ([0, 0.0001], [2 * max_price, 0.1])
                
                params, _ = optimize.curve_fit(
                    exp_decay, x, y, 
                    p0=p0_simple, bounds=bounds_simple, 
                    method='trf', maxfev=10000
                )
                
                a, b = params
                x_curve = np.linspace(0, x.max(), 200)
                y_curve = exp_decay(x_curve, a, b)
                curve_type = 'simple_exponential'
                
                logger.info("Successfully fit simple exponential decay curve")
                return x_curve, y_curve, curve_type
                
            except Exception as e2:
                logger.warning(f"Secondary curve fitting failed: {str(e2)}")
                
                try:
                    # 3. Third try: Log-linear approach (log transform then linear fit)
                    log_y = np.log(y)
                    valid = np.isfinite(log_y)
                    x_valid = x[valid]
                    log_y_valid = log_y[valid]
                    
                    if len(x_valid) > 1:
                        z = np.polyfit(x_valid, log_y_valid, 1)
                        a = np.exp(z[1])
                        b = -z[0]
                        
                        x_curve = np.linspace(0, x.max(), 200)
                        y_curve = a * np.exp(-b * x_curve)
                        curve_type = 'log_linear'
                        
                        logger.info("Successfully fit log-linear exponential curve")
                        return x_curve, y_curve, curve_type
                
                except Exception as e3:
                    logger.warning(f"Log-linear curve fitting failed: {str(e3)}")
                    
                    try:
                        # 4. Last resort: Simple linear fit
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        
                        x_curve = np.linspace(0, x.max(), 200)
                        y_curve = p(x_curve)
                        curve_type = 'linear'
                        
                        logger.info("Falling back to linear trend")
                        return x_curve, y_curve, curve_type
                        
                    except Exception as e4:
                        logger.error(f"All curve fitting methods failed: {str(e4)}")
                        return None, None, None
    
    def _create_scatter_plot(self, filtered_df):
        """Create the price scatter plot with trend line"""
        # Calculate days since newest car for each data point
        newest_date = filtered_df['productionDate'].max()
        filtered_df['days_since_newest'] = (newest_date - filtered_df['productionDate']).dt.days
        
        # Calculate display dates
        today = pd.Timestamp.today().normalize()
        filtered_df['display_date'] = today - pd.to_timedelta(filtered_df['days_since_newest'], unit='D')
        
        # Create basic scatter plot
        fig = px.scatter(
            filtered_df, 
            x='display_date', 
            y='price',
            color='km_per_year',
            size_max=8,
            color_continuous_scale='viridis',
            range_color=[0, filtered_df['km_per_year'].quantile(0.95)],
            hover_data=['model', 'subModel', 'hand', 'km', 'city', 'productionDate', 'link'],
            labels={'display_date': 'Date', 
                   'price': 'Price (₪)', 
                   'km_per_year': 'Kilometers per Year'},
            title=f'Vehicle Prices by Age ({len(filtered_df)} vehicles)'
        )
        
        # Create custom data for hover and click functionality
        custom_data = np.column_stack((
            filtered_df['model'], 
            filtered_df['subModel'], 
            filtered_df['hand'], 
            filtered_df['km'], 
            filtered_df['city'],
            filtered_df['productionDate'],
            filtered_df['link']
        ))
        
        # Update trace properties
        fig.update_traces(
            marker=dict(
                size=8,
                opacity=0.8,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            customdata=custom_data,
            hovertemplate='<b>%{customdata[0]} %{customdata[1]}</b><br>' +
                         'Price: ₪%{y:,.0f}<br>' +
                         'Production Date: %{customdata[5]}<br>' +
                         'Hand: %{customdata[2]}<br>' +
                         'KM: %{customdata[3]:,.0f}<br>' +
                         'City: %{customdata[4]}<br>' +
                         '<b>👆 Click to view ad</b>'
        )
        
        # Improve layout and appearance
        fig.update_layout(
            clickmode='event+select',
            hoverdistance=100,
            hovermode='closest',
            dragmode='zoom',
            plot_bgcolor='rgba(240,240,240,0.2)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Roboto, sans-serif"),
            xaxis=dict(
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor='#eee',
                autorange="reversed"
            ),
            yaxis=dict(
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor='#eee'
            ),
            title=dict(
                font=dict(size=16)
            ),
            legend=dict(
                title_font=dict(size=13),
                font=dict(size=11)
            ),
            coloraxis_colorbar=dict(
                title="Km/Year",
                title_font=dict(size=13),
                tickfont=dict(size=11)
            ),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        # Add trend curve if we have enough data points
        if len(filtered_df) > 1:
            # Get the x and y data for curve fitting
            x_data = filtered_df['days_since_newest'].values
            y_data = filtered_df['price'].values
            
            # Fit the curve
            x_curve, y_curve, curve_type = self._fit_trend_curve(x_data, y_data)
            
            if x_curve is not None and y_curve is not None:
                # Get curve name based on type
                curve_names = {
                    'exponential': 'Exponential Trend',
                    'simple_exponential': 'Exponential Trend',
                    'log_linear': 'Exponential Trend (Simplified)',
                    'linear': 'Linear Trend'
                }
                
                curve_colors = {
                    'exponential': 'red',
                    'simple_exponential': 'red',
                    'log_linear': 'orange',
                    'linear': 'orange'
                }
                
                curve_styles = {
                    'exponential': 'solid',
                    'simple_exponential': 'solid',
                    'log_linear': 'dash',
                    'linear': 'dash'
                }
                
                # Add the trend line
                curve_name = curve_names.get(curve_type, 'Trend Line')
                curve_color = curve_colors.get(curve_type, 'red')
                curve_style = curve_styles.get(curve_type, 'solid')
                
                # Convert x_curve from days to actual dates for plotting
                curve_dates = today - pd.to_timedelta(x_curve, unit='D')
                
                fig.add_trace(go.Scatter(
                    x=curve_dates,
                    y=y_curve,
                    mode='lines',
                    name=curve_name,
                    line=dict(color=curve_color, width=3, dash=curve_style),
                    hoverinfo='none'
                ))
        
        return fig
    
    def _create_summary_stats(self, filtered_df):
        """Create summary statistics component"""
        # Get style for summary cards
        summary_style = self.styles['summary_card']
        
        # Create the summary stats cards
        return html.Div([
            html.Div([
                html.P("Number of Vehicles", style=summary_style['label']),
                html.P(f"{len(filtered_df)}", style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Average Price", style=summary_style['label']),
                html.P(f"₪{filtered_df['price'].mean():,.0f}", style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Price Range", style=summary_style['label']),
                html.P(f"₪{filtered_df['price'].min():,.0f} - ₪{filtered_df['price'].max():,.0f}", 
                       style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Average km/year", style=summary_style['label']),
                html.P(f"{filtered_df['km_per_year'].mean():,.0f}", style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Average Vehicle Age", style=summary_style['label']),
                html.P(f"{filtered_df['number_of_years'].mean():.1f} years", style=summary_style['value'])
            ], style=summary_style['card']),
        ], style=summary_style['container'])
    
    def _setup_callbacks(self):
        """Set up all dashboard callbacks"""
        @self.app.callback(
            Output('submodel-checklist', 'options'),
            Input('model-filter', 'value'),
        )
        def update_submodel_options(selected_models):
            if not selected_models or len(selected_models) == 0:
                # If no models selected, show all submodels with model prefix
                submodel_options = []
                for sm in sorted(self.df['subModel'].unique()):
                    models_for_submodel = self.df[self.df['subModel'] == sm]['model'].unique()
                    if len(models_for_submodel) == 1:
                        label = f"[{models_for_submodel[0]}] {sm}"
                    else:
                        label = f"[{models_for_submodel[0]}+] {sm}"
                    submodel_options.append({'label': label, 'value': sm})
            else:
                # Filter submodels based on selected models
                filtered_df = self.df[self.df['model'].isin(selected_models)]
                submodel_options = []
                for sm in sorted(filtered_df['subModel'].unique()):
                    models_for_submodel = filtered_df[filtered_df['subModel'] == sm]['model'].unique()
                    if len(models_for_submodel) == 1:
                        label = f" {sm} [{models_for_submodel[0]}]"
                    else:
                        models_str = '+'.join(models_for_submodel)
                        label = f" {sm} [{models_str}]"
                    submodel_options.append({'label': label, 'value': sm})
            
            return list(sorted(submodel_options, key=lambda x: x['label']))
        
        @self.app.callback(
            Output('submodel-checklist', 'value'),
            Input('clear-submodel-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def clear_submodel_selection(n_clicks):
            return []
        
        @self.app.callback(
            [Output('price-date-scatter', 'figure'),
             Output('summary-stats', 'children')],
            [Input('km-filter', 'value'),
             Input('hand-filter', 'value'),
             Input('model-filter', 'value'),
             Input('apply-submodel-button', 'n_clicks'),
             Input('adtype-filter', 'value')],
            [State('submodel-checklist', 'value')]
        )
        def update_graph_and_stats(km_range, hand, models, submodel_btn_clicks, adtype, submodel_list):
            # Apply filters
            filtered_df = self.df.copy()
            
            # Apply km/year filter
            if km_range != 'all':
                min_km, max_km = map(int, km_range.split('-'))
                filtered_df = filtered_df[filtered_df['km_per_year'] <= max_km]
                if min_km > 0:  # For the "> 25,000" filter
                    filtered_df = filtered_df[filtered_df['km_per_year'] > min_km]
            
            # Apply hand (previous owners) filter
            if hand != 'all':
                min_hand, max_hand = map(int, hand.split('-'))
                filtered_df = filtered_df[filtered_df['hand'] <= max_hand]
            
            # Apply model filter
            if models and len(models) > 0:
                filtered_df = filtered_df[filtered_df['model'].isin(models)]
                
            # Apply submodel filter
            if submodel_list and len(submodel_list) > 0:
                filtered_df = filtered_df[filtered_df['subModel'].isin(submodel_list)]
                
            # Apply ad type filter
            if adtype != 'all':
                filtered_df = filtered_df[filtered_df['listingType'] == adtype]
            
            # Create scatter plot
            fig = self._create_scatter_plot(filtered_df)
            
            # Create summary statistics
            summary = self._create_summary_stats(filtered_df)
            
            return fig, summary
        
        # Client-side callback to open links in new tab
        self.app.clientside_callback(
            """
            function(clickData) {
                if(clickData && clickData.points && clickData.points.length > 0) {
                    const link = clickData.points[0].customdata[6];
                    if(link && link.length > 0) {
                        window.open(link, '_blank');
                    }
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output('clicked-link', 'data'),
            Input('price-date-scatter', 'clickData'),
            prevent_initial_call=True
        )

    def create_and_run(self):
        """Create and run the dashboard"""
        logger.info(f"Creating dashboard on port {self.port}")
        
        # Create the Dash app
        self.app = dash.Dash(
            __name__, 
            title="Vehicle Price Analyzer",
            external_stylesheets=EXTERNAL_STYLESHEETS,
            suppress_callback_exceptions=True
        )
        
        # Get filter options
        filter_options = self._get_filter_options()
        
        # Create app layout
        self.app.layout = self._create_app_layout(filter_options)
        
        # Set up callbacks
        self._setup_callbacks()
        
        # Run the app
        logger.info(f"Starting dashboard on http://127.0.0.1:{self.port}/")
        self.app.run_server(debug=False, port=self.port)


def main():
    """Main function to run the Vehicle Price Analyzer"""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize data manager
        data_manager = VehicleDataManager(args.output_dir)
        
        # Step 1: Scrape the data if not skipped
        if not args.skip_scrape:
            data_manager.scrape_data(args.manufacturer, args.model, args.max_pages)
        
        # Step 2: Process the scraped data
        csv_path = data_manager.process_data()
        
        # Step 3: Load the data
        df = data_manager.load_data(csv_path)
        
        # Step 4: Create and run the dashboard
        dashboard = VehicleDashboard(df, args.port)
        dashboard.create_and_run()
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
