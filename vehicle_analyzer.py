# vehicle_analyzer.py
"""
Vehicle Price Analyzer - A tool to scrape, process and visualize vehicle price data.

This application:
1. Scrapes vehicle listings from Yad2 (an Israeli classifieds site)
2. Processes the HTML files into structured data
3. Creates an interactive dashboard to analyze vehicle pricing trends

Usage:
    python vehicle_price_analyzer.py [options]

Options:
    --output-dir DIR      Directory to save scraped data (default: 'scraped_vehicles')
    --manufacturer ID     Manufacturer ID to scrape (default: 38)
    --model ID            Model ID to scrape (default: 10514)
    --max-pages NUM       Maximum number of pages to scrape (default: 25)
    --skip-scrape         Skip scraping and use existing data
    --port NUM            Port to run the web server on (default: 8050)
    --list-models         List all available manufacturers and models
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
import json

# Import the scraper modules
from scraper import VehicleScraper
import yad2_parser
import vehicle_models

# For web visualization
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash.exceptions import PreventUpdate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vehicle_analyzer')

# Dashboard colors
COLORS = {
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'danger': '#e74c3c',
    'light': '#f9f9f9',
    'text': '#2c3e50'
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vehicle Price Analyzer')
    parser.add_argument('--output-dir', type=str, default='scraped_vehicles',
                        help='Directory to save scraped data')
    parser.add_argument('--manufacturer', type=int, default=38,
                        help='Manufacturer ID to scrape')
    parser.add_argument('--model', type=int, default=10514,
                        help='Model ID to scrape')
    parser.add_argument('--max-pages', type=int, default=25,
                        help='Maximum number of pages to scrape')
    parser.add_argument('--skip-scrape', action='store_true',
                        help='Skip scraping and use existing data')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run the web server on')
    parser.add_argument('--list-models', action='store_true',
                        help='List all available manufacturers and models')
    return parser.parse_args()

def scrape_data(output_dir, manufacturer, model, max_pages):
    """Run the scraper to collect vehicle data"""
    logger.info(f"Scraping data for manufacturer={manufacturer}, model={model}...")
    
    try:
        # Initialize the scraper with the provided parameters
        scraper = VehicleScraper(output_dir, manufacturer, model)
        
        # Run the scraper with the maximum number of pages
        scraper.scrape_pages(max_page=max_pages)
        
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise
    
def process_data(output_dir):
    """Process the scraped HTML files into a CSV"""
    logger.info("Processing scraped HTML files...")
    
    try:
        # Get the directory name for output file naming
        dir_name = Path(output_dir).name
        
        # Process the HTML files in the directory
        yad2_parser.process_directory(output_dir)
        
        # Build the output file path
        output_file = f"{dir_name}_summary.csv"
        output_path = os.path.join(output_dir, output_file)
        
        # Check if the CSV file exists
        if not os.path.exists(output_path):
            logger.error(f"Could not find processed data at {output_path}")
            raise FileNotFoundError(f"Processed data file not found: {output_path}")
            
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def load_data(csv_path):
    """Load and prepare the CSV data for visualization"""
    try:
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Filter out cars with no price or price = 0
        df = df[df['price'] > 0]
        
        # Convert date strings to datetime objects
        df['productionDate'] = pd.to_datetime(df['productionDate'])
        
        # Extract year from production date for easier filtering
        df['productionYear'] = df['productionDate'].dt.year
        df['productionMonth'] = df['productionDate'].dt.month
        
        # Format production date for display
        df['productionDateFormatted'] = df['productionDate'].dt.strftime('%Y-%m-%d')
        
        logger.info(f"Loaded {len(df)} vehicle records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def define_dashboard_styles():
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
        },
        'tabs': {
            'margin-bottom': '20px'
        }
    }

def get_filter_options(df):
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
    for h in sorted(df['hand'].unique()):
        if h > 0:
            hands.append({'label': f'Hand ≤ {h}', 'value': f'0-{h}'})
    
    # Sub-model options
    sub_models = [{'label': 'All Sub-models', 'value': 'all'}]
    for sm in sorted(df['subModel'].unique()):
        sub_models.append({'label': sm, 'value': sm})
    
    # Model options
    models = [{'label': m, 'value': m} for m in sorted(df['model'].unique())]
    
    # Ad type options
    ad_types = [{'label': 'All', 'value': 'all'}]
    for at in sorted(df['listingType'].unique()):
        ad_types.append({'label': at, 'value': at})
    
    return {
        'km_ranges': km_ranges,
        'hands': hands,
        'sub_models': sub_models,
        'models': models,
        'ad_types': ad_types
    }

def fit_trend_curve(x_data, y_data):
    """Fit an exponential trend curve with fallback options
    
    Args:
        x_data: Array of x values (days since newest)
        y_data: Array of y values (prices)
        
    Returns:
        tuple: (x_curve, y_curve, curve_type) if successful, (None, None, None) otherwise
    """
    from scipy import optimize
    
    # Ensure valid data
    valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
    x = x_data[valid_indices]
    y = y_data[valid_indices]
    
    if len(x) <= 1:
        logger.warning("Not enough data points for curve fitting")
        return None, None, None
    
    try:
        # Try exponential decay with offset: a * exp(-b * x) + c
        def exp_decay_with_offset(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        # Calculate parameters
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
        
        return x_curve, y_curve, 'exponential'
        
    except Exception as e1:
        logger.warning(f"Primary curve fitting failed: {str(e1)}")
        
        try:
            # Simple exponential model: a * exp(-b * x)
            def exp_decay(x, a, b):
                return a * np.exp(-b * x)
            
            p0 = [max_price, 0.001]
            bounds = ([0, 0.0001], [2 * max_price, 0.1])
            
            params, _ = optimize.curve_fit(
                exp_decay, x, y, 
                p0=p0, bounds=bounds, 
                method='trf', maxfev=10000
            )
            
            a, b = params
            x_curve = np.linspace(0, x.max(), 200)
            y_curve = exp_decay(x_curve, a, b)
            
            return x_curve, y_curve, 'simple_exponential'
            
        except Exception as e2:
            logger.warning(f"Secondary curve fitting failed: {str(e2)}")
            
            try:
                # Basic exponential fit using log transform
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
                    
                    return x_curve, y_curve, 'log_linear'
                
            except Exception as e3:
                logger.warning(f"Log-linear curve fitting failed: {str(e3)}")
                
                try:
                    # Last resort: linear fit
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    x_curve = np.linspace(0, x.max(), 200)
                    y_curve = p(x_curve)
                    
                    return x_curve, y_curve, 'linear'
                    
                except Exception as e4:
                    logger.error(f"All curve fitting methods failed: {str(e4)}")
    
    return None, None, None

def create_scatter_plot_by_date(filtered_df):
    """Create scatter plot with trend line by production date
    
    Args:
        filtered_df: DataFrame with filtered vehicle data
        
    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    # Calculate days since newest for each data point
    newest_date = filtered_df['productionDate'].max()
    filtered_df['days_since_newest'] = (newest_date - filtered_df['productionDate']).dt.days
    
    # Create basic scatter plot
    fig = px.scatter(
        filtered_df, 
        x='productionDate',  # Using actual production date
        y='price',
        color='km_per_year',
        size_max=8,
        color_continuous_scale='viridis',
        range_color=[0, filtered_df['km_per_year'].quantile(0.95)],
        hover_data=['model', 'subModel', 'hand', 'km', 'city', 'productionDateFormatted', 'link'],
        labels={'productionDate': 'Production Date', 
               'price': 'Price (₪)', 
               'km_per_year': 'Kilometers per Year'},
        title=f'Vehicle Prices by Production Date ({len(filtered_df)} vehicles)'
    )
    
    # Create custom data for hover and click functionality
    custom_data = np.column_stack((
        filtered_df['model'], 
        filtered_df['subModel'], 
        filtered_df['hand'], 
        filtered_df['km'], 
        filtered_df['city'],
        filtered_df['productionDateFormatted'],
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
            dtick="M3",  # 3-month intervals
            tickformat="%Y-%m"  # Year-month format
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
        x_curve, y_curve, curve_type = fit_trend_curve(x_data, y_data)
        
        if x_curve is not None and y_curve is not None:
            # Get curve name and style based on type
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
            curve_dates = newest_date - pd.to_timedelta(x_curve, unit='D')
            
            fig.add_trace(go.Scatter(
                x=curve_dates,
                y=y_curve,
                mode='lines',
                name=curve_name,
                line=dict(color=curve_color, width=3, dash=curve_style),
                hoverinfo='none'
            ))
    
    return fig

def create_scatter_plot_by_km(filtered_df):
    """Create scatter plot with trend line by kilometers
    
    Args:
        filtered_df: DataFrame with filtered vehicle data
        
    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    # Create basic scatter plot
    fig = px.scatter(
        filtered_df, 
        x='km',
        y='price',
        color='number_of_years',
        size_max=8,
        color_continuous_scale='viridis',
        range_color=[0, filtered_df['number_of_years'].quantile(0.95)],
        hover_data=['model', 'subModel', 'hand', 'productionDateFormatted', 'city', 'link'],
        labels={'km': 'Kilometers', 
               'price': 'Price (₪)', 
               'number_of_years': 'Vehicle Age (Years)'},
        title=f'Vehicle Prices by Kilometers ({len(filtered_df)} vehicles)'
    )
    
    # Create custom data for hover and click functionality
    custom_data = np.column_stack((
        filtered_df['model'], 
        filtered_df['subModel'], 
        filtered_df['hand'], 
        filtered_df['productionDateFormatted'], 
        filtered_df['city'],
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
                     'Kilometers: %{x:,.0f}<br>' +
                     'Production Date: %{customdata[3]}<br>' +
                     'Hand: %{customdata[2]}<br>' +
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
            tickformat=",d"  # Comma-separated thousands
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
            title="Age (Years)",
            title_font=dict(size=13),
            tickfont=dict(size=11)
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Add trend curve if we have enough data points
    if len(filtered_df) > 1:
        try:
            # Fit a polynomial regression line
            from scipy.stats import linregress
            from scipy import optimize
            
            # Sort by kilometers for the trend line
            sorted_df = filtered_df.sort_values('km')
            x = sorted_df['km'].values
            y = sorted_df['price'].values
            
            # Try to fit a polynomial regression
            try:
                # Try exponential decay with offset: a * exp(-b * x) + c
                def exp_decay_with_offset(x, a, b, c):
                    return a * np.exp(-b * x) + c
                
                # Calculate parameters
                max_price = np.max(y)
                mean_price = np.mean(y)
                min_price = np.min(y)
                
                p0 = [max_price - min_price, 0.00001, min_price]
                bounds = ([0, 0.0000001, 0], [2 * max_price, 0.001, mean_price])
                
                params, _ = optimize.curve_fit(
                    exp_decay_with_offset, x, y, 
                    p0=p0, bounds=bounds, 
                    method='trf', maxfev=10000
                )
                
                a, b, c = params
                x_smooth = np.linspace(min(x), max(x), 200)
                y_smooth = exp_decay_with_offset(x_smooth, a, b, c)
                
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    name='Exponential Trend',
                    line=dict(color='red', width=3),
                    hoverinfo='none'
                ))
            except:
                # Fallback to polynomial
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                
                x_smooth = np.linspace(min(x), max(x), 200)
                y_smooth = p(x_smooth)
                
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    name='Polynomial Trend',
                    line=dict(color='red', width=3),
                    hoverinfo='none'
                ))
                
        except Exception as e:
            logger.warning(f"Error fitting trend curve for km plot: {str(e)}")
    
    return fig

def create_summary_stats(filtered_df, styles):
    """Create summary statistics component
    
    Args:
        filtered_df: DataFrame with filtered vehicle data
        styles: Dictionary with styling information
        
    Returns:
        dash component with summary statistics
    """
    style = styles['summary_card']
    
    return html.Div([
        html.Div([
            html.P("Number of Vehicles", style=style['label']),
            html.P(f"{len(filtered_df)}", style=style['value'])
        ], style=style['card']),
        
        html.Div([
            html.P("Average Price", style=style['label']),
            html.P(f"₪{filtered_df['price'].mean():,.0f}", style=style['value'])
        ], style=style['card']),
        
        html.Div([
            html.P("Price Range", style=style['label']),
            html.P(f"₪{filtered_df['price'].min():,.0f} - ₪{filtered_df['price'].max():,.0f}", 
                  style=style['value'])
        ], style=style['card']),
        
        html.Div([
            html.P("Average km/year", style=style['label']),
            html.P(f"{filtered_df['km_per_year'].mean():,.0f}", style=style['value'])
        ], style=style['card']),
        
        html.Div([
            html.P("Average Vehicle Age", style=style['label']),
            html.P(f"{filtered_df['number_of_years'].mean():.1f} years", style=style['value'])
        ], style=style['card']),
    ], style=style['container'])

def create_dashboard(df, manufacturer_name, model_name, port=8050):
    """Create and run an interactive Dash app for visualizing the data"""
    logger.info(f"Creating dashboard on port {port}")
    
    # Set up styles and options
    styles = define_dashboard_styles()
    filter_options = get_filter_options(df)
    
    # External stylesheets
    external_stylesheets = [
        {
            'href': 'https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap',
            'rel': 'stylesheet'
        }
    ]
    
    # Create the app
    app = dash.Dash(
        __name__, 
        title=f"{manufacturer_name} {model_name} - Price Analyzer",
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True
    )
    
    # Create the app layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1(f"{manufacturer_name} {model_name} Price Analysis", style={'margin': '0'})
        ], style=styles['header']),
        
        # Filter section
        html.Div([
            # KM per year filter
            html.Div([
                html.Label("Filter by km/year:", style=styles['label']),
                dcc.Dropdown(
                    id='km-filter',
                    options=filter_options['km_ranges'],
                    value='all',
                    clearable=False
                ),
            ], style=styles['filter']),
            
            # Owner hand filter
            html.Div([
                html.Label("Filter by owner hand:", style=styles['label']),
                dcc.Dropdown(
                    id='hand-filter',
                    options=filter_options['hands'],
                    value='all',
                    clearable=False
                ),
            ], style=styles['filter']),
            
            # Model filter
            html.Div([
                html.Label("Filter by model:", style=styles['label']),
                dcc.Dropdown(
                    id='model-filter',
                    options=filter_options['models'],
                    value=[],
                    multi=True,
                    placeholder="Select model(s)"
                ),
            ], style=styles['filter']),
            
            # Listing type filter
            html.Div([
                html.Label("Filter by listing type:", style=styles['label']),
                dcc.Dropdown(
                    id='adtype-filter',
                    options=filter_options['ad_types'],
                    value='all',
                    clearable=False
                ),
            ], style=styles['filter']),

            # Submodel filter section
            html.Div([
                html.Label("Filter by sub-model:", style=styles['label']),
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
                        style=styles['button']
                    ),
                    html.Button(
                        'Clear Selection', 
                        id='clear-submodel-button', 
                        style=styles['clear_button']
                    ),
                ], style={'display': 'flex', 'gap': '10px'}),
            ], style={'width': '23%', 'min-width': '200px', 'padding': '10px', 'flex-grow': '1'}),
            
        ], style=styles['filter_container']),
        
        # Click instruction
        html.Div([
            html.P("👆 Click on any point in the graph to open the vehicle ad in a new tab")
        ], style=styles['click_instruction']),
        
        # Tabs for different visualizations
        dcc.Tabs([
            dcc.Tab(label='Price by Date', children=[
                html.Div([
                    dcc.Graph(id='price-date-scatter')
                ], style=styles['graph']),
            ]),
            dcc.Tab(label='Price by Kilometers', children=[
                html.Div([
                    dcc.Graph(id='price-km-scatter')
                ], style=styles['graph']),
            ]),
        ], style=styles['tabs']),
        
        # Summary section
        html.Div([
            html.H3("Data Summary", style=styles['summary_header']),
            html.Div(id='summary-stats')
        ], style=styles['summary']),
        
        # Store for clicked links
        dcc.Store(id='clicked-link', storage_type='memory'),
    ], style=styles['container'])
    
    # Set up callbacks
    setup_callbacks(app, df, styles)
    
    # Run the app
    logger.info(f"Starting dashboard on http://127.0.0.1:{port}/")
    app.run_server(debug=False, port=port)

def setup_callbacks(app, df, styles):
    """Set up callbacks for the dashboard
    
    Args:
        app: Dash app instance
        df: DataFrame with vehicle data
        styles: Dictionary with styling information
    """
    # Callback to update submodel options based on selected models
    @app.callback(
        Output('submodel-checklist', 'options'),
        Input('model-filter', 'value'),
    )
    def update_submodel_options(selected_models):
        if not selected_models or len(selected_models) == 0:
            # If no models selected, show all submodels with model prefix
            submodel_options = []
            for sm in sorted(df['subModel'].unique()):
                models_for_submodel = df[df['subModel'] == sm]['model'].unique()
                if len(models_for_submodel) == 1:
                    label = f"[{models_for_submodel[0]}] {sm}"
                else:
                    label = f"[{models_for_submodel[0]}+] {sm}"
                submodel_options.append({'label': label, 'value': sm})
        else:
            # Filter submodels based on selected models
            filtered_df = df[df['model'].isin(selected_models)]
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
    
    # Callback to clear submodel selection
    @app.callback(
        Output('submodel-checklist', 'value'),
        Input('clear-submodel-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_submodel_selection(n_clicks):
        return []
    
    # Callback to update graphs and summary based on filters
    @app.callback(
        [Output('price-date-scatter', 'figure'),
         Output('price-km-scatter', 'figure'),
         Output('summary-stats', 'children')],
        [Input('km-filter', 'value'),
         Input('hand-filter', 'value'),
         Input('model-filter', 'value'),
         Input('apply-submodel-button', 'n_clicks'),
         Input('adtype-filter', 'value')],
        [State('submodel-checklist', 'value')]
    )
    def update_graphs_and_stats(km_range, hand, models, submodel_btn_clicks, adtype, submodel_list):
        # Apply filters
        filtered_df = df.copy()
        
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
        
        # Create scatter plots
        fig_date = create_scatter_plot_by_date(filtered_df)
        fig_km = create_scatter_plot_by_km(filtered_df)
        
        # Create summary statistics
        summary = create_summary_stats(filtered_df, styles)
        
        return fig_date, fig_km, summary
    
    # Client-side callback to open links in new tab
    app.clientside_callback(
        """
        function(clickData) {
            if(clickData && clickData.points && clickData.points.length > 0) {
                const link = clickData.points[0].customdata[6] || clickData.points[0].customdata[5];
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
    
    # Client-side callback to open links in new tab for km scatter plot
    app.clientside_callback(
        """
        function(clickData) {
            if(clickData && clickData.points && clickData.points.length > 0) {
                const link = clickData.points[0].customdata[5];
                if(link && link.length > 0) {
                    window.open(link, '_blank');
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('clicked-link', 'data', allow_duplicate=True),
        Input('price-km-scatter', 'clickData'),
        prevent_initial_call=True
    )

def list_all_models():
    """List all available manufacturers and models"""
    models_data = vehicle_models.get_all_models()
    
    # Format and print the data
    print("\n=== Available Manufacturers and Models ===\n")
    
    for manufacturer in sorted(models_data.keys()):
        manufacturer_id = models_data[manufacturer]['id']
        print(f"{manufacturer} (ID: {manufacturer_id}):")
        
        models = models_data[manufacturer]['models']
        for model in sorted(models.keys()):
            model_id = models[model]
            print(f"  - {model} (ID: {model_id})")
            print(f"    Command: python vehicle_analyzer.py --manufacturer {manufacturer_id} --model {model_id}")
        
        print()

def main():
    """Main function to run the Vehicle Price Analyzer"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Handle --list-models flag
        if args.list_models:
            list_all_models()
            return
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Scrape the data if not skipped
        if not args.skip_scrape:
            scrape_data(args.output_dir, args.manufacturer, args.model, args.max_pages)
        
        # Step 2: Process the scraped data
        csv_path = process_data(args.output_dir)
        
        # Step 3: Load the data
        df = load_data(csv_path)
        
        # Clean up the CSV file after loading
        try:
            os.unlink(csv_path)
            logger.info(f"Removed temporary CSV file: {csv_path}")
        except Exception as e:
            logger.warning(f"Could not remove temporary CSV file: {str(e)}")
        
        # Get manufacturer and model names
        manufacturer_name = df['make'].iloc[0] if 'make' in df.columns else "Vehicle"
        model_name = df['model'].iloc[0] if 'model' in df.columns else "Analysis"
        
        # Step 4: Create and run the dashboard
        create_dashboard(df, manufacturer_name, model_name, args.port)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

