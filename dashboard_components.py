# dashboard_components.py
import dash
from dash import dcc, html, dash_table
from dash.dash_table.Format import Format, Scheme, Group
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy import optimize

# צבעי לוח מחוונים
COLORS = {
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'danger': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'light': '#f9f9f9',
    'dark': '#343a40',
    'text': '#2c3e50',
    'border': '#dee2e6',
    'hover': '#e9ecef'
}

def define_dashboard_styles():
    """הגדרת סגנונות CSS ללוח המחוונים"""
    return {
        'container': {
            'font-family': 'Roboto, sans-serif',
            'max-width': '1400px',
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
        'success_button': {
            'background-color': COLORS['success'],
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
        },
        'data_table': {
            'table': {
                'background-color': 'white',
                'padding': '15px',
                'border-radius': '5px',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.05)',
                'margin-bottom': '20px'
            },
            'header': {
                'backgroundColor': COLORS['primary'],
                'color': 'white',
                'fontWeight': 'bold',
                'padding': '12px 15px',
                'textAlign': 'left'
            },
            'cell': {
                'padding': '12px 15px',
                'textAlign': 'left',
                'borderBottom': '1px solid #eee'
            }
        },
        'modal': {
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
        },
        'add_vehicle_container': {
            'backgroundColor': 'white',
            'padding': '15px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
            'marginBottom': '20px'
        },
        'price_range_container': {
            'display': 'flex',
            'alignItems': 'center',
            'gap': '10px',
            'margin-top': '10px'
        },
        'text_input': {
            'width': '100%',
            'padding': '8px 12px',
            'borderRadius': '4px',
            'border': f'1px solid {COLORS["border"]}',
            'fontSize': '14px'
        }
    }

def get_filter_options(df):
    """יצירת אפשרויות לפילטרים בלוח המחוונים"""
    # אפשרויות טווח קילומטרים
    km_ranges = [
        {'label': 'הכל', 'value': 'all'},
        {'label': '≤ 10,000 ק"מ/שנה', 'value': '0-10000'},
        {'label': '≤ 15,000 ק"מ/שנה', 'value': '0-15000'},
        {'label': '≤ 20,000 ק"מ/שנה', 'value': '0-20000'},
        {'label': '≤ 25,000 ק"מ/שנה', 'value': '0-25000'},
        {'label': '> 25,000 ק"מ/שנה', 'value': '25000-999999'}
    ]
    
    # אפשרויות יד (סינון לפי בעלים קודמים)
    hands = [{'label': 'כל הידיים', 'value': 'all'}]
    for h in sorted(df['hand'].unique()):
        if h > 0:
            hands.append({'label': f'יד ≤ {h}', 'value': f'0-{h}'})
    
    # אפשרויות תת-דגם
    sub_models = [{'label': 'כל תת-הדגמים', 'value': 'all'}]
    for sm in sorted(df['subModel'].unique()):
        sub_models.append({'label': sm, 'value': sm})
    
    # אפשרויות דגם
    models = [{'label': m, 'value': m} for m in sorted(df['model'].unique())]
    
    # אפשרויות סוג מודעה
    ad_types = [{'label': 'הכל', 'value': 'all'}]
    for at in sorted(df['listingType'].unique()):
        ad_types.append({'label': at, 'value': at})
    
    # אפשרויות סוג תיבת הילוכים
    if 'gearBox' in df.columns:
        transmissions = [{'label': 'הכל', 'value': 'all'}]
        for tr in sorted(df['gearBox'].unique()):
            transmissions.append({'label': tr, 'value': tr})
    else:
        transmissions = [{'label': 'הכל', 'value': 'all'}]
    
    return {
        'km_ranges': km_ranges,
        'hands': hands,
        'sub_models': sub_models,
        'models': models,
        'ad_types': ad_types,
        'transmissions': transmissions
    }

def fit_trend_curve(x_data, y_data):
    """התאמת עקומת מגמה מעריכית עם אפשרויות גיבוי
    
    Args:
        x_data: מערך ערכי x (ימים מאז החדש ביותר)
        y_data: מערך ערכי y (מחירים)
        
    Returns:
        tuple: (x_curve, y_curve, curve_type) אם מוצלח, (None, None, None) אחרת
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # וידוא נתונים תקפים
    valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
    x = x_data[valid_indices]
    y = y_data[valid_indices]
    
    if len(x) <= 1:
        logger.warning("אין מספיק נקודות נתונים להתאמת עקומה")
        return None, None, None
    
    try:
        # ניסיון להתאים דעיכה מעריכית עם היסט: a * exp(-b * x) + c
        def exp_decay_with_offset(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        # חישוב פרמטרים
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
        logger.warning(f"התאמת עקומה ראשונית נכשלה: {str(e1)}")
        
        try:
            # מודל מעריכי פשוט: a * exp(-b * x)
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
            logger.warning(f"התאמת עקומה משנית נכשלה: {str(e2)}")
            
            try:
                # התאמה מעריכית בסיסית באמצעות טרנספורמציית לוג
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
                logger.warning(f"התאמת עקומת log-linear נכשלה: {str(e3)}")
                
                try:
                    # מוצא אחרון: התאמה לינארית
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    x_curve = np.linspace(0, x.max(), 200)
                    y_curve = p(x_curve)
                    
                    return x_curve, y_curve, 'linear'
                    
                except Exception as e4:
                    logger.error(f"כל שיטות התאמת העקומה נכשלו: {str(e4)}")
    
    return None, None, None

def create_scatter_plot_by_date(filtered_df):
    """יצירת תרשים פיזור עם קו מגמה לפי תאריך ייצור
    
    Args:
        filtered_df: DataFrame עם נתוני רכב מסוננים
        
    Returns:
        plotly.graph_objects.Figure: התרשים שנוצר
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # חישוב ימים מאז החדש ביותר עבור כל נקודת נתונים
    newest_date = filtered_df['productionDate'].max()
    filtered_df['days_since_newest'] = (newest_date - filtered_df['productionDate']).dt.days
    
    # יצירת תרשים פיזור בסיסי
    fig = px.scatter(
        filtered_df, 
        x='productionDate',  # שימוש בתאריך ייצור בפועל
        y='price',
        color='km_per_year',
        size_max=8,
        color_continuous_scale='viridis',
        range_color=[0, filtered_df['km_per_year'].quantile(0.95)],
        hover_data=['model', 'subModel', 'hand', 'km', 'city', 'productionDateFormatted', 'testDateFormatted', 'gearBox', 'link'],
        labels={'productionDate': 'תאריך ייצור', 
               'price': 'מחיר (₪)', 
               'km_per_year': 'קילומטרים לשנה'},
        title=f'מחירי רכב לפי תאריך ייצור ({len(filtered_df)} רכבים)'
    )
    
    # יצירת נתונים מותאמים אישית למעבר עכבר ולחיצה
    custom_data = np.column_stack((
        filtered_df['model'], 
        filtered_df['subModel'], 
        filtered_df['hand'], 
        filtered_df['km'], 
        filtered_df['city'],
        filtered_df['productionDateFormatted'],
        filtered_df['testDateFormatted'],
        filtered_df['gearBox'],
        filtered_df['createdAtFormatted'],
        filtered_df['shortDescription'],
        filtered_df['link']
    ))
    
    # עדכון מאפייני העקבה
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.8,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        customdata=custom_data,
        hovertemplate='<b>%{customdata[0]} %{customdata[1]}</b><br>' +
                     'מחיר: ₪%{y:,.0f}<br>' +
                     'תאריך ייצור: %{customdata[5]}<br>' +
                     'תאריך טסט: %{customdata[6]}<br>' +
                     'תיבת הילוכים: %{customdata[7]}<br>' +
                     'יד: %{customdata[2]}<br>' +
                     'ק"מ: %{customdata[3]:,.0f}<br>' +
                     'עיר: %{customdata[4]}<br>' +
                     'תאריך מודעה: %{customdata[8]}<br>' +
                     '<b>👆 לחץ להצגת המודעה</b>'
    )
    
    # שיפור פריסה ומראה
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
            dtick="M3",  # מרווחים של 3 חודשים
            tickformat="%Y-%m"  # פורמט שנה-חודש
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
            title="ק\"מ/שנה",
            title_font=dict(size=13),
            tickfont=dict(size=11)
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # הוספת עקומת מגמה אם יש מספיק נקודות נתונים
    if len(filtered_df) > 1:
        # קבלת נתוני x ו-y להתאמת עקומה
        x_data = filtered_df['days_since_newest'].values
        y_data = filtered_df['price'].values
        
        # התאמת העקומה
        x_curve, y_curve, curve_type = fit_trend_curve(x_data, y_data)
        
        if x_curve is not None and y_curve is not None:
            # קבלת שם וסגנון עקומה לפי סוג
            curve_names = {
                'exponential': 'מגמה מעריכית',
                'simple_exponential': 'מגמה מעריכית',
                'log_linear': 'מגמה מעריכית (פשוטה)',
                'linear': 'מגמה לינארית'
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
            
            # הוספת קו המגמה
            curve_name = curve_names.get(curve_type, 'קו מגמה')
            curve_color = curve_colors.get(curve_type, 'red')
            curve_style = curve_styles.get(curve_type, 'solid')
            
            # המרת x_curve מימים לתאריכים בפועל לתרשים
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
    """יצירת תרשים פיזור עם קו מגמה לפי קילומטרים
    
    Args:
        filtered_df: DataFrame עם נתוני רכב מסוננים
        
    Returns:
        plotly.graph_objects.Figure: התרשים שנוצר
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # יצירת תרשים פיזור בסיסי
    fig = px.scatter(
        filtered_df, 
        x='km',
        y='price',
        color='number_of_years',
        size_max=8,
        color_continuous_scale='viridis',
        range_color=[0, filtered_df['number_of_years'].quantile(0.95)],
        hover_data=['model', 'subModel', 'hand', 'productionDateFormatted', 'testDateFormatted', 'gearBox', 'city', 'link'],
        labels={'km': 'קילומטרים', 
               'price': 'מחיר (₪)', 
               'number_of_years': 'גיל הרכב (שנים)'},
        title=f'מחירי רכב לפי קילומטרים ({len(filtered_df)} רכבים)'
    )
    
    # יצירת נתונים מותאמים אישית למעבר עכבר ולחיצה
    custom_data = np.column_stack((
        filtered_df['model'], 
        filtered_df['subModel'], 
        filtered_df['hand'], 
        filtered_df['productionDateFormatted'],
        filtered_df['testDateFormatted'],
        filtered_df['gearBox'],
        filtered_df['city'],
        filtered_df['createdAtFormatted'],
        filtered_df['shortDescription'],
        filtered_df['link']
    ))
    
    # עדכון מאפייני העקבה
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.8,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        customdata=custom_data,
        hovertemplate='<b>%{customdata[0]} %{customdata[1]}</b><br>' +
                     'מחיר: ₪%{y:,.0f}<br>' +
                     'קילומטרים: %{x:,.0f}<br>' +
                     'תאריך ייצור: %{customdata[3]}<br>' +
                     'תאריך טסט: %{customdata[4]}<br>' +
                     'תיבת הילוכים: %{customdata[5]}<br>' +
                     'יד: %{customdata[2]}<br>' +
                     'עיר: %{customdata[6]}<br>' +
                     'תאריך מודעה: %{customdata[7]}<br>' +
                     '<b>👆 לחץ להצגת המודעה</b>'
    )
    
    # שיפור פריסה ומראה
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
            tickformat=",d"  # אלפים מופרדים בפסיקים
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
            title="גיל (שנים)",
            title_font=dict(size=13),
            tickfont=dict(size=11)
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # הוספת עקומת מגמה אם יש מספיק נקודות נתונים
    if len(filtered_df) > 1:
        try:
            # התאמת קו רגרסיה פולינומיאלי
            from scipy.stats import linregress
            from scipy import optimize
            
            # מיון לפי קילומטרים עבור קו המגמה
            sorted_df = filtered_df.sort_values('km')
            x = sorted_df['km'].values
            y = sorted_df['price'].values
            
            # ניסיון להתאים רגרסיה פולינומיאלית
            try:
                # ניסיון להתאים דעיכה מעריכית עם היסט: a * exp(-b * x) + c
                def exp_decay_with_offset(x, a, b, c):
                    return a * np.exp(-b * x) + c
                
                # חישוב פרמטרים
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
                    name='מגמה מעריכית',
                    line=dict(color='red', width=3),
                    hoverinfo='none'
                ))
            except:
                # גיבוי לפולינומיאלי
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                
                x_smooth = np.linspace(min(x), max(x), 200)
                y_smooth = p(x_smooth)
                
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    name='מגמה פולינומיאלית',
                    line=dict(color='red', width=3),
                    hoverinfo='none'
                ))
                
        except Exception as e:
            logger.warning(f"שגיאה בהתאמת עקומת המגמה לתרשים קילומטרים: {str(e)}")
    
    return fig

def create_summary_stats(filtered_df, styles):
    """יצירת רכיב סטטיסטיקות סיכום
    
    Args:
        filtered_df: DataFrame עם נתוני רכב מסוננים
        styles: מילון עם מידע עיצובי
        
    Returns:
        רכיב dash עם סטטיסטיקות סיכום
    """
    style = styles['summary_card']
    
    return html.Div([
        html.Div([
            html.P("מספר רכבים", style=style['label']),
            html.P(f"{len(filtered_df)}", style=style['value'])
        ], style=style['card']),
        
        html.Div([
            html.P("מחיר ממוצע", style=style['label']),
            html.P(f"₪{filtered_df['price'].mean():,.0f}", style=style['value'])
        ], style=style['card']),
        
        html.Div([
            html.P("טווח מחירים", style=style['label']),
            html.P(f"₪{filtered_df['price'].min():,.0f} - ₪{filtered_df['price'].max():,.0f}", 
                  style=style['value'])
        ], style=style['card']),
        
        html.Div([
            html.P("ממוצע ק\"מ לשנה", style=style['label']),
            html.P(f"{filtered_df['km_per_year'].mean():,.0f}", style=style['value'])
        ], style=style['card']),
        
        html.Div([
            html.P("גיל ממוצע של הרכב", style=style['label']),
            html.P(f"{filtered_df['number_of_years'].mean():.1f} שנים", style=style['value'])
        ], style=style['card']),
    ], style=style['container'])


def create_data_table(filtered_df):
    """יצירת טבלת נתונים אינטראקטיבית של רכבים
    
    Args:
        filtered_df: DataFrame עם נתוני רכב מסוננים
        
    Returns:
        רכיב dash_table.DataTable
    """
    # בחירת עמודות להצגה
    display_columns = [
        'model', 'subModel', 'productionDateFormatted', 'price', 
        'km', 'km_per_year', 'hand', 'gearBox', 'city', 'testDateFormatted', 
        'createdAtFormatted', 'listingType', 'shortDescription', 'link'
    ]
    
    # תצורות עמודות עם עיצוב
    columns = [
        {'name': 'דגם', 'id': 'model', 'type': 'text'},
        {'name': 'תת-דגם', 'id': 'subModel', 'type': 'text'},
        {'name': 'תאריך ייצור', 'id': 'productionDateFormatted', 'type': 'text'},
        {'name': 'מחיר (₪)', 'id': 'price', 'type': 'numeric', 
         'format': Format(group=Group.yes, precision=0, scheme=Scheme.fixed)},
        {'name': 'קילומטרים', 'id': 'km', 'type': 'numeric', 
         'format': Format(group=Group.yes, precision=0, scheme=Scheme.fixed)},
        {'name': 'ק"מ/שנה', 'id': 'km_per_year', 'type': 'numeric', 
         'format': Format(group=Group.yes, precision=0, scheme=Scheme.fixed)},
        {'name': 'יד', 'id': 'hand', 'type': 'numeric'},
        {'name': 'תיבת הילוכים', 'id': 'gearBox', 'type': 'text'},
        {'name': 'עיר', 'id': 'city', 'type': 'text'},
        {'name': 'תאריך טסט', 'id': 'testDateFormatted', 'type': 'text'},
        {'name': 'תאריך מודעה', 'id': 'createdAtFormatted', 'type': 'text'},
        {'name': 'סוג מודעה', 'id': 'listingType', 'type': 'text'},
        {'name': 'תיאור', 'id': 'shortDescription', 'type': 'text'},
        {'name': 'קישור', 'id': 'link', 'type': 'text', 'presentation': 'markdown'}
    ]
    
    # הכנת הנתונים עם קישורי markdown
    table_data = filtered_df[display_columns].copy()
    table_data['link'] = table_data['link'].apply(lambda x: f"[צפה במודעה]({x})")
    
    # יצירת רכיב DataTable
    return dash_table.DataTable(
        id='vehicle-table',
        columns=columns,
        data=table_data.to_dict('records'),
        page_size=15,  # מספר שורות בעמוד
        filter_action='native',  # אפשר סינון
        sort_action='native',  # אפשר מיון
        sort_mode='multi',  # אפשר מיון לפי מספר עמודות
        column_selectable='multi',  # אפשר בחירת עמודות
        row_selectable=False,
        selected_columns=[],
        selected_rows=[],
        page_action='native',
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': COLORS['primary'],
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'right'
        },
        style_cell={
            'textAlign': 'right',
            'padding': '12px 15px',
            'fontFamily': 'Roboto, sans-serif',
            'height': 'auto',
            'whiteSpace': 'normal',
            'minWidth': '100px', 
            'width': '150px', 
            'maxWidth': '300px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        export_format='csv',  # אפשר ייצוא ל-CSV
        export_headers='display',
        hidden_columns=[],  # אין עמודות מוסתרות בהתחלה
        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in table_data.to_dict('records')
        ],
        tooltip_duration=None,
        css=[{
            'selector': '.dash-spreadsheet td div',
            'rule': 'max-height: none !important; line-height: 1.4;'
        }]
    )
