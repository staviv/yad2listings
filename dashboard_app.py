# dashboard_app.py
import os
import sys
import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import logging
import pandas as pd
from pathlib import Path

# יבוא מודולים אחרים של האפליקציה
import yad2_parser
from data_processor import scrape_data, process_data, load_data
from dashboard_components import (
    define_dashboard_styles, 
    get_filter_options, 
    create_scatter_plot_by_date, 
    create_scatter_plot_by_km, 
    create_summary_stats, 
    create_data_table,
    COLORS
)
from yad2_url_parser import parse_yad2_url
from vehicle_list import load_vehicle_list, add_vehicle_to_list

logger = logging.getLogger(__name__)

def create_dashboard(df, output_dir, args, port=8050):
    """יצירה והפעלה של אפליקציית Dash אינטראקטיבית לויזואליזציה של הנתונים"""
    logger.info(f"יוצר לוח מחוונים בפורט {port}")
    
    # קבלת שמות יצרן ודגם
    manufacturer_name = df['make'].iloc[0] if 'make' in df.columns else "רכב"
    model_name = df['model'].iloc[0] if 'model' in df.columns else "ניתוח"
    
    # הגדרת סגנונות ואפשרויות
    styles = define_dashboard_styles()
    filter_options = get_filter_options(df)
    
    # ערכי טווח מחירים
    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    
    # גיליונות סגנון חיצוניים
    external_stylesheets = [
        dbc.themes.BOOTSTRAP,
        {
            'href': 'https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap',
            'rel': 'stylesheet'
        }
    ]
    
    # יצירת האפליקציה
    app = dash.Dash(
        __name__, 
        title=f"{manufacturer_name} {model_name} - מנתח מחירים",
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True
    )
    
    # יצירת הפריסה של האפליקציה
    app.layout = html.Div([
        # כותרת
        html.Div([
            html.H1(f"ניתוח מחירי {manufacturer_name} {model_name}", style={'margin': '0'})
        ], style=styles['header']),
        
        # הוספת רכב
        html.Div([
            html.H4("הוספת רכב באמצעות קישור יד2", style={'marginBottom': '10px'}),
            html.Div([
                dcc.Input(
                    id='yad2-url-input',
                    type='text',
                    placeholder='הדבק קישור יד2 (למשל, https://www.yad2.co.il/vehicles/cars?manufacturer=X&model=Y)',
                    style={'width': '100%', 'padding': '10px', 'marginBottom': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}
                ),
                dcc.RadioItems(
                    id='gear-box-selection',
                    options=[
                        {'label': 'כל סוגי תיבות ההילוכים', 'value': None},
                        {'label': 'אוטומט בלבד (102)', 'value': 102},
                        {'label': 'ידני בלבד (101)', 'value': 101}
                    ],
                    value=None,
                    inline=True,
                    style={'marginBottom': '10px'}
                ),
                html.Button(
                    'הוסף לרשימת הרכבים',
                    id='add-to-list-button',
                    style=styles['success_button'],
                    n_clicks=0
                ),
                html.Div(id='add-to-list-result', style={'marginTop': '10px', 'color': COLORS['primary']}),
                html.Button(
                    'הבא נתוני רכב',
                    id='add-vehicle-button',
                    style=styles['button'],
                    n_clicks=0
                ),
                html.Div(id='add-vehicle-result', style={'marginTop': '10px', 'color': COLORS['primary']})
            ])
        ], style=styles['add_vehicle_container']),
        
        # חלק הפילטרים
        html.Div([
            # פילטר ק"מ לשנה
            html.Div([
                html.Label("סנן לפי ק\"מ/שנה:", style=styles['label']),
                dcc.Dropdown(
                    id='km-filter',
                    options=filter_options['km_ranges'],
                    value='all',
                    clearable=False
                ),
            ], style=styles['filter']),
            
            # פילטר יד בעלים
            html.Div([
                html.Label("סנן לפי יד:", style=styles['label']),
                dcc.Dropdown(
                    id='hand-filter',
                    options=filter_options['hands'],
                    value='all',
                    clearable=False
                ),
            ], style=styles['filter']),
            
            # פילטר תיבת הילוכים
            html.Div([
                html.Label("סנן לפי תיבת הילוכים:", style=styles['label']),
                dcc.Dropdown(
                    id='transmission-filter',
                    options=filter_options['transmissions'],
                    value='all',
                    clearable=False
                ),
            ], style=styles['filter']),
            
            # פילטר מחיר
            html.Div([
                html.Label("סנן לפי מחיר:", style=styles['label']),
                html.Div([
                    dcc.Input(
                        id='min-price-input',
                        type='number',
                        placeholder=f'מינימום: {min_price:,}',
                        min=min_price,
                        max=max_price,
                        step=1000,
                        style={'width': '45%', 'padding': '8px'}
                    ),
                    html.Span(" - ", style={'margin': '0 10px'}),
                    dcc.Input(
                        id='max-price-input',
                        type='number',
                        placeholder=f'מקסימום: {max_price:,}',
                        min=min_price,
                        max=max_price,
                        step=1000,
                        style={'width': '45%', 'padding': '8px'}
                    ),
                ], style=styles['price_range_container']),
            ], style=styles['filter']),
            
            # פילטר דגם
            html.Div([
                html.Label("סנן לפי דגם:", style=styles['label']),
                dcc.Dropdown(
                    id='model-filter',
                    options=filter_options['models'],
                    value=[],
                    multi=True,
                    placeholder="בחר דגם/ים"
                ),
            ], style=styles['filter']),
            
            # פילטר סוג מודעה
            html.Div([
                html.Label("סנן לפי סוג מודעה:", style=styles['label']),
                dcc.Dropdown(
                    id='adtype-filter',
                    options=filter_options['ad_types'],
                    value='all',
                    clearable=False
                ),
            ], style=styles['filter']),

            # חלק פילטר תת-דגם
            html.Div([
                html.Label("סנן לפי תת-דגם:", style=styles['label']),
                html.Div([
                    dcc.Checklist(
                        id='submodel-checklist',
                        options=[],  # יאוכלס דינמית
                        value=[],
                        labelStyle={'display': 'block', 'margin-bottom': '8px', 'cursor': 'pointer'},
                        style={'max-height': '200px', 'overflow-y': 'auto', 'padding': '10px', 
                              'background-color': '#f5f9ff', 'border-radius': '5px'}
                    ),
                ]),
                html.Div([
                    html.Button(
                        'החל פילטרים', 
                        id='apply-submodel-button', 
                        style=styles['button']
                    ),
                    html.Button(
                        'נקה בחירה', 
                        id='clear-submodel-button', 
                        style=styles['clear_button']
                    ),
                ], style={'display': 'flex', 'gap': '10px'}),
            ], style={'width': '23%', 'min-width': '200px', 'padding': '10px', 'flex-grow': '1'}),
            
        ], style=styles['filter_container']),
        
        # הוראת לחיצה
        html.Div([
            html.P("👆 לחץ על כל נקודה בגרף כדי לפתוח את מודעת הרכב בכרטיסייה חדשה")
        ], style=styles['click_instruction']),
        
        # כרטיסיות לויזואליזציות שונות
        dcc.Tabs([
            dcc.Tab(label='מחיר לפי תאריך', children=[
                html.Div([
                    dcc.Graph(id='price-date-scatter')
                ], style=styles['graph']),
            ]),
            dcc.Tab(label='מחיר לפי קילומטרים', children=[
                html.Div([
                    dcc.Graph(id='price-km-scatter')
                ], style=styles['graph']),
            ]),
            dcc.Tab(label='טבלת נתונים', children=[
                html.Div([
                    html.Div([
                        html.Button(
                            'הצג/הסתר עמודות',
                            id='toggle-columns-button',
                            style={**styles['button'], 'width': 'auto', 'margin-right': '10px'}
                        ),
                        html.Div(
                            id='column-toggle-container',
                            style={'display': 'none', 'marginTop': '10px', 'marginBottom': '20px'}
                        ),
                    ], style={'marginBottom': '15px', 'display': 'flex', 'flexDirection': 'column'}),
                    html.Div(id='vehicle-table-container')
                ], style=styles['data_table']['table']),
            ]),
            dcc.Tab(label='פרטי רכב', children=[
                html.Div([
                    html.Div(id='vehicle-details-container', children=[
                        html.P("לחץ על רכב בטבלת הנתונים או בתרשים הפיזור כדי להציג את פרטיו.",
                               style={'textAlign': 'center', 'color': COLORS['secondary'], 'fontStyle': 'italic'})
                    ])
                ], style=styles['data_table']['table']),
            ]),
            dcc.Tab(label='רשימת רכבים', children=[
                html.Div([
                    html.H4("רכבים ברשימת הסריקה", style={'marginBottom': '15px'}),
                    html.Div(id='vehicle-list-container'),
                    html.Button(
                        'רענן רשימת רכבים',
                        id='refresh-vehicle-list-button',
                        style=styles['button'],
                        n_clicks=0
                    )
                ], style=styles['data_table']['table']),
            ]),
        ], style=styles['tabs']),
        
        # חלק סיכום
        html.Div([
            html.H3("סיכום נתונים", style=styles['summary_header']),
            html.Div(id='summary-stats')
        ], style=styles['summary']),
        
        # אחסון לקישורים שנלחצו ומצבים
        dcc.Store(id='clicked-link', storage_type='memory'),
        dcc.Store(id='selected-vehicle', storage_type='memory'),
        dcc.Store(id='user-preferences', storage_type='local'),
        
        # div מוסתר להפעלת סריקה מתוך URL
        html.Div(id='scrape-trigger', style={'display': 'none'}),
    ], style=styles['container'])
    
    # הגדר קולבקים
    setup_callbacks(app, df, output_dir, args, styles)
    
    # הפעל את האפליקציה
    logger.info(f"מפעיל לוח מחוונים בכתובת http://127.0.0.1:{port}/")
    app.run(debug=False, port=port)

def setup_callbacks(app, df, output_dir, args, styles):
    """הגדרת קולבקים ללוח המחוונים
    
    Args:
        app: אפליקציית Dash
        df: DataFrame עם נתוני רכב
        output_dir: ספריית יעד לשמירת נתונים
        args: ארגומנטים משורת הפקודה
        styles: מילון עם מידע עיצובי
    """
    # קולבק לעדכון אפשרויות תת-דגם בהתבסס על הדגמים שנבחרו
    @app.callback(
        Output('submodel-checklist', 'options'),
        Input('model-filter', 'value'),
    )
    def update_submodel_options(selected_models):
        if not selected_models or len(selected_models) == 0:
            # אם לא נבחרו דגמים, הצג את כל תתי-הדגמים עם קידומת דגם
            submodel_options = []
            for sm in sorted(df['subModel'].unique()):
                models_for_submodel = df[df['subModel'] == sm]['model'].unique()
                if len(models_for_submodel) == 1:
                    label = f"[{models_for_submodel[0]}] {sm}"
                else:
                    label = f"[{models_for_submodel[0]}+] {sm}"
                submodel_options.append({'label': label, 'value': sm})
        else:
            # סנן תתי-דגמים על פי הדגמים שנבחרו
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
    
    # קולבק לניקוי בחירת תת-דגם
    @app.callback(
        Output('submodel-checklist', 'value'),
        Input('clear-submodel-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_submodel_selection(n_clicks):
        return []
    
    # קולבק לעדכון גרפים וסיכום בהתבסס על פילטרים
    @app.callback(
        [Output('price-date-scatter', 'figure'),
         Output('price-km-scatter', 'figure'),
         Output('vehicle-table-container', 'children'),
         Output('summary-stats', 'children')],
        [Input('km-filter', 'value'),
         Input('hand-filter', 'value'),
         Input('model-filter', 'value'),
         Input('apply-submodel-button', 'n_clicks'),
         Input('adtype-filter', 'value'),
         Input('transmission-filter', 'value'),
         Input('min-price-input', 'value'),
         Input('max-price-input', 'value')],
        [State('submodel-checklist', 'value'),
         State('user-preferences', 'data')]
    )
    def update_graphs_and_stats(km_range, hand, models, submodel_btn_clicks, adtype, transmission, 
                               min_price, max_price, submodel_list, preferences):
        # החלת פילטרים
        filtered_df = df.copy()
        
        # החלת פילטר ק"מ/שנה
        if km_range != 'all':
            min_km, max_km = map(int, km_range.split('-'))
            filtered_df = filtered_df[filtered_df['km_per_year'] <= max_km]
            if min_km > 0:  # עבור הפילטר "> 25,000"
                filtered_df = filtered_df[filtered_df['km_per_year'] > min_km]
        
        # החלת פילטר יד (בעלים קודמים)
        if hand != 'all':
            min_hand, max_hand = map(int, hand.split('-'))
            filtered_df = filtered_df[filtered_df['hand'] <= max_hand]
        
        # החלת פילטר דגם
        if models and len(models) > 0:
            filtered_df = filtered_df[filtered_df['model'].isin(models)]
            
        # החלת פילטר תת-דגם
        if submodel_list and len(submodel_list) > 0:
            filtered_df = filtered_df[filtered_df['subModel'].isin(submodel_list)]
            
        # החלת פילטר סוג מודעה
        if adtype != 'all':
            filtered_df = filtered_df[filtered_df['listingType'] == adtype]
            
        # החלת פילטר תיבת הילוכים
        if transmission != 'all' and 'gearBox' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['gearBox'] == transmission]
            
        # החלת פילטר מחיר
        if min_price is not None:
            filtered_df = filtered_df[filtered_df['price'] >= min_price]
        if max_price is not None:
            filtered_df = filtered_df[filtered_df['price'] <= max_price]
        
        # יצירת תרשימי פיזור
        fig_date = create_scatter_plot_by_date(filtered_df)
        fig_km = create_scatter_plot_by_km(filtered_df)
        
        # יצירת טבלת נתונים
        data_table = create_data_table(filtered_df)
        
        # החלת העדפות עמודות של המשתמש אם זמינות
        if preferences and 'hidden_columns' in preferences:
            data_table.hidden_columns = preferences['hidden_columns']
        
        # יצירת סטטיסטיקות סיכום
        summary = create_summary_stats(filtered_df, styles)
        
        return fig_date, fig_km, data_table, summary
    
    # קולבק להחלפת נראות בוחר העמודות
    @app.callback(
        [Output('column-toggle-container', 'style'),
         Output('column-toggle-container', 'children')],
        [Input('toggle-columns-button', 'n_clicks')],
        [State('column-toggle-container', 'style'),
         State('vehicle-table', 'columns')]
    )
    def toggle_column_selector(n_clicks, current_style, current_columns):
        if not n_clicks:
            raise PreventUpdate
            
        # החלפת נראות של בוחר העמודות
        visible = current_style.get('display') != 'block'
        new_style = {'display': 'block', 'marginTop': '10px', 'marginBottom': '20px'} if visible else {'display': 'none'}
        
        # יצירת תיבות סימון החלפת עמודות
        column_options = [{'label': col['name'], 'value': col['id']} for col in current_columns]
        column_toggles = dcc.Checklist(
            id='column-toggle-checklist',
            options=column_options,
            value=[col['id'] for col in current_columns if col.get('id') not in []],
            labelStyle={'display': 'inline-block', 'marginRight': '20px', 'cursor': 'pointer'}
        )
        
        return new_style, column_toggles
    
    # קולבק לעדכון עמודות מוסתרות ושמירת העדפות
    @app.callback(
        [Output('vehicle-table', 'hidden_columns'),
         Output('user-preferences', 'data')],
        [Input('column-toggle-checklist', 'value')],
        [State('vehicle-table', 'columns'),
         State('user-preferences', 'data')]
    )
    def update_hidden_columns(selected_columns, all_columns, current_preferences):
        if selected_columns is None:
            raise PreventUpdate
            
        all_column_ids = [col['id'] for col in all_columns]
        hidden_columns = [col_id for col_id in all_column_ids if col_id not in selected_columns]
        
        # עדכון העדפות משתמש
        preferences = current_preferences or {}
        preferences['hidden_columns'] = hidden_columns
        
        return hidden_columns, preferences
    
    # קולבק לטיפול בבחירת רכב לתצוגת פרטים
    @app.callback(
        Output('vehicle-details-container', 'children'),
        [Input('price-date-scatter', 'clickData'),
         Input('price-km-scatter', 'clickData'),
         Input('vehicle-table', 'active_cell')],
        [State('vehicle-table', 'data')]
    )
    def display_vehicle_details(date_click, km_click, active_cell, table_data):
        ctx = callback_context
        
        if not ctx.triggered:
            raise PreventUpdate
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        selected_vehicle = None
        
        if trigger_id == 'price-date-scatter' and date_click:
            # קבלת פרטי רכב מתרשים פיזור
            point_index = date_click['points'][0]['pointIndex']
            customdata = date_click['points'][0]['customdata']
            
            model = customdata[0]
            submodel = customdata[1]
            hand = customdata[2]
            km = customdata[3]
            city = customdata[4]
            production_date = customdata[5]
            test_date = customdata[6]
            gear_box = customdata[7]
            created_at = customdata[8]
            description = customdata[9] if len(customdata) > 9 else ''
            link = customdata[-1]
            price = date_click['points'][0]['y']
            
            selected_vehicle = {
                'model': model,
                'submodel': submodel,
                'hand': hand,
                'km': km,
                'city': city,
                'production_date': production_date,
                'test_date': test_date,
                'gear_box': gear_box,
                'created_at': created_at,
                'description': description,
                'link': link,
                'price': price
            }
            
        elif trigger_id == 'price-km-scatter' and km_click:
            # קבלת פרטי רכב מתרשים פיזור קילומטרים
            point_index = km_click['points'][0]['pointIndex']
            customdata = km_click['points'][0]['customdata']
            
            model = customdata[0]
            submodel = customdata[1]
            hand = customdata[2]
            production_date = customdata[3]
            test_date = customdata[4]
            gear_box = customdata[5]
            city = customdata[6]
            created_at = customdata[7]
            description = customdata[8] if len(customdata) > 8 else ''
            link = customdata[-1]
            price = km_click['points'][0]['y']
            km = km_click['points'][0]['x']
            
            selected_vehicle = {
                'model': model,
                'submodel': submodel,
                'hand': hand,
                'km': km,
                'city': city,
                'production_date': production_date,
                'test_date': test_date,
                'gear_box': gear_box,
                'created_at': created_at,
                'description': description,
                'link': link,
                'price': price
            }
            
        elif trigger_id == 'vehicle-table' and active_cell:
            # קבלת פרטי רכב מטבלה
            row = active_cell['row']
            if row < len(table_data):
                selected_row = table_data[row]
                selected_vehicle = {
                    'model': selected_row.get('model', ''),
                    'submodel': selected_row.get('subModel', ''),
                    'hand': selected_row.get('hand', ''),
                    'km': selected_row.get('km', 0),
                    'city': selected_row.get('city', ''),
                    'production_date': selected_row.get('productionDateFormatted', ''),
                    'test_date': selected_row.get('testDateFormatted', ''),
                    'gear_box': selected_row.get('gearBox', ''),
                    'created_at': selected_row.get('createdAtFormatted', ''),
                    'description': selected_row.get('description', selected_row.get('shortDescription', '')),
                    'link': selected_row.get('link', '').replace('[צפה במודעה](', '').replace(')', ''),
                    'price': selected_row.get('price', 0)
                }
        
        if not selected_vehicle:
            return html.P("לחץ על רכב בטבלת הנתונים או בתרשים הפיזור כדי להציג את פרטיו.",
                         style={'textAlign': 'center', 'color': COLORS['secondary'], 'fontStyle': 'italic'})
        
        # יצירת תצוגת הפרטים
        return html.Div([
            html.H3(f"{selected_vehicle['model']} {selected_vehicle['submodel']}", 
                   style={'borderBottom': f'2px solid {COLORS["secondary"]}', 'paddingBottom': '10px'}),
            
            html.Div([
                html.Div([
                    html.H4("מידע בסיסי", style={'color': COLORS['primary']}),
                    html.Table([
                        html.Tr([
                            html.Td("מחיר:", style={'fontWeight': 'bold', 'padding': '5px 15px 5px 0'}),
                            html.Td(f"₪{selected_vehicle['price']:,.0f}")
                        ]),
                        html.Tr([
                            html.Td("תאריך ייצור:", style={'fontWeight': 'bold', 'padding': '5px 15px 5px 0'}),
                            html.Td(selected_vehicle['production_date'])
                        ]),
                        html.Tr([
                            html.Td("תאריך טסט:", style={'fontWeight': 'bold', 'padding': '5px 15px 5px 0'}),
                            html.Td(selected_vehicle['test_date'])
                        ]),
                        html.Tr([
                            html.Td("תאריך פרסום:", style={'fontWeight': 'bold', 'padding': '5px 15px 5px 0'}),
                            html.Td(selected_vehicle['created_at'])
                        ]),
                        html.Tr([
                            html.Td("יד:", style={'fontWeight': 'bold', 'padding': '5px 15px 5px 0'}),
                            html.Td(selected_vehicle['hand'])
                        ]),
                        html.Tr([
                            html.Td("קילומטרים:", style={'fontWeight': 'bold', 'padding': '5px 15px 5px 0'}),
                            html.Td(f"{selected_vehicle['km']:,.0f}")
                        ]),
                        html.Tr([
                            html.Td("תיבת הילוכים:", style={'fontWeight': 'bold', 'padding': '5px 15px 5px 0'}),
                            html.Td(selected_vehicle['gear_box'])
                        ]),
                        html.Tr([
                            html.Td("עיר:", style={'fontWeight': 'bold', 'padding': '5px 15px 5px 0'}),
                            html.Td(selected_vehicle['city'])
                        ])
                    ])
                ], style={'flex': '1', 'minWidth': '300px'}),
                
                html.Div([
                    html.H4("תיאור", style={'color': COLORS['primary']}),
                    html.Div(
                        selected_vehicle['description'].split('\n'),
                        style={'whiteSpace': 'pre-line', 'backgroundColor': '#f5f9ff', 'padding': '10px', 'borderRadius': '5px'}
                    ),
                    html.Div([
                        html.A(
                            "צפה במודעה ביד2",
                            href=selected_vehicle['link'],
                            target="_blank",
                            style={
                                'display': 'inline-block',
                                'backgroundColor': COLORS['secondary'],
                                'color': 'white',
                                'padding': '10px 15px',
                                'borderRadius': '5px',
                                'textDecoration': 'none',
                                'fontWeight': 'bold',
                                'marginTop': '15px'
                            }
                        )
                    ])
                ], style={'flex': '1', 'minWidth': '300px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '30px'})
        ])

    # קולבק לטיפול בהוספת רכב לרשימה
    @app.callback(
        Output('add-to-list-result', 'children'),
        [Input('add-to-list-button', 'n_clicks')],
        [State('yad2-url-input', 'value'),
         State('gear-box-selection', 'value')]
    )
    def add_vehicle_to_list_callback(n_clicks, url, gear_box_override):
        if not n_clicks or not url:
            raise PreventUpdate
            
        manufacturers, models, gear_box = parse_yad2_url(url)
        
        # אם סופק gear_box_override, הוא גובר על זה מה-URL
        if gear_box_override is not None:
            gear_box = gear_box_override
        
        if not manufacturers and not models:
            return html.Div([
                html.P("לא ניתן לנתח יצרנים ומזהי דגמים מה-URL.", 
                      style={'color': COLORS['danger']})
            ])
        
        try:
            vehicles_added = []
            
            # מקרה 1: יש גם יצרנים וגם מודלים
            if manufacturers and models:
                # אם יש יצרן אחד ומספר מודלים, התאם את היצרן לכל המודלים
                if len(manufacturers) == 1 and len(models) > 1:
                    for model_id in models:
                        add_vehicle_to_list(manufacturers[0], model_id, description=url, gearBox=gear_box)
                        vehicles_added.append((manufacturers[0], model_id))
                # אם יש מודל אחד ומספר יצרנים, התאם את המודל לכל היצרנים
                elif len(models) == 1 and len(manufacturers) > 1:
                    for manufacturer_id in manufacturers:
                        add_vehicle_to_list(manufacturer_id, models[0], description=url, gearBox=gear_box)
                        vehicles_added.append((manufacturer_id, models[0]))
                # אם יש מספר שווה של יצרנים ומודלים, התאם אותם אחד לאחד
                elif len(manufacturers) == len(models):
                    for manufacturer_id, model_id in zip(manufacturers, models):
                        add_vehicle_to_list(manufacturer_id, model_id, description=url, gearBox=gear_box)
                        vehicles_added.append((manufacturer_id, model_id))
                # אחרת, התאם את הזוגות הראשונים עד למינימום האורכים
                else:
                    for i in range(min(len(manufacturers), len(models))):
                        add_vehicle_to_list(manufacturers[i], models[i], description=url, gearBox=gear_box)
                        vehicles_added.append((manufacturers[i], models[i]))
            # מקרה 2: יש רק מודלים
            elif models and not manufacturers:
                for model_id in models:
                    add_vehicle_to_list(None, model_id, description=url, gearBox=gear_box)
                    vehicles_added.append((None, model_id))
            # מקרה 3: יש רק יצרנים
            elif manufacturers and not models:
                for manufacturer_id in manufacturers:
                    add_vehicle_to_list(manufacturer_id, None, description=url, gearBox=gear_box)
                    vehicles_added.append((manufacturer_id, None))
            
            return html.Div([
                html.P(f"נוספו {len(vehicles_added)} רכבים לרשימה בהצלחה:", 
                      style={'color': COLORS['success']}),
                html.Ul([
                    html.Li(f"יצרן {m or 'כל היצרנים'}, דגם {mo or 'כל הדגמים'}") for m, mo in vehicles_added
                ]),
                html.P("הרכבים ייכללו בסריקת הנתונים הבאה.",
                      style={'color': COLORS['primary'], 'marginTop': '5px'})
            ])
            
        except Exception as e:
            return html.Div([
                html.P(f"שגיאה בהוספת רכבים לרשימה: {str(e)}", 
                      style={'color': COLORS['danger']})
            ])

    # קולבק לטיפול בסריקת רכב מתוך URL של יד2
    @app.callback(
        [Output('add-vehicle-result', 'children'),
         Output('scrape-trigger', 'children')],
        [Input('add-vehicle-button', 'n_clicks')],
        [State('yad2-url-input', 'value'),
         State('gear-box-selection', 'value')]
    )
    def add_vehicle_from_url(n_clicks, url, gear_box_override):
        if not n_clicks or not url:
            raise PreventUpdate
            
        manufacturers, models, gear_box = parse_yad2_url(url)
        
        # אם סופק gear_box_override, הוא גובר על זה מה-URL
        if gear_box_override is not None:
            gear_box = gear_box_override
        
        if not manufacturers and not models:
            return html.Div([
                html.P("לא ניתן לנתח יצרנים ומזהי דגמים מה-URL.", 
                      style={'color': COLORS['danger']})
            ]), ""
        
        # הפעל תהליך סריקה עבור כל זוג יצרן-מודל
        try:
            success_messages = []
            
            # מקרה 1: יש גם יצרנים וגם מודלים
            if manufacturers and models:
                # אם יש יצרן אחד ומספר מודלים, התאם את היצרן לכל המודלים
                if len(manufacturers) == 1 and len(models) > 1:
                    for model_id in models:
                        scrape_data(output_dir, manufacturers[0], model_id, args.max_pages, gear_box)
                        success_messages.append(f"יצרן {manufacturers[0]}, דגם {model_id}, תיבת הילוכים {gear_box}")
                # אם יש מודל אחד ומספר יצרנים, התאם את המודל לכל היצרנים
                elif len(models) == 1 and len(manufacturers) > 1:
                    for manufacturer_id in manufacturers:
                        scrape_data(output_dir, manufacturer_id, models[0], args.max_pages, gear_box)
                        success_messages.append(f"יצרן {manufacturer_id}, דגם {models[0]}, תיבת הילוכים {gear_box}")
                # אם יש מספר שווה של יצרנים ומודלים, התאם אותם אחד לאחד
                elif len(manufacturers) == len(models):
                    for manufacturer_id, model_id in zip(manufacturers, models):
                        scrape_data(output_dir, manufacturer_id, model_id, args.max_pages, gear_box)
                        success_messages.append(f"יצרן {manufacturer_id}, דגם {model_id}, תיבת הילוכים {gear_box}")
                # אחרת, התאם את הזוגות הראשונים עד למינימום האורכים
                else:
                    for i in range(min(len(manufacturers), len(models))):
                        scrape_data(output_dir, manufacturers[i], models[i], args.max_pages, gear_box)
                        success_messages.append(f"יצרן {manufacturers[i]}, דגם {models[i]}, תיבת הילוכים {gear_box}")
            # מקרה 2: יש רק מודלים
            elif models and not manufacturers:
                for model_id in models:
                    scrape_data(output_dir, None, model_id, args.max_pages, gear_box)
                    success_messages.append(f"יצרן כלשהו, דגם {model_id}, תיבת הילוכים {gear_box}")
            # מקרה 3: יש רק יצרנים
            elif manufacturers and not models:
                for manufacturer_id in manufacturers:
                    scrape_data(output_dir, manufacturer_id, None, args.max_pages, gear_box)
                    success_messages.append(f"יצרן {manufacturer_id}, דגם כלשהו, תיבת הילוכים {gear_box}")
            
            # עיבוד כל הנתונים שנסרקו
            process_data(output_dir)
            
            return html.Div([
                html.P(f"נסרקו נתונים בהצלחה עבור:", 
                      style={'color': COLORS['success']}),
                html.Ul([
                    html.Li(msg) for msg in success_messages
                ]),
                html.P("אנא רענן את הדף כדי לראות את הנתונים המעודכנים.",
                      style={'color': COLORS['primary'], 'marginTop': '5px'})
            ]), f"scrape-completed-{len(success_messages)}"
            
        except Exception as e:
            return html.Div([
                html.P(f"שגיאה בסריקת נתונים: {str(e)}", 
                      style={'color': COLORS['danger']})
            ]), ""
    
    # קולבק להצגת רשימת רכבים
    @app.callback(
        Output('vehicle-list-container', 'children'),
        [Input('refresh-vehicle-list-button', 'n_clicks')]
    )
    def display_vehicle_list(n_clicks):
        from vehicle_list import load_vehicle_list
        
        # טעינת רשימת הרכבים
        vehicles = load_vehicle_list()
        
        if not vehicles:
            return html.P("אין רכבים ברשימה. הוסף רכבים באמצעות הטופס למעלה.",
                         style={'textAlign': 'center', 'color': COLORS['secondary'], 'fontStyle': 'italic'})
        
        # יצירת כותרת טבלה
        header = html.Tr([
            html.Th("מזהה יצרן", style={'padding': '10px', 'textAlign': 'right', 'backgroundColor': COLORS['primary'], 'color': 'white'}),
            html.Th("מזהה דגם", style={'padding': '10px', 'textAlign': 'right', 'backgroundColor': COLORS['primary'], 'color': 'white'}),
            html.Th("תיבת הילוכים", style={'padding': '10px', 'textAlign': 'right', 'backgroundColor': COLORS['primary'], 'color': 'white'}),
            html.Th("תיאור", style={'padding': '10px', 'textAlign': 'right', 'backgroundColor': COLORS['primary'], 'color': 'white'})
        ])
        
        # יצירת שורות טבלה
        rows = []
        for vehicle in vehicles:
            mfr_text = str(vehicle.get('manufacturer', 'הכל')) if vehicle.get('manufacturer') is not None else 'הכל'
            model_text = str(vehicle.get('model', 'הכל')) if vehicle.get('model') is not None else 'הכל'
            
            gear_box_value = vehicle.get('gearBox', 'הכל')
            gear_box_text = "אוטומט (102)" if gear_box_value == 102 else "ידני (101)" if gear_box_value == 101 else "הכל"
            
            row = html.Tr([
                html.Td(mfr_text, style={'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                html.Td(model_text, style={'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                html.Td(gear_box_text, style={'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                html.Td(vehicle.get('description', ''), style={'padding': '8px', 'borderBottom': '1px solid #ddd'})
            ])
            rows.append(row)
        
        # יצירת הטבלה
        vehicle_table = html.Table(
            [header] + rows,
            style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '20px'}
        )
        
        return html.Div([
            html.P(f"סה\"כ רכבים ברשימה: {len(vehicles)}", 
                  style={'marginBottom': '15px', 'fontWeight': 'bold', 'color': COLORS['primary']}),
            vehicle_table
        ])
    
    # קולבק צד לקוח לפתיחת קישורים בכרטיסייה חדשה
    app.clientside_callback(
        """
        function(clickData) {
            if(clickData && clickData.points && clickData.points.length > 0) {
                const link = clickData.points[0].customdata[10] || clickData.points[0].customdata[9];
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
    
    # קולבק צד לקוח לפתיחת קישורים בכרטיסייה חדשה עבור תרשים פיזור ק"מ
    app.clientside_callback(
        """
        function(clickData) {
            if(clickData && clickData.points && clickData.points.length > 0) {
                const link = clickData.points[0].customdata[9];
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
