# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:41:16 2022

@author: Chang.Liu
"""

from pandas import offsets
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import dash_table
from datetime import date, datetime, timedelta


# =============================================================================
# Div Uploader (Step 4)
# =============================================================================

factset_card = [
    dbc.CardHeader('Factset'),
    dbc.CardBody(
        [
            html.H5('Compare with factset', className="card-title"),
            html.Div([dcc.Graph(id='factset-graph'),
            ], id='facset-content'),
            html.Br(),
            dbc.Alert(id="factset-warning-msg", color="info",
                      is_open=False, fade=True),            
        ]
    ),
]

bbg_card = [
    dbc.CardHeader('Bloomberg'),
    dbc.CardBody(
        [
            html.H5('Bloomberg data', className="card-title"),
            html.Div([dcc.Graph(id='bbg-graph'),
            # html.Div([dash_table.DataTable(
            #     id='bbg-data-table',
            # )]),
            ], id='bbg-content'),
            html.Br(),
            dbc.Alert(id="bbg-warning-msg", color="info", 
                      is_open=False, fade=True),            
        ]
    ),
]  

bg_card = [
    dbc.CardHeader('BG'),
    dbc.CardBody(
        [
            html.H5('BG Data', className="card-title"),
            html.Div([
                dcc.Graph(id='bg-db-graph'),
                dash_table.DataTable(
                    id='bg-db-data-table',
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    # page_action='none',
                    fixed_rows={'headers': True},
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    },
                    page_size=20)
                ], id='bg-content'),
            html.Br(),
            dbc.Alert(id="bg-db-warning-msg", color="info",
                      is_open=False, fade=True)
        ]
    ),
]

split_card = [
    dbc.CardHeader('Split History'),
    dbc.CardBody(
        [
            html.H5('Split History', className="card-title"),
            html.Div([
                dash_table.DataTable(id='split-selected-data-table')
                ], id='split-content'),
            html.Br(),
            dbc.Alert(id="split-warning-msg", color="info",
                      is_open=False, fade=True),
        ]
    ),
]

row_2 = dbc.Row(
    [
        dbc.Col(dbc.Card(factset_card, id='factset-card', color="dark", outline=True)),
        dbc.Col(dbc.Card(bbg_card, id='bbg-card', color="dark", outline=True))
    ]
)
 
row_3 = dbc.Row(
    [
        dbc.Col(dbc.Card(split_card, id='split-card', color="dark", outline=True)),
        dbc.Col(dbc.Card(bg_card, id='bg-db-card', color="dark", outline=True)),
    ]
)

graphs_panel = html.Div([row_2, dbc.Row(dbc.Col(html.Br())), row_3])

basic_info_panel = dbc.Card(
    dbc.CardBody([
        html.Br(),
        dbc.Row(dbc.Col(dcc.Markdown(id="basic-info")), justify='center'),
        html.Div(id='payment-exist-msg'), 
        html.Br(),
        html.Div([
            dbc.Row(dbc.Col(dbc.Alert(id="comparison-msg", 
                                      color="info", fade=True)),
                    justify='center'),
            dash_table.DataTable(
                id='comparison-data-table',
        )]),
        html.Br(),
        dbc.Row(dbc.Col(dash_table.DataTable(id='fsym-id-data-table'))),

        html.Br(),
        dcc.Graph(id='fsym-id-graph'),
        ]),
    color="dark", outline=True)

main_panel_fsym_id_selection = html.Div([
    dbc.Row(dbc.Alert(id="add-to-editor-msg", 
                      children='Saved for modifying later',
                      color="success", duration=500, 
                      is_open=False, fade=True)),
    dbc.Row(dbc.Alert(id='warning-end-dropdown', color="warning",
                      duration=700, is_open=False, fade=True)),
    dbc.Row([dbc.Col(dbc.Button(id="prev-button", n_clicks=0, children='Prev'), 
                     width=1),
             dbc.Col(dbc.Button(id="next-button", n_clicks=0, 
                                children='Next'), 
                     width=1),
             dbc.Col(dcc.Dropdown(id="fsym-id-dropdown")),
             dbc.Col(dbc.Button(id="modify-button", n_clicks=0, 
                                children='Modify Later', color='warning'), 
                     width=1), 
             dbc.Col(dbc.Switch(
                    id="modify-switch",
                    label="All modified secid added",
                    value=False), width=1), 
             ], justify='start')
    ])           
    
        
main_panel_upload_save_panel = html.Div([
    html.Br(),
    dbc.Row(dbc.Col(
        dbc.Button(id="upload-button", n_clicks=0, 
                   children='Upload original to DB', color='success'), 
        width=2), justify='end'),
    dbc.Row(dbc.Col(dbc.Alert(id="save-msg", color="info", 
                              is_open=False, duration=600, fade=True)),
            justify='end'),
    ])

# Hidden datatable for storing data
main_panel_hidden_storage = html.Div([
    dcc.Store(id='data-table'),
    dcc.Store(id='new-data-data-table'),  
    dcc.Store(id='skipped-data-table'),
    
    dcc.Store(id='split-db-data-table'),        
    dcc.Store(id='bg-div-data-table'),        
    dcc.Store(id='div-selected-data-table'),
    dcc.Store(id='basic-info-data-table'),
    dcc.Store(id='factset-data-table')
    ])
        
main_panel = html.Div([
    main_panel_hidden_storage,
    main_panel_fsym_id_selection,
    basic_info_panel,
    graphs_panel,
    main_panel_upload_save_panel,   
    html.Br()
    ], id = 'main-panel')

def div_uploader():
   return html.Details([
       html.Summary(html.I('Step 4: View data')),
       html.Br(),
       dbc.Alert('No data to be checked for the current selection.', 
                 id='no-overall-data-msg', color="warning", 
                 is_open=False, fade=True),
       dbc.Alert(id="no-data-msg", color="info", is_open=False, fade=True),
       main_panel,
       ], id='uploader')


# =============================================================================
# Div Editor (Step 5)
# =============================================================================

def highlight_special_row(special_color: str, 
                          skipped_color: str, suspension_color: str):
    """
    Highlight the suspension, skipped, and div initiation
    rows in the given color

    Parameters
    ----------
    special_color : str
        Color code to highlight div initiation rows.
    skipped_colo : str
        Color code to highlight skipped rows.
    suspension_color : str
        Color code to highlight suspension rows.

    Returns
    -------
    List
        List of query for highlighting Dash Datatable.

    """
    return [{
            'if': {
                'filter_query': f'{{div_type}} = {case}',
            },
            'backgroundColor': color,
            'color': 'white'
        } for case, color in zip(['special', 'skipped', 'suspension'], 
                                 [special_color, skipped_color,
                                  suspension_color])
        ] + \
        [{
            'if': {
                'filter_query': '{div_initiation} = 1',
            },
            'color': 'white'
        }]  

output_table_dropdown = {
                'div_type': {
                    'options': [
                        {'label': i, 'value': i}
                        for i in ['regular', 'special', 'suspension']
                    ]
                },
                'div_freq': {
                      'options': [
                        {'label': str(i), 'value': i}
                        for i in [1, 2, 4, 12]
                    ]
                },
                'listing_currency': {
                    'options': [
                        {'label': i, 'value': i}
                        for i in ['USD','CAD','EUR','GBP','JPY']

                    ]
                },
                'payment_currency': {
                      'options': [
                        {'label': i, 'value': i}
                        for i in ['USD','CAD','EUR','GBP','JPY']
                    ]
                },
                'div_initiation': {
                      'options': [
                        {'label': str(i), 'value': i}
                        for i in [0, 1]
                    ]
                },
                'skipped': {
                      'options': [
                        {'label': str(i), 'value': i}
                        for i in [0, 1]
                    ]
                }
            }

output_table_columns = [{'name': 'fsym_id', 'id': 'fsym_id', 
                         'type': 'text', 'editable': False}] +\
    [{'name': i, 'id':i, 'presentation': 'dropdown', 'editable': True}\
     for i in ['listing_currency', 'payment_currency']] +\
    [{'name': i, 'id': i, 'type': 'datetime', 'editable': True} \
     for i in ['declared_date', 'exdate', 'record_date', 'payment_date']] +\
    [{'name': 'payment_amount', 'id': 'payment_amount', 
      'type': 'numeric', 'editable': True}] +\
    [{'name': i, 'id':i, 'presentation': 'dropdown', 'editable': True}
     for i in ['div_type','div_freq', 'div_initiation', 'skipped']] +\
    [{'name': i, 'id': i, 'type': 'numeric', 'editable': True}
     for i in [ 'num_days_exdate', 'num_days_paydate']]
        

output_table = dbc.Row(dbc.Col(dash_table.DataTable(
            id='output-data-table',
            columns=output_table_columns,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            fixed_rows={'headers': True},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
            # page_size=20,
            # style_table={'height': '300px', 'overflowY': 'auto'},
            style_data_conditional=highlight_special_row('#fc8399', '#53cfcb', '#fbaea0'),
            style_cell={
                # 'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': 0,
            },
            dropdown=output_table_dropdown,
            row_deletable=True,
        
            # Workaround for bug regarding display row dropdown with Boostrap
            css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}]
        )), justify='center')

modified_data_history_table = dbc.Row(dbc.Col(dash_table.DataTable(
            id='modified-data-rows',
            data=[],
            sort_action="native",
            sort_mode="multi",
            page_action="native",
            page_current= 0,
            page_size= 10,
            style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': 0,
            },
            row_deletable=True,
            # tooltip_data=[
            #     {
            #         column: {'value': str(value), 'type': 'markdown'}
            #         for column, value in row.items()
            #     } for row in data.to_dict('records')
            # ],
            # tooltip_duration=None
        )), justify='center')

editor_upload_save_panel = html.Div([
    dbc.Row(dbc.Col(
        dbc.Button(id="upload-modified-button", n_clicks=0, 
                   children='Upload modified to DB', color='success'), 
        width=2), justify='end'),
    dbc.Row(html.Br()),
    dbc.Row(dbc.Col(dbc.Alert(id="save-modified-msg", color="info", 
                              is_open=False, duration=1500, fade=True)))
    ])

editor_main_component = html.Div(dbc.Card(
        dbc.CardBody([
            html.Br(),
            # dbc.Row(dbc.Col(dbc.Label('Select a fsym id'), width=10)),
            dbc.Row(html.H5("Edit Entries"), justify='start'),
            dbc.Row(dbc.Col(dcc.Dropdown(
                        id="editor-fsym-id-dropdown",
                    )), justify='center'),
            output_table,
            
            html.Br(),
            html.Br(),  
            dbc.Row(html.H5("Edit History"), justify='start'),
            modified_data_history_table,
            html.Br(),
            editor_upload_save_panel
            
        ]),             
        className="w-85", color="dark", outline=True),
    id="editor-main")

def div_editor():
    return html.Details([
        html.Summary([html.I('Step 5: Edit selected data')]),
        
        # Hidden datatable for storing data
        html.Div([dash_table.DataTable(
            id='edit-data-table',
            editable=True,
        )], style= {'display': 'none'}),
        

        # dbc.Row(dbc.Col(html.H3("Dividend Entry Editor")), justify='start'),
        dbc.Row(html.Br()),
        dbc.Alert("""Select "All modified secid added" from Step 4 to show editor""", 
                  id="editor-open-msg", color="info", is_open=True, fade=True),
        editor_main_component
        ])


# =============================================================================
# Top select panel (Step 1 to Step 3)
# =============================================================================

data_view_type_selection = html.Div(
    id='view-type-div', 
    children=[
        dbc.Row(dbc.Col(dbc.Label('Select the type of data'), width=10)),
        dbc.RadioItems(
            id='view-type-radio',
            options=[
                {'label': 'All Goods', 'value': 'all_goods'},
                {'label': 'Mismatched', 'value': 'mismatch'},
                {'label': 'Skipped', 'value': 'skipped'}
                ], value='mismatch', inline=True)
        ]
    )

date_selection = dcc.DatePickerSingle(
    id='div-date-picker',
    min_date_allowed=date(2010, 8, 30),
    max_date_allowed=(datetime.today() + offsets.MonthEnd(0) - timedelta(days=1)),
    initial_visible_month=datetime.today(),
    date=(date.today() + offsets.MonthEnd(0)).strftime('%Y-%m-%d'),
    # disabled_days=[],
    # display_format='YYYYMMDD',
    clearable =True)

customize_path_widget = html.Details([
    html.Summary('Customize path'),
    html.Div([
        dbc.Label('Path for new dividend data', style={'font-size': '15px'}),
        dbc.Input(placeholder="Path for new dividend data", 
                  type="text", id='div-data-path'),
        html.Br(),
        dbc.Label('Path for seclist'),
        dbc.Input(placeholder="Path for seclist", type="text", id='seclist-path'),
        dbc.Alert(id="path-warning-msg1", color="danger", is_open=False, fade=True),
        dbc.Alert(id="path-warning-msg2", color="danger", is_open=False, fade=True),
    
        html.Br(),
        dbc.Button(id="submit-path-button", n_clicks=0, 
                   children='Submit path', color='success')
        ]),],) 

step_1_date_and_file_selection = html.Details([
    html.Br(),
    html.Summary(html.I('Step 1: Select date and file path (optional)')),
    dbc.Row(dbc.Alert(id='no-file-warning-msg', color="warning",
              duration=4200, is_open=False, fade=True)),
    dbc.Row([dbc.Col(date_selection),
             dbc.Col(customize_path_widget)], justify="evenly")            
    
    ], open=True)

step_2_index_only_selection = html.Details([
    html.Summary(html.I('Step 2: Select index numbers')),
    dbc.Row(dbc.Col(dbc.Label('Getting index members only?'))),
    dbc.Row(dbc.RadioItems(
        id='index-only-radio',
        options=[
            {'label': 'Yes', 'value': True},
            {'label': 'No', 'value': False}
            ],
        value=False,
        labelStyle={'display': 'inline-block'}), justify="center"),
    ], open=True)

step_3_view_type_selection = html.Details([
    html.Summary(html.I('Step 3: Select which type of data to view.')),
    data_view_type_selection,
    ], open=True)


# =============================================================================
# App Layout
# =============================================================================

app_layout = dbc.Container(
    dbc.Card(
        dbc.CardBody([
            # Data storage
            dcc.Store(id='view-type-list'),
            dcc.Store(id='modify-list', data=[]),
        
            html.Br(),
            dbc.Row(dbc.Col(html.H1("Dividend Entry Uploader",
                                    className="card-title"), width=4),
                    justify="center"),
            html.Br(),
            step_1_date_and_file_selection,
            html.Hr(),
            dbc.Row([dbc.Col(step_2_index_only_selection),
                     dbc.Col(step_3_view_type_selection)], justify="evenly"),
            html.Hr(), 
            dcc.Loading(id="is-loading-div-uploader", children=[div_uploader()]),
            html.Hr(), 
            div_editor(),
        ]), className='mt-3')
    , fluid=True)