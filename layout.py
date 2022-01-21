# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:41:16 2022

@author: Chang.Liu
"""

import pandas as pd

from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table
from datetime import date, datetime



# =============================================================================
# Div Uploader
# =============================================================================
factset_card = [
    dbc.CardHeader('Factset'),
    dbc.CardBody(
        [
            html.H5('Compare with factset', className="card-title"),
            dcc.Graph(id='factset-graph'),
        ]
    ),
]

bbg_card = [
    dbc.CardHeader('Bloomberg'),
    dbc.CardBody(
        [
            html.H5('Bloomberg data', className="card-title"),
            dcc.Graph(id='bbg-graph'),
            # html.Div([dash_table.DataTable(
            #     id='bbg-data-table',
            # )]),
    
        ]
    ),
]  

bg_card = [
    dbc.CardHeader('BG'),
    dbc.CardBody(
        [
            html.H5('BG Data', className="card-title"),
            dcc.Graph(id='bg-db-graph'),
            html.Div([dash_table.DataTable(
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
                page_size=20,
            )]),  
        ]
    ),
]

split_card = [
    dbc.CardHeader('Split History'),
    dbc.CardBody(
        [
            html.H5('Split History', className="card-title"),
            html.Div([dash_table.DataTable(id='split-data-table',)])
        ]
    ),
]

row_2 = dbc.Row(
    [
        dbc.Col([dbc.Card(factset_card, id='factset-card', color="dark", outline=True),
                  dbc.Alert(id="factset-warning-msg", color="info", is_open=False)]),
        dbc.Col(dbc.Card(bbg_card, id='bbg-card', color="dark", outline=True))
    ]
)
row_3 = dbc.Row(
    [
        dbc.Col(dbc.Card(split_card, id='split-card', color="dark", outline=True)),
        # dbc.Col([dbc.Card(bg_card, id='bg-db-card', color="dark", outline=True),
    ]
)

graphs_panel = html.Div([row_2, dbc.Row(dbc.Col(html.Br())), row_3])

main_panel_core_functionalities = dbc.Card(
    dbc.CardBody([
        html.Br(),
        dbc.Row(dbc.Col(dcc.Markdown(id="basic-info")), justify='center'),
        # dbc.Row(dbc.Col(dbc.Label('Comparison table')), justify='center'),
        html.Div(id='payment-exist-msg'), 
        html.Br(),
        html.Div([
            dbc.Row(dbc.Col(dbc.Alert(id="comparison-msg", color="info")), justify='center'),
            dash_table.DataTable(
                id='comparison-data-table',
        )]),
        html.Br(),
        dbc.Row(dbc.Col(dash_table.DataTable(id='fsym-id-data-table'))),

        html.Br(),
        dcc.Graph(id='fsym-id-graph'),
        graphs_panel
        ])
    )

main_panel_fsym_id_selection = html.Div([
    dbc.Row(dbc.Alert(id="add-to-editor-msg", 
                      children='Saved for modifying later',
                      color="success", duration=500, is_open=False)),
    dbc.Row(dbc.Alert(id='warning-end-dropdown', color="warning",
                      duration=700, is_open=False)),
    dbc.Row([dbc.Col(dbc.Button(id="prev-button", n_clicks=0, children='Prev'), 
                     width=1),
             dbc.Col(dbc.Button(id="next-button", n_clicks=0, 
                                children='Next'), 
                     width=1),
             dbc.Col(dcc.Dropdown(id="fsym-id-dropdown")),
             dbc.Col(dbc.Button(id="modify-button", n_clicks=0, 
                                children='Modify Later', color='warning'), 
                     width=1), 
             ], justify='start')
    ])           
    
        
main_panel_upload_save_panel = html.Div([
    dbc.Row(dbc.Col(
        dbc.Button(id="upload-button", n_clicks=0, 
                   children='Upload original to DB', color='success'), 
        width=2), justify='end'),
    dbc.Row(dbc.Col(dbc.Alert(id="save-msg", color="info", 
                              is_open=False, duration=600),
                    width=10), justify='end'),
    ])

main_panel_hidden_storage = html.Div([
        html.Div([dash_table.DataTable(
            id='data-table',
            editable=True,
            )], style= {'display': 'none'}),
        
        html.Div([dash_table.DataTable(
            id='new-data-data-table'
            )], style= {'display': 'none'})    
        ]
    )
        
main_panel = html.Div([
    dbc.Alert(id="no-data-msg", color="info", is_open=False),
    dbc.Card(
        id = 'main-panel',
        children = [
            dbc.CardBody([    
                # Hidden datatable for storing data
                main_panel_hidden_storage,
                main_panel_fsym_id_selection,
                main_panel_core_functionalities,
                main_panel_upload_save_panel,   
                html.Br()])
            ]
        )
    ], id='main-panel-div')

def div_uploader():
   return html.Div([
        main_panel,
       ], id='uploader')


# =============================================================================
# Div Editor
# =============================================================================

def highlight_special_row(special_color, skipped_color, suspension_color):
    return [{
            'if': {
                'filter_query': f'{{div_type}} = {case}',
            },
            'backgroundColor': color,
            'color': 'white'
        } for case, color in zip(['special', 'skipped', 'suspension'], 
                                 [special_color, skipped_color, suspension_color])
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

edit_entry_button = html.Div([
    dbc.Button(
        children="Edit entries",
        id="collapse-button",
        className="mb-3",
        color="primary",
        n_clicks=0)
    ], id='collapse-button-div')

editor_upload_save_panel = html.Div([
    dbc.Row(dbc.Col(
        dbc.Button(id="upload-modified-button", n_clicks=0, 
                   children='Upload modified to DB', color='success'), 
        width=2), justify='end'),
    dbc.Row(html.Br()),
    dbc.Row(dbc.Col(dbc.Alert(id="save-modified-msg", color="info", 
                              is_open=False, duration=600),
                    width=10),
            justify='end')
    ])

editor_collapsible_component = dbc.Collapse(dbc.Card(
        dbc.CardBody([
        dbc.Row(dbc.Col(html.H3("Dividend Entry Editor")), justify='start'),
        dbc.Row(html.Br()),

        # Hidden datatable for storing data
        html.Div([dash_table.DataTable(
            id='edit-data-table',
            # columns=[{'name': i, 'id':i} for i in modify_data.columns],
            editable=True,
            # data=modify_data.to_dict('records')
        )], style= {'display': 'none'}),
        
        # dbc.Row(dbc.Col(dbc.Label('Select a fsym id'), width=10)),
        dbc.Row(dbc.Col(dcc.Dropdown(
                    id="editor-fsym-id-dropdown",
                    # options=[{"label": 'All', "value": 'All'}] + 
                    #[{"label": i, "value": i} for i in modify_data['fsym_id']],
                )), justify='center'),
        output_table,
        
        html.Br(),
        html.Br(),  
        dbc.Row(html.H3("Edit History"), justify='start'),
        modified_data_history_table,
        html.Br(),
        editor_upload_save_panel
        ]),             
        className="w-85"),
        id="collapse-editor",
        is_open=False,
    )



def div_editor():
    return html.Div([
        edit_entry_button,
        editor_collapsible_component
        ])


# =============================================================================
# Top select panel
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
    max_date_allowed=(datetime.today() + pd.offsets.MonthEnd(0)),
    date = date(2021, 11, 30),
    # date=(datetime.today() + pd.offsets.MonthEnd(0)),
    # disabled_days=,
    # display_format='YYYYMMDD',
    clearable =True)

index_only_selection = dbc.RadioItems(
    id='index-only-radio',
    options=[
        {'label': 'Yes', 'value': True},
        {'label': 'No', 'value': False}
        ],
    value=False,
    labelStyle={'display': 'inline-block'})

top_select_panel = dbc.Card(
    dbc.CardBody([
        html.Br(),
        dbc.Row(dbc.Col(html.H1("Dividend Entry Uploader", className="card-title"))),
        html.Br(),
        date_selection,

        html.Br(),
        html.Br(),
        dbc.Row(dbc.Col(dbc.Label('Getting index members only?'), width=10)),
        index_only_selection,
        html.Div(id='progress-div', children=[html.Progress(id="progress_bar")]),
        html.Hr(),
        data_view_type_selection,
    ]), className='mt-3')


# =============================================================================
# App Layout
# =============================================================================

app_layout = dbc.Container([
    dcc.Store(id='view-type-list'),
    dcc.Store(id='modify-list', data=[]),

    top_select_panel,
    html.Br(), 

    dcc.Loading(id="is-loading-div-uploader", children=[div_uploader()]),
    html.Br(), 
    div_editor()
    ], fluid=True)