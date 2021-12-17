# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:52:24 2021

@author: Chang.Liu
"""

import os
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Plotting
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, date 

import sys
sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')
from bg_data_importer import DataImporter

data_importer = DataImporter(False)
def highlight_sepcials(s, columns):
    sepcials = pd.Series(data=False, index=s.index)
    sepcials[columns] = s[columns] == 1
    return ['background-color: yellow' if sepcials.any() else None for v in sepcials]

def highlight_initiation(s, columns):
    sepcials = pd.Series(data=False, index=s.index)
    sepcials[columns] = s[columns] > 450
    return ['background-color: green' if sepcials.any() else None for v in sepcials]

def massage_div_df(df_):
    df = df_.copy()
    df['num_days_exdate'] = df.groupby('fsym_id')['exdate'].apply(lambda x:x.shift(1) - x).dt.days.fillna(0)
    df['num_days_paydate'] = df.groupby('fsym_id')['payment_date'].apply(lambda x:x.shift(1)-x).dt.days.fillna(0)
    int_cols = ['div_initiation', 'skipped', 'num_days_exdate', 'num_days_paydate']
    df[int_cols] = df[int_cols].astype(int)
    df[df.select_dtypes(include='datetime').columns] = \
        df[df.select_dtypes(include='datetime').columns].apply(lambda x: x.dt.strftime('%Y-%m-%d'))
    return df

def get_raw_data(fsym_id):
    seclist = str(tuple(fsym_id)) if len(fsym_id) > 1 else f"'{fsym_id[0]}'"
    query = f"""select * from fstest.dbo.bg_div 
                where fsym_id in {seclist}
                order by fsym_id asc, exdate desc
             """
    df = data_importer.load_data(query)
    df = massage_div_df(df)
    return df


seclist = ['CRHZ50-R','D4VRCD-R','DYZT1C-R','F86WNV-R','F9CCB4-R','FFKCF9-R','FWD88N-R',
'FZX1RR-R','G96265-R','GW7J66-R','H59VN3-R','H7HFJJ-R','J0FV3X-R','JDDDQJ-R',
'JJ8MHV-R','KFMHQN-R','KMQ9YZ-R','M75128-R','MH3J5L-R','NL5NGR-R','PHPQLF-R',
'QBSNDL-R','S7YFVK-R','SNVTLF-R','V05ZYN-R','V3MNCR-R','W5C59W-R','WMSKQW-R',
'WRTQ85-R','WYM8VT-R','WZNMF1-R','X0NPP6-R','X2DZC9-R','X44KDF-R']
data = get_raw_data(seclist)
cur_list = ['USD','CAD','EUR','GBP','JPY']


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", 
                "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

fsym_id = data['fsym_id'].unique()
def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return dbc.Row(
        id="control-card",
        children=[
            dbc.Col(html.P("Select fsym_id"), width=2),
            dbc.Col(dcc.Dropdown(
                id="fsym-id-dropdown",
                options=[{"label": i, "value": i} for i in fsym_id],
                value=fsym_id[0],
            ), width=4),
            html.Br(),
            
            dbc.Col(html.P("Select row"), width=2),
            dbc.Col(dcc.Dropdown(id="idx-dropdown", 
                                 options=[{'label': 0, 'value': 0}, 
                                          {'label': 1, 'value': 1}], value=0)
                    , width=4),
            html.Br()
        ],
    )
def make_currency_radio(type_lst):
    res = []
    for option in type_lst:
        res = res + [
            dbc.Col(html.P(f'{option} currency'), width=2),
            dbc.Col(dbc.RadioItems(
                id=f'{option}-currency-radio',
                options=[{'label': cur, 'value': cur } for cur in cur_list],
                value='USD'), width=4)
            ]
    return res
    
def make_currency_radio_selector():
    return dbc.Row(
        children=make_currency_radio(['listing', 'payment']),
        style = dict(horizontalAlign = 'center'))

def make_div_ini_skip(type_lst):
    res = []
    for option in type_lst:
        res = res + [
            dbc.Col(dbc.Label(f'Div {option}'), width=1),
            dbc.Col(dbc.RadioItems(
                    id=f'div-{option}-radio',
                    options=[{'label': flag, 'value': flag } for flag in [0, 1]],
                    value='USD'), width=2)
            ]
    return res
    
def make_div_input():
    return dbc.Row(
        children=[
            dbc.Col(html.P('Div type')),
            dbc.Col(dbc.RadioItems(
                id='div-type-radio',
                options=[
                    {'label': 'regular', 'value': 'regular'},
                    {'label': 'special', 'value': 'special'},
                    {'label': 'suspension', 'value': 'suspension'}
                ],
                value='regular'
            )),
            dbc.Col(html.P('Div freq')),
            dbc.Col(dbc.RadioItems(
                id='div-freq-radio',
                options=[{'label': frq, 'value': frq } for frq in [1, 2, 4, 12]],
                value=2
            ))] + make_div_ini_skip(['initiation', 'skipped']),
        style = dict(horizontalAlign = 'center'))

def make_id_input():
    return dbc.Row(
        id="id_input_container",
        children=[
            dbc.Col(dbc.Input(id='fsym_id_input', type='text', 
                      placeholder='Enter fsym_id')),
            dbc.Col(dbc.Input(id='bbg_id_input', type='text', 
                      placeholder='Enter bbg id')),
            dbc.Col(dbc.Input(
                        id="payment-amount-input",
                        type='number',
                        placeholder='Enter payment amount')),])

    # dcc.DatePickerSingle(
    #     id='my-date-picker-single',
    #     min_date_allowed=date(1995, 8, 5),
    #     max_date_allowed=date(2017, 9, 19),
    #     initial_visible_month=date(2017, 8, 5),
    #     date=date(2017, 8, 25)
    # ),
def get_date_picker(type_lst):
    res = []
    for type in type_lst:
        res = res + [
            dbc.Col(html.P(f'{type} date')),
            dbc.Col(dcc.DatePickerSingle(
                id=f'{type}-date-picker',
                min_date_allowed=date(1995, 8, 5),
                max_date_allowed=date(2017, 9, 19),
                initial_visible_month=date(2017, 8, 5),
                date=date(2017, 8, 25)
                ))
            ]
    return res
    
def make_date_picker():
    return dbc.Row(get_date_picker(['declared', "exdate", 'record', 'payment']))


def make_buttons():
    return dbc.Row([
                dbc.Col(dbc.Button(id="update-btn", children='Update Change', color='success',
                                    style = dict(horizontalAlign = 'center'))),
                dbc.Col(dbc.Button(id="delete-btn", children='Delete Payment', color='danger',
                                    style = dict(horizontalAlign = 'center'))),
                dbc.Col(dbc.Button(id="revert-btn", children='Revert Change', n_clicks=0, color='warning',
                                    style = dict(horizontalAlign = 'center'))),
                dbc.Col(dbc.Button(id="modified-btn", children='View Modified Payment', color='info',
                                    style = dict(horizontalAlign = 'center')))], justify='evenly')


# App Layout
app.layout = dbc.Container([
    html.H1("Dividend Entry Editor"),
    html.Br(),
    
    html.Div(
            id="top-column",
            children=[generate_control_card()]
    ),
    
    html.Hr(),
    html.Br(),
    
    html.Div(
            id="id-input-panel",
            children=[make_id_input()]
    ),
    
    html.Br(),    

    html.Div(
            id="date-input-panel",
            children=[make_date_picker()]
    ),
    
    html.Br(),
    
    html.Div(
            id="currency-select-panel",
            children=[make_currency_radio_selector()]
    ),

    
    html.Br(),
    
    html.Div(
            id="div-input-panel",
            children=[make_div_input()]
    ),
    
    html.Br(),

    # html.Div(
    #         id="top-column",
    #         children=[generate_control_card()]
    # )
    
    # html.Br(),
    
    html.Br(),    
    html.Div(
        id='button-panel',
        children=[make_buttons()]),
    
    html.Hr(),
    html.Div(id='display-selected-values'),
    
    dash_table.DataTable(
        id='updated-row',
        columns=[{'name': i, 'id':i} for i in data.columns]),
    
    html.Br(),
    
    # dash_table.DataTable(
    #     id='selected-row',
    #     columns=[{'name': i, 'id':i} for i in data.columns]),
    
    # dash_table.DataTable(
    #     id='selected-fsym-id-table',
    #     columns=[{'name': i, 'id':i} for i in data.columns],
    #     data=data.to_dict('records'))
    
    dash_table.DataTable(
        id='selected-fsym-id-table',
        columns=[{'name': i, 'id':i} for i in data.columns],
        data=data.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_current= 0,
        page_size= 20
    )


])

@app.callback(
    Output('idx-dropdown', 'options'),
    Input('fsym-id-dropdown', 'value'))
def set_idx_options(selected_fsym_id):
    idx = data[data['fsym_id' == selected_fsym_id]].reset_index(drop=True)
    return [{'label': i, 'value': i} for i in idx.index]


@app.callback(
    Output('idx-dropdown', 'options'),
    Input('fsym-id-dropdown', 'value'))
def update_new_row(selected_fsym_id):
    idx = data[data['fsym_id' == selected_fsym_id]].reset_index(drop=True)
    return [{'label': i, 'value': i} for i in idx.index]


# @app.callback(
#     Output(),
#     Input('fsym-id-dropdown', 'value'),
#     Input('idx-dropdown', 'value'))
# def set_idx_options(selected_fsym_id, selected_idx):
#     if len(selected_fsym_id) != 0:
#         idx = data[data['fsym_id' == selected_fsym_id]].reset_index(drop=True)
#         return [{'label': i, 'value': i} for i in idx.index]
#     else:
#         return dash.no_update

@app.callback(
    Output('display-selected-values', 'children'),
    Input('fsym-id-dropdown', 'value'),
    Input('idx-dropdown', 'value'))
def set_display_children(selected_fsym_id, selected_idx):
    return f'{selected_idx} is an index in {selected_fsym_id}'

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8030)