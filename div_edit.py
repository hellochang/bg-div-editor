# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:52:24 2021

@author: Chang.Liu
"""
import json
import time
import os
import dash
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# Plotting
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
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

print(data.dtypes)
def highlight_special_case_row():
    return [
        {
            'if': {
                'filter_query': f'{{div_type}} = {case}',
            },
            'backgroundColor': color,
            'color': 'white'
        } for case, color in zip(['special', 'skipped', 'suspension'], ['#fc8399', '#53cfcb', '#fbaea0'])
    ] + \
     [
        {
            'if': {
                'filter_query': '{div_initiation} = 1',
            },
            'backgroundColor': '#fc8399',
            'color': 'white'
        }
    ]
# App Layout
app.layout = dbc.Container([
    html.Br(),
    dbc.Row(html.H1("Dividend Entry Editor")),
    html.Br(),
    dbc.Row(dbc.Label('Select a fsym id')),
    dcc.Dropdown(
                id="fsym-id-dropdown",
                options=[{"label": 'All', "value": 'All'}] + [{"label": i, "value": i} for i in fsym_id],
                value=fsym_id[0],
            ),
    
    # Hidden datatable for storing data
    html.Div([dash_table.DataTable(
        id='data-table',
        columns=[{'name': i, 'id':i} for i in data.columns],
        editable=True,
        data=data.to_dict('records')
    )], style= {'display': 'none'}),
    

    dbc.Row(html.H3("Data")),


    dash_table.DataTable(
        id='output-data-table',
        columns=[{'name': 'fsym_id', 'id': 'fsym_id', 'type': 'text', 'editable': False}] +
      [{'name': i, 'id':i, 'presentation': 'dropdown', 'editable': True} for i in ['listing_currency', 'payment_currency']] + 
        [{'name': i, 'id': i, 'type': 'datetime', 'editable': True} for i in ['declared_date', 'exdate', 'record_date', 'payment_date']] +
        [{'name': 'payment_amount', 'id': 'payment_amount', 'type': 'numeric', 'editable': True}]+
            [{'name': i, 'id':i, 'presentation': 'dropdown', 'editable': True} 
               for i in ['div_type','div_freq', 'div_initiation', 'skipped']] +
            [{'name': i, 'id': i, 'type': 'numeric', 'editable': True} for i in [ 'num_days_exdate',
           'num_days_paydate']],
            
        # data=[],
        # editable=True,
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
        # style_table={'height': '300px', 'overflowY': 'auto'},
        style_data_conditional=highlight_special_case_row(),
        style_cell={
            # 'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0,
        },
        dropdown={
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
                    for i in cur_list
                ]
            },
            'payment_currency': {
                  'options': [
                    {'label': i, 'value': i}
                    for i in cur_list
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
        },
        row_deletable=True,
        
        # tooltip_data=[
        #     {
        #         column: {'value': str(value), 'type': 'markdown'}
        #         for column, value in row.items()
        #     } for row in data.to_dict('records')
        # ],
        # tooltip_duration=None,
        # Workaround for bug regarding display row dropdown with Boostrap
        css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}]
    ),
    html.Div(id='table-dropdown-container'),

    
    html.Br(),
    html.Br(),  
    html.H3("Modified History"),

    dash_table.DataTable(
        id='modified-data-rows',
        columns= [{'name': 'action', 'id':'action'}]+[{'name': i, 'id':i} for i in data.columns],
        data=[],
        # filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_current= 0,
        page_size= 30,
        style_data_conditional=highlight_special_case_row(),
        style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0,
        },
        row_deletable=True,
        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in data.to_dict('records')
        ],
        tooltip_duration=None
    ),
    
    html.Br(),  
    html.H5('Save changes'),

    dbc.Alert(id="save-msg", children="Press this button to save changes", color="info"),
    dbc.Button(id="save-button", n_clicks=0, children='Save', color='success'),
    
])

@app.callback(
    Output('output-data-table', 'data'),
    Input('fsym-id-dropdown', 'value'),
    State('data-table', 'data'))
def filter_fysm_id(selected, datatable):
    if selected == 'All':
        return datatable
    df = pd.DataFrame(datatable)
    return df[df['fsym_id'] == selected].to_dict('records')


@app.callback(
    Output('data-table', 'data'),
    Input('output-data-table', 'data'),
    State('data-table', 'data'),
    State('modified-data-rows', 'data'),
    State('modified-data-rows', 'data_previous'))
def update_data_table(modified_datatable, datatable, rows, rows_prev):
    df = pd.DataFrame(datatable)
    modified_df = pd.DataFrame(modified_datatable)  
    fsym_id = modified_df['fsym_id'].unique()[0]
    df = df.loc[~(df['fsym_id'] == fsym_id)]
    # df[df['fsym_id'] == fsym_id] = modified_df
    res = pd.concat([df, modified_df]).to_dict('records')
    return res + [row for row in rows_prev if row not in rows] if rows is not None else res


# @app.callback(
#     Output('data-table', 'data'),
#     Input('modified-data-rows', 'data'),
#     State('modified-data-rows', 'data_previous'),
#     State('data-table', 'data'))
# def undo_delete_data_table(rows, rows_prev, datatable):
#     return datatable + [row for row in rows_prev if row not in rows]


@app.callback(
    # Output('data-table', 'data'),
    Output('modified-data-rows', 'data'),
    # Input('data-table', 'data_timestamp'),
    Input('output-data-table', 'data_previous'),
    State('output-data-table', 'data'),
    State('modified-data-rows', 'data'))
def update_modified_data_table(rows_prev, rows, modified_rows):
    if (len(rows) == len(rows_prev)):
        modified_rows = [i for i in rows if i not in rows_prev] if modified_rows is None else modified_rows + [i for i in rows if i not in rows_prev]
        modified_rows[-1]['action'] = 'update'
        modified_rows= modified_rows + [i for i in rows_prev if i not in rows]
        modified_rows[-1]['action'] = 'original'
    if (len(rows) < len(rows_prev)):
         modified_rows = modified_rows + [i for i in rows_prev if i not in rows]
         modified_rows[-1]['action'] = 'delete'
    # if (len(rows) == len(rows_prev)):ss
    #     modified_rows = [i.update({'action': 'update'}) for i in rows if i not in rows_prev] if modified_rows is None else modified_rows + [i.update({'action': 'update'}) for i in rows if i not in rows_prev]
    #     modified_rows= modified_rows + [i.update({'action': 'original'}) for i in rows_prev if i not in rows]
    # if (len(rows) < len(rows_prev)):
    #      modified_rows = modified_rows + [i.update({'action': 'delete'}) for i in rows_prev if i not in rows] 
    return modified_rows

@app.callback(
        Output("save-msg", "children"),
        Output("save-msg", "color"),
        Input("save-button", "n_clicks"),
        State("data-table", "data")
        )
def export_data(nclicks, modified_data): 
    if nclicks == 0:
        raise PreventUpdate
    else:
        df = pd.DataFrame(modified_data)
        datatypes = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
        df = df.astype(datatypes)
        print(df.dtypes)
        df.to_csv('edited_div_data')
        return 'Data saved to DB', 'success'

# Running the server
if __name__ == "__main__":
    # View the app at http://192.168.2.77:8080/ or
    #   http://[host computer's IP address]:8080/
    # app.run_server(debug=False, host='0.0.0.0', port = 8080)
    app.run_server(debug=True, port=8030, dev_tools_silence_routes_logging = False)