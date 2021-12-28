# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:52:24 2021

@author: Chang.Liu
"""

import json
import time
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

# import os
# os.chdir(r'C:\Users\Chang.Liu\Documents\dev\div_data_uploader')
from bg_data_uploader import *
import sys
sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')
from bg_data_importer import DataImporter

data_importer = DataImporter(False)

# =============================================================================
# Div Modifier Helpers
# =============================================================================
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


# =============================================================================
# Div Uploader Helpers
# =============================================================================
def plot_dividend_data(fsym_id, alt_cur=None):
    fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'},
                                     yaxis=dict(title='Amount'),
                                     yaxis2=dict(title='Freq', overlaying='y', side='right', range=[0, 14])))
    new = new_data[new_data['fsym_id'] == fsym_id].copy()
    new['div_type'] = new['div_type'].str.replace(' ', '')
    new_regular = new[new['div_type'] != 'special']
    new_special = new[new['div_type'] == 'special']
    
    bg = get_bg_div_data(fsym_id)
    bg['div_type'] = bg['div_type'].str.replace(' ', '')
    bg_regular = bg[bg['div_type'] != 'special']
    bg_special = bg[bg['div_type'] == 'special']
    
    already_exist = list(set(bg['exdate']).intersection(set(new['exdate'])))
    if len(already_exist) > 0:
        with outs2:
            clear_output()
            display(f'Payments on {str(already_exist)} already exists.')
        return
    fig.add_trace(go.Scatter(x=new_regular['exdate'], y=new_regular['payment_amount'], mode='lines+markers', name='New Regular', line=dict(color='orchid')))
    fig.add_trace(go.Scatter(x=new_special['exdate'], y=new_special['payment_amount'], mode='markers',name='New Special', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=new_special['exdate'], y=new_special['div_freq'], mode='markers', name='Div Freq', line=dict(color='green'), yaxis='y2'))
    fig.add_trace(go.Scatter(x=bg_regular['exdate'], y=bg_regular['payment_amount'], mode='lines+markers',name='BG Regular', line=dict(color='grey')))
    fig.add_trace(go.Scatter(x=bg_regular['exdate'], y=bg_regular['div_freq'], mode='markers', name='Div Freq', line=dict(color='green'), yaxis='y2'))
    if sum(bg_regular['div_initiation']) > 0:
        fig.add_trace(go.Scatter(x=bg_regular['exdate'], y=bg_regular['div_initiation'].astype(int).replace(0, np.nan), mode='markers',name='BG Init', line=dict(color='dodgerblue')))
    fig.add_trace(go.Scatter(x=bg_special['exdate'], y=bg_special['payment_amount'], mode='markers',name='BG Special', line=dict(color='black')))
    
    if alt_cur is not None:
        fig.add_trace(go.Scatter(x=alt_cur['exdate'], y=alt_cur['payment_amount'], mode='lines+markers',name='Alt Cur', line=dict(color='purple')))
    
    if new.shape[0]>1:
        fig.update_layout(shapes=[dict(type="rect",xref="x",yref="paper",x0=new['exdate'].min(),y0=0,x1=new['exdate'].max(),y1=1,fillcolor="LightSalmon",opacity=0.5,layer="below",line_width=0)])
    
    fig.show()
def compare_new_data_with_factset(secid, factset=None):
    if factset is None:
        factset = factset_new_data(secid)
    if check_exist:
        factset = factset[factset['exdate'] <= update_date]
    bbg = new_data[new_data['fsym_id']==secid].copy()
    new_data_comparison = pd.merge(bbg, factset, how='outer', on=['fsym_id','exdate','div_type'], suffixes=('_bbg','_factset'))
    new_data_comparison = new_data_comparison.sort_values(['exdate'])
    new_data_comparison = new_data_comparison.reset_index(drop=True)
    new_data_comparison = new_data_comparison[new_data_comparison.filter(regex='fsym_id|exdate|payment_date|amount').columns]
    new_data_comparison['check_amount'] = np.where(abs(new_data_comparison['payment_amount_factset']-new_data_comparison['payment_amount_bbg'])>0.001, 'Mismatch', 'Good')
    new_data_comparison['check_payment_date'] = np.where(new_data_comparison['payment_date_factset']!=new_data_comparison['payment_date_bbg'], 'Mismatch', 'Good')
    return new_data_comparison

def plot_data(x):
    global check_exist
    check_exist = check_existence(select.value)
    with outs2:
            clear_output()
            display(basic_info(select.value, new_data))
    if not check_exist:
        with outs:
            clear_output()
            print("This is a new name.")
            bbg = new_data[new_data['fsym_id']==select.value].copy()
            factset = factset_new_data(select.value)
            comparison = compare_new_data_with_factset(select.value, factset)
            plot_dividend_data_comparison(factset, bbg)
            display(HTML(comparison.to_html()))
    else:
        (last_cur, fstest_cur) = dividend_currency(select.value, new_data)
        with outs2:
            df = compare_new_data_with_factset(select.value)
            if fstest_cur != last_cur:
                print(f'Possible currency change. Last payment:{last_cur}. Factset payment:{fstest_cur}')
            display(HTML(df.to_html()))
        with outs:
            clear_output()
            df = compare_new_data_with_factset(select.value)
            new = new_data[new_data['fsym_id']==select.value].copy()
            new['listing_currency'] = fstest_cur
            new['payment_currency'] = last_cur
            display('New Dividend Data')
            display(HTML(new.to_html()))
            plot_dividend_data(select.value)
            
def print_new_data_comparison(x):
    df = compare_new_data_with_factset(select.value)
    with outs:
        clear_output()
        display(HTML(df.to_html()))

def prepare_bbg_data(fsym_id, alt_cur='', regular_skipped='regular'):
    if regular_skipped == 'regular':
        new = new_data[new_data['fsym_id']==fsym_id].copy()
    elif regular_skipped == 'skipped' :
        new = skipped[skipped['fsym_id']==fsym_id].copy()
        
    (last_cur, fstest_cur) = dividend_currency(fsym_id, new_data)
    if last_cur is None:
        last_cur = fstest_cur
#     new['listing_currency'] = fstest_cur
    if alt_cur != '':
        new['payment_currency'] = alt_cur.upper()
    else:
        new['payment_currency'] = last_cur
    return new

def upload_new_data_to_database(x):
    new = prepare_bbg_data(select.value, textbox.value)
    upload_to_db(new)
    with outs2:
        clear_output()
    with outs:
        clear_output()
        display('Uploaded')
        textbox.value=''
#         plot_dividend_data(select.value)

def upload_to_database_CAD(x):
    new = prepare_bbg_data(select.value,'CAD')
    upload_to_db(new)
    with outs2:
        clear_output()
    with outs:
        clear_output()
        display('Uploaded')
        textbox.value=''
        
def upload_to_database_USD(x):
    new = prepare_bbg_data(select.value,'USD')
    upload_to_db(new)
    with outs2:
        clear_output()
    with outs:
        clear_output()
        display('Uploaded')
        textbox.value=''
        
def upload_skipped_to_database(x):
    new = prepare_bbg_data(select.value, textbox.value, 'skipped')
    upload_to_db(new)
    with outs2:
        clear_output()
    with outs:
        clear_output()
        display('Uploaded')
        textbox.value=''

def check_split_history(x):
    with outs2:
        query = f"""select 
                    fsym_id,p_split_date,p_split_factor, 
                    exp(sum(log(p_split_factor))  OVER (ORDER BY p_split_date desc)) cum_split_factor 
                    from fstest.fp_v2.fp_basic_splits where fsym_id= '{select.value}'
                    order by p_split_date
                """
        df = data.load_data(query)
        display(HTML(df.to_html()))
        
def plot_db(x):
    query = f"select * from fstest.dbo.bg_div where fsym_id ='{select.value}'"
    df = data.load_data(query)
    with outs:
        clear_output()
        plot_generic_dividend_data(df)
        display(HTML(df.to_html()))

def plot_bbg(x):
    df = prepare_bbg_data(select.value, textbox.value)
    with outs2:
        clear_output()
        plot_generic_dividend_data(df)
        display(HTML(df.to_html()))
        
def plot_generic_dividend_data(df):
    fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'}))
    df['div_type'] = df['div_type'].str.replace(' ', '')
    df_regular = df[df['div_type'] == 'regular']
    df_special = df[df['div_type'] == 'special']
    fig.add_trace(go.Scatter(x=df_regular['exdate'], y=df_regular['payment_amount'], mode='lines+markers', name='Regular', line=dict(color='orchid')))
    fig.add_trace(go.Scatter(x=df_special['exdate'], y=df_special['payment_amount'], mode='markers',name='Special', line=dict(color='black')))
    if df_regular['div_initiation'].sum() > 0:
        fig.add_trace(go.Scatter(x=df_regular['exdate'], y=df_regular['div_initiation'].astype(int).replace(0, np.nan), mode='markers',name='BG Init', line=dict(color='dodgerblue')))
    if 'suspension' in df['div_type'].values:
        df_suspension = df[df['div_type'] == 'suspension']
        fig.add_trace(go.Scatter(x=df_suspension['exdate'], y=[df_regular['payment_amount'].max()]*df_suspension.shape[0], mode='markers',name='BG Suspension', line=dict(color='Red')))
    fig.show()
        
def plot_comparison(x):
    bbg = new_data[new_data['fsym_id']==select.value].copy()
    query = f"select * from fstest.dbo.bg_div where fsym_id ='{select.value}'"
    bg = data.load_data(query)
    with outs:
        clear_output()
        plot_dividend_data_comparison(bg, bbg)
        
def plot_dividend_data_comparison(bg, bbg):
    fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'}))
    bg['div_type'] = bg['div_type'].str.replace(' ', '')
    bg_regular = bg[bg['div_type'] == 'regular']
    bg_special = bg[bg['div_type'] == 'special']
    
    bbg['div_type'] = bbg['div_type'].str.replace(' ', '')
    bbg_regular = bbg[bbg['div_type'] == 'regular']
    bbg_special = bbg[bbg['div_type'] == 'special']
    
    fig.add_trace(go.Scatter(x=bg_regular['exdate'], y=bg_regular['payment_amount'], mode='lines+markers', name='BG Regular', line=dict(color='orchid')))
    fig.add_trace(go.Scatter(x=bbg_regular['exdate'], y=bbg_regular['payment_amount'], mode='lines+markers', name='BBG Regular', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=bg_special['exdate'], y=bg_special['payment_amount'], mode='markers',name='BG Special', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=bbg_special['exdate'], y=bbg_special['payment_amount'], mode='markers',name='BBG Special', line=dict(color='black')))
    fig.show()
    
def show_all_goods(x):
    with outs:
        clear_output()
        display(HTML(all_goods.to_html()))
    select.options = sorted(all_goods['fsym_id'].to_list())
        
def show_mismatch(x):
    with outs:
        clear_output()
    with outs2:
        clear_output()
    select.options = sorted(manual_list)

def show_skipped(x):
    with outs:
        clear_output()
        select.options = list(skipped['fsym_id'].unique())
        display(HTML(skipped.to_html()))
    with outs2:
        clear_output()

# =============================================================================
# Dash app
# =============================================================================

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", 
                "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

fsym_id = data['fsym_id'].unique()

print(data.dtypes)
def highlight_special_case_row():
    return [{
            'if': {
                'filter_query': f'{{div_type}} = {case}',
            },
            'backgroundColor': color,
            'color': 'white'
        } for case, color in zip(['special', 'skipped', 'suspension'], ['#fc8399', '#53cfcb', '#fbaea0'])
        ] + \
        [{
            'if': {
                'filter_query': '{div_initiation} = 1',
            },
            'backgroundColor': '#fc8399',
            'color': 'white'
        }]
seclist = []#TODO combine with the other Card component later
manual_list = []
skipped = pd.DataFrame([])
@app.callback(
    Output('new_data-data-table', 'data'),
    # Output('load-data-msg', 'value'),
    Input('date-picker', 'date'),
    State('index-only-radio', 'value'))
def load_data(update_date, index_flag):
    # monthend_date = (datetime.today() + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
    # update_date = input(f"Enter update date in yyyy-mm-dd format (Default: '{monthend_date}'): ")
    # if update_date == "":
    #     update_date = monthend_date
    print(f'update_date:{update_date}')
    
    main_dir = Path('I:\Scheduled_Jobs')
    f_date = update_date.replace('-','')#TODO!!!!!!!!!
    new_data = pd.read_parquet(main_dir /'output'/ Path(f"new_dvd_data_{f_date}.parquet"))
    update_list = pd.read_csv(main_dir / Path(f"input\sec_list_{f_date}.csv"))
    # splits = pd.read_csv(main_dir / Path(f"input\splits_{f_date}.csv")) 
    #TODO don't think splits is used anywhere. double check later
    
    (new_data, skipped, pro_rata, splits) = \
        preprocess_bbg_data(new_data, update_list, index_flag, update_date)
    new_data = bbg_data_single_security(new_data)
    skipped = process_skipped(skipped)
    seclist = sorted(list(new_data['fsym_id'].unique()))
    (manual_list, all_goods) = bulk_upload(new_data, update_date, new_data)
    print('Loaded')#TODO
    return new_data.to_dict('records')
        
def div_uploader():
    return dbc.Card(
        dbc.CardBody([
            # Hidden datatable for storing data
            html.Div([dash_table.DataTable(
                id='new_data-data-table',
                columns=[{'name': i, 'id':i} for i in data.columns],
                data=data.to_dict('records')
            )], style= {'display': 'none'}),
        
            # # Hidden datatable for storing data
            # html.Div([dash_table.DataTable(
            #     id='skipped-data-table',
            #     columns=[{'name': i, 'id':i} for i in data.columns],
            #     data=data.to_dict('records')
            # )], style= {'display': 'none'}),
            
            # # Hidden datatable for storing data
            # html.Div([dash_table.DataTable(
            #     id='all_good-data-table',
            #     columns=[{'name': i, 'id':i} for i in data.columns],
            #     data=data.to_dict('records')
            # )], style= {'display': 'none'}),
            
            dbc.Row(dbc.Col(html.H1("Dividend Entry Uploader", className="card-title"))),
            
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=date(2000, 8, 5),
                max_date_allowed=datetime.today(),
                # initial_visible_month=date(2017, 8, 5),
                date=(datetime.today() + pd.offsets.MonthEnd(0)),
                # disabled_days=,
                clearable =True),
            
            dbc.Row(dbc.Col(dbc.Label('Getting index members only?'), width=10)),
            dcc.RadioItems(
                id='index-only-radio',
                options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False}
                ],
                value=True,
                labelStyle={'display': 'inline-block'}),
            
            html.Hr(),
            dbc.Row(dbc.Col(dbc.Label('Select a fsym id'), width=10)),
            dbc.Row(dbc.Col(
                dcc.Dropdown(
                    id="upload-fsym-id-dropdown",
                    options=[{"label": 'All', "value": 'All'}] +\
                        [{"label": i, "value": i} for i in fsym_id],
                    value=fsym_id[0]), width=10), justify='center'),
            dbc.Row(dbc.Col(
                dcc.Dropdown(
                    id="currency-dropdown",
                    options=[{"label": 'All', "value": 'All'}] +\
                        [{"label": i, "value": i} for i in cur_list],
                    value=fsym_id[0]), width=10), justify='center'),
            dbc.Row(),
            dbc.Row(),
            html.Details([
                html.Summary('Label of the item'),
                html.Div('Contents')
            ]),
            dbc.Row(dbc.Col(dbc.Button(id="upload-skipped-button", n_clicks=0, 
                                       children='Upload skipped', color='success'), 
                            width=2), justify='end'),
            html.Br(),
            dbc.Row(dbc.Col(dbc.Button(id="upload-button", n_clicks=0, 
                                       children='Upload to DB', color='success'), 
                            width=2), justify='end'),
            ]))

    
def div_editor():
    return dbc.Card(
        dbc.CardBody([
        
        html.Br(),
        dbc.Row(html.H1("Dividend Entry Editor"), justify='center'),
        html.Br(),
        dbc.Row(dbc.Col(html.H3("Data")), justify='start'),

        dbc.Row(dbc.Col(dbc.Label('Select a fsym id'), width=10)),
        dbc.Row(dbc.Col(dcc.Dropdown(
                    id="fsym-id-dropdown",
                    options=[{"label": 'All', "value": 'All'}] + [{"label": i, "value": i} for i in fsym_id],
                    value=fsym_id[0],
                ), width=10), justify='center'),
        
        # Hidden datatable for storing data
        html.Div([dash_table.DataTable(
            id='data-table',
            columns=[{'name': i, 'id':i} for i in data.columns],
            editable=True,
            data=data.to_dict('records')
        )], style= {'display': 'none'}),
        
    
    
        # dbc.Row(html.Br()),
        
        dbc.Row(dbc.Col(dash_table.DataTable(
            id='output-data-table',
            columns=[{'name': 'fsym_id', 'id': 'fsym_id', 'type': 'text', 'editable': False}] +
            [{'name': i, 'id':i, 'presentation': 'dropdown', 'editable': True} for i in ['listing_currency', 'payment_currency']] + 
            [{'name': i, 'id': i, 'type': 'datetime', 'editable': True} for i in ['declared_date', 'exdate', 'record_date', 'payment_date']] +
            [{'name': 'payment_amount', 'id': 'payment_amount', 'type': 'numeric', 'editable': True}] +
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
        ), width=10), justify='center'),
        dbc.Row(html.Div(id='table-dropdown-container')),
    
        
        html.Br(),
        html.Br(),  
        dbc.Row(html.H3("Modified History"), justify='start'),
    
        dbc.Row(dbc.Col(dash_table.DataTable(
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
        ), width=10), justify='center'),
        
        dbc.Row(html.H5('Save changes'), justify='start'),
    
        dbc.Row(dbc.Col(dbc.Alert(id="save-msg", children="Press this button to save changes", color="info"), width=10), justify='center'),
        dbc.Row(dbc.Col(dbc.Button(id="save-button", n_clicks=0, children='Save', color='success'), width=2), justify='end'),
        
        ]),             
        className="w-85")


# App Layout
app.layout = dbc.Container([
    div_uploader(), html.Br(), div_editor()], fluid=True)

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
def export_modified_data(nclicks, modified_data): 
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