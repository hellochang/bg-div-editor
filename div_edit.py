# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:52:24 2021

@author: Chang.Liu
"""

import json
import time
import dash
from flask import Flask
# import dash_table
# import dash_core_components as dcc
import dash_bootstrap_components as dbc
# import dash_html_components as html
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# Plotting
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, date 

import functools
from dash.long_callback import DiskcacheLongCallbackManager

## Diskcache
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# import os
# os.chdir(r'C:\Users\Chang.Liu\Documents\dev\div_data_uploader')
from bg_data_uploader import *
# import sys
# sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')
# from bg_data_importer import DataImporter

data_importer_dash = DataImporter(False)

# =============================================================================
# Div Modifier Helpers
# =============================================================================

cur_list = ['USD','CAD','EUR','GBP','JPY']


# =============================================================================
# Div Uploader Helpers
# =============================================================================
def plot_dividend_data(fsym_id, new_data, alt_cur=None):
    msg = ''
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
        already_exist = [date.strftime('%Y/%m/%d/') for date in already_exist],
        msg = f'Payments on {already_exist} already exists.'
        # with outs2:
        #     clear_output()
            # display(f'Payments on {str(already_exist)} already exists.')
        return fig, msg
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
        fig.update_layout(shapes=[dict(type="rect", xref="x", yref="paper",x0=new['exdate'].min(),y0=0,x1=new['exdate'].max(),y1=1,fillcolor="LightSalmon",opacity=0.5,layer="below",line_width=0)])
    return fig, msg

#TODO func signature change
def compare_new_data_with_factset(secid, update_date, bbg, factset=None):
    if factset is None:
        factset = factset_new_data(secid)
    if check_exist:
        factset = factset[factset['exdate'] <= update_date]
    # bbg = new_data[new_data['fsym_id']==secid].copy()
    # print('preprocess_bbg_data')
    # print(dates_cols)
    bbg['exdate'] = pd.to_datetime(bbg['exdate'], format='%Y-%m-%d')
    # print(bbg.dtypes)
    # print(factset.dtypes)
    new_data_comparison = pd.merge(bbg, factset, how='outer', on=['fsym_id','exdate','div_type'], suffixes=('_bbg','_factset'))
    new_data_comparison = new_data_comparison.sort_values(['exdate'])
    new_data_comparison = new_data_comparison.reset_index(drop=True)
    new_data_comparison = new_data_comparison[new_data_comparison.filter(regex='fsym_id|exdate|payment_date|amount').columns]
    new_data_comparison['check_amount'] = np.where(abs(new_data_comparison['payment_amount_factset']-new_data_comparison['payment_amount_bbg'])>0.001, 'Mismatch', 'Good')
    new_data_comparison['check_payment_date'] = np.where(new_data_comparison['payment_date_factset']!=new_data_comparison['payment_date_bbg'], 'Mismatch', 'Good')
    return new_data_comparison

#TODO changed func sig
def prepare_bbg_data(new_data, alt_cur=''):
    # if regular_skipped == 'regular':
    #     new = new_data[new_data['fsym_id']==fsym_id].copy()
    # elif regular_skipped == 'skipped' :
    #     new = skipped[skipped['fsym_id']==fsym_id].copy()
        
    (last_cur, fstest_cur) = dividend_currency(new_data)
    if last_cur is None:
        last_cur = fstest_cur
        new_data['listing_currency'] = fstest_cur
    if alt_cur != '':
        new_data['payment_currency'] = alt_cur.upper()
    else:
        new_data['payment_currency'] = last_cur
    return new_data


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
    return fig
        

        
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
    return fig
    

# =============================================================================
# Dash app
# =============================================================================
# server = Flask(__name__)
app = dash.Dash(
    __name__,
     # server=server, url_base_pathname='/',
    meta_tags=[{"name": "viewport", 
                "content": "width=device-width, initial-scale=1"}],
    # long_callback_manager=long_callback_manager,
    # prevent_initial_callbacks=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.config.suppress_callback_exceptions = True

# fsym_id = data['fsym_id'].unique()

# print(data.dtypes)
modify_data = pd.DataFrame([], columns=['fsym_id'])
@app.callback(
    Output("collapse-editor", "is_open"),
    # Output('edit-data-table', 'data'),
    Output('editor-fsym-id-dropdown', 'options'),
    Output('editor-fsym-id-dropdown', 'value'),
    Input("collapse-button", "n_clicks"),
    Input('modify-list', 'data'),
    State("collapse-editor", "is_open"),
    State('new-data-data-table', 'data'),
    running=[
        (Output("collapse-button", "disabled"), True, False),
    ],
    manager=long_callback_manager,
    )
def get_editor_data(n, modify_lst, is_open, datatable):
    df = pd.DataFrame(datatable)
    # print('get_editor_data')
    # print(datatable)
    # print(modify_lst)

    lst = pd.DataFrame(modify_lst)
    # print(lst)
    lst = lst['name']
    global modify_data #TODO Use dictionary for loops instead
    modify_data = df[df['fsym_id'].isin(lst)]
    fsym_ids = [{"label": i, "value": i} for i in modify_data['fsym_id']]
    # fsym_ids = [{"label": 'All', "value": 'All'}] + [{"label": i, "value": i} for i in modify_data['fsym_id']]
    # print('modify_data')
    # print(modify_data)
    if n:
        return not is_open, fsym_ids , fsym_ids[0]['value']
    print(is_open)
    # print(options)
    return is_open, fsym_ids , fsym_ids[0]['value']

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

@app.callback(
    Output('modified-data-rows', 'style_data_conditional'),
    Input('modified-data-rows', 'data'))
def highlight_changed_cell(data):
    df = pd.DataFrame(data)
    lst_idx, lst_changed_cell = find_changed_cell(df)
    # print('highlight_changed_cell')
    # print(lst_idx)
    # print(lst_changed_cell)
    return [
        {
            'if': {
                'filter_query': '{{id}} = {}'.format(i),  # matching rows of a hidden column with the id, `id`
                # 'row_index': i,
                'column_id': col
            },
            'backgroundColor': 'DodgerBlue',
            'color': 'white'
        }
        for i, col in zip(lst_idx, lst_changed_cell)
    ]
def find_changed_cell(df):
    df = df[~(df['action'] =='delete')]
    cols = df.columns[1:12].tolist()
    # print('find_changed_cell')
    # print(cols)
    lst_changed_cell = []

    # rows_orig = df[df['action'] == 'original'].reset_index()
    # rows_update = df[df['action'] == 'update'].reset_index()
    # num_row = rows_orig.shape[0]
    lst_idx = []
    # print(rows_orig)
    # print(rows_update)
    for idx, row in df.iterrows():
        if row['action'] == 'update':
            # print(f'cycle {idx}::')
            next_row = df.loc[idx+1]
            # print('row')
            # print(row)
            # print('next_row')
            # print(next_row)
            for col in cols:
                both_nan = (row[col] != row[col]) and (next_row[col] != next_row[col])
                if (row[col] != next_row[col]) and not both_nan:
                    lst_idx = lst_idx + [idx, idx+1]
                    lst_changed_cell = lst_changed_cell + [col, col]
        # print('lst_idx')
        # print(lst_idx)
    return lst_idx, lst_changed_cell   
# def find_changed_cell(df):
#     # df = df[~(df['action'] =='delete')]
#     cols = df.columns[1:12].tolist()
#     # print('find_changed_cell')
#     # print(cols)
#     lst_changed_cell = []

#     rows_orig = df[df['action'] == 'original'].reset_index()
#     rows_update = df[df['action'] == 'update'].reset_index()
#     num_row = rows_orig.shape[0]
#     lst_idx = [i for i in range(2*num_row)]
#     # print(rows_orig)
#     # print(rows_update)
#     for i in range(num_row):
#         row_o = rows_orig.loc[i]
#         # print(row_o)
#         row_u = rows_update.loc[i]       
#         # print(row_u)

#         for col in cols:
#             if row_o[col] != row_u[col]:
#                 lst_changed_cell = lst_changed_cell + [col, col]
#     return lst_idx, lst_changed_cell
seclist = []#TODO combine with the other Card component later
# all_goods = []
# # manual_list = []
#     dcc.Store(id='all-goods-list'),
#     dcc.Store(id='mismatch-list'),
#     dcc.Store(id='skipped-list'),
@app.long_callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('mismatch-list', 'data'),
    Output('progress-div', 'style'),
    # Output('all-goods-data-table', 'data'),
    # Output('all-goods-data-table', 'columns'),
    # Output('skipped-data-table', 'data'),
    # Output('skipped-data-table', 'columns'),
    # Output('mismatch-data-table', 'data'),
    # Output('mismatch-data-table', 'columns'),
    
    # Output('fsym-id-dropdown', 'options'),
    # Output('fsym-id-dropdown', 'value'),
    # Output('all-goods-msg', 'is_open'),
    # Output('no-data-msg', 'children'),
    # Output('no-data-msg', 'is_open'),
    # Output('main-panel-div', 'style'),
    # Output('uploader', 'style'),

    # Output('skipped-msg', 'is_open'),
    # Output('skipped-data-table-div', 'style'),
    # Output('skipped-dropdown', 'options'),
    # Output('mismatch-dropdown', 'options'),    
    # Output('load-data-msg', 'value'),
    Input('div-date-picker', 'date'), 
    Input('index-only-radio', 'value'),
    running=[
        (Output("div-date-picker", "disabled"), True, False),
        # (Output("collapse-editor", "is_open"), True, False),
        (Output("collapse-button-div", "style"), {'display': 'none'}, {}),

        # (Output('index-only-radio', "options"), [{'disabled': True}, {'disabled': True}], [{'disabled': False}, {'disabled': False}]),
        # (Output("index-only-radio", "style"), {'display': 'none'}, {}),
        (Output("main-panel-div", "style"), {'display': 'none'}, {}),
        (Output("view-type-div", "style"), {'display': 'none'}, {}),
    ],
    manager=long_callback_manager,
    progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
)
# @functools.lru_cache(maxsize=32)
def load_data_to_dash(set_progress, update_date, index_flag):
# def load_data_to_dash(update_date, index_flag):
    # monthend_date = (datetime.today() + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
    # update_date = input(f"Enter update date in yyyy-mm-dd format (Default: '{monthend_date}'): ")
    # if update_date == "":
    #     update_date = monthend_date
    print(f'update_date:{update_date}')
    f_date = update_date.replace('-','')
    new_data=pd.read_parquet(rf'\\bgndc\Analysts\Scheduled_Jobs\output\new_dvd_data_{f_date}.parquet')
    update_list = pd.read_csv(rf'\\bgndc\Analysts\Scheduled_Jobs\input\sec_list_{f_date}.csv')
    # splits = pd.read_csv(main_dir / Path(f"input\splits_{f_date}.csv")) 
    #TODO don't think splits is used anywhere. double check later
    total=4
    i=0
    # global skipped
    (new_data, skipped, pro_rata, splits) = \
        preprocess_bbg_data(new_data, update_list, index_flag, update_date)
    
    set_progress((str(i + 1), str(total)))

    new_data = bbg_data_single_security(new_data)
    set_progress((str(i + 1), str(total)))

    skipped = process_skipped(skipped)
    seclist = sorted(list(new_data['fsym_id'].unique()))
    set_progress((str(i + 1), str(total)))

    # global manual_list
    # global all_goods
    (manual_list, all_goods) = bulk_upload(new_data, update_date)
    set_progress((str(i + 1), str(total)))
 
    print('load_data_to_dash')
    # print(new_data.dtypes)
    
    print('Loaded')#TODO
    print(all_goods)
    print(manual_list)
    print(skipped)
    # print('new_data')
    # print(new_data)
    dropdown_options = pd.DataFrame({'all_goods': pd.Series(sorted(all_goods['fsym_id'].to_list())) if all_goods is not None else pd.Series([]),
                  'mismatch': pd.Series(sorted(manual_list)) if manual_list is not None else pd.Series([]),
                  'skipped': pd.Series(skipped['fsym_id'].unique()) if skipped is not None else pd.Series([])})
    # print(dropdown_options)
 
    # print(has_data)
    # if not has_data:
    #     raise dash.exceptions.PreventUpdate
    #     return {}, [], f'There is no entry to be reviewed for {selected_review_option}', not has_data, {'display': 'none'}

    # no_data_msg = f'There is no entry to be reviewed for {selected_review_option}' if not has_data else 'Data loaded'
    # display_option = {'display': 'none'} if not has_data else {}
    return new_data.to_dict('records'), [{'name': i, 'id':i} for i in new_data.columns],\
            dropdown_options.to_dict('records'), {'display': 'none'}
           # fsym_id_dropdown_options, fsym_id_dropdown_options[0]['value'],\
           # no_data_msg,\
           # not has_data, display_option
           # {'display': 'none'} if not has_skipped else {}
                          # all_goods, all_goods_col,\
                              
             # skipped.to_dict('records'), [{'name': i, 'id':i} for i in skipped.columns],\

            # [{"label": i, "value": i} for i in sorted(all_goods['fsym_id'].to_list())],\
            # [{"label": all_goods['fsym_id'].to_list()[0], "value": all_goods['fsym_id'].to_list()[0]}],\
            # [{"label": i, "value": i} for i in sorted(manual_list)]
            # manual_list.to_dict('records'), [{'name': i, 'id':i} for i in manual_list.columns],\

@app.callback(
    Output('new-data-data-table', 'data'),
    Output('new-data-data-table', 'columns'),
        # Output('all-goods-msg', 'is_open'),
    Output('fsym-id-dropdown', 'options'),
    # Output('fsym-id-dropdown', 'value'),
    Output('no-data-msg', 'children'),
    Output('no-data-msg', 'is_open'),
    Output('main-panel', 'style'),
    Output('modified-data-rows', 'columns'),
    Input('view-type-radio', 'value'),
    Input('data-table', 'data'),
    Input('mismatch-list', 'data'))
    # Input('modified-data-rows', 'data'),
    # State('modified-data-rows', 'data_previous'),
    # State('fsym-id-data-table', 'data'))
# @functools.lru_cache(maxsize=32)
def load_selected_data(selected_review_option, datatable, view_type_lst):
    df = pd.DataFrame(datatable)
    df_selection = pd.DataFrame(view_type_lst)
    print('load_selected_data')
    print(df_selection)
    all_goods = df_selection['all_goods'].unique()
    manual_list = df_selection['mismatch'].unique()
    skipped = df_selection['skipped'].unique()
    # print('load_selected_data')
    # print(datatable)
    # print(df)
    # print(manual_list)

    if selected_review_option == 'all_goods':
        selected_ids = all_goods
    if selected_review_option == 'mismatch':
        selected_ids = manual_list
    if selected_review_option == 'skipped': 
        selected_ids = skipped       
    
    if selected_ids[-1] is None:
        selected_ids = selected_ids[:-1] 
    has_data = selected_ids.size > 0
    fsym_id_dropdown_options = [{"label": i, "value": i} for i in selected_ids] if has_data else []
    print('load_selected_data')
    print(df)
    df = df[df['fsym_id'].isin(selected_ids)] if has_data else pd.DataFrame([])

    # print(fsym_id_dropdown_options)

    # print('Selected')#TODO
    # print('Selected: ')#TODO

    # print(all_goods)
    # # print('all_goods')#TODO

    # print(manual_list)
    # # print('manual_list')#TODO

    # # print(skipped['fsym_id'].unique())
    # print(skipped)#TODO

    # print(has_data)
    # if not has_data:
    no_data_msg = f'There is no entry to be reviewed for {selected_review_option}' if not has_data else 'Data loaded'
    display_option = {'display': 'none'} if not has_data else {}
    
    # modified_df = pd.DataFrame(modified_datatable) 
    # print('modified_df')
    # print(modified_df)
    # if modified_df.shape[0] > 0: 
    #     fsym_id = modified_df['fsym_id'].unique()[0]
    #     df = df.loc[~(df['fsym_id'] == fsym_id)]
    #     # df[df['fsym_id'] == fsym_id] = modified_df
    #     res = pd.concat([df, modified_df])
    #     print('res')
    #     print(res)
    #     res = res.to_dict('records')
    # else:
    #     print('res for empty case')
    #     print(df)
    #     res = df.to_dict('records')
    # # print('fsym_id_dropdown_options')
    # # print(fsym_id_dropdown_options)
    # undo_delete_rows = [row for row in rows_prev if row not in rows] if rows is not None and rows_prev is not None else []
    # res = res + undo_delete_rows
    
    return df.to_dict('records'),\
        [{'name': i, 'id':i} for i in df.columns],\
           fsym_id_dropdown_options,\
           no_data_msg,\
           not has_data, display_option,\
           [{'name': 'action', 'id':'action'}] + [{'name': i, 'id':i} for i in df.columns]
                          # fsym_id_dropdown_options[0]['value'] if has_data else None,\

#     Output('new-data-data-table', 'data'),
#     Input('output-data-table', 'data'),
#     State('new-data-data-table', 'data'),
#     State('modified-data-rows', 'data'),
#     State('modified-data-rows', 'data_previous'))
# def update_data_table(modified_datatable, datatable, rows, rows_prev):
#     df = pd.DataFrame(datatable)
#     modified_df = pd.DataFrame(modified_datatable)  
#     fsym_id = modified_df['fsym_id'].unique()[0]
#     df = df.loc[~(df['fsym_id'] == fsym_id)]
#     # df[df['fsym_id'] == fsym_id] = modified_df
#     res = pd.concat([df, modified_df]).to_dict('records')
#     return res + [row for row in rows_prev if row not in rows] if rows is not None else res

@app.callback(
    Output('fsym-id-data-table', 'data'),
    Output('fsym-id-data-table', 'columns'),
    Input('fsym-id-dropdown', 'value'),
    # State('modified-data-rows', 'data'),
    # State('modified-data-rows', 'data_previous'),
    Input('new-data-data-table', 'data'))
def filter_fysm_id_data(selected, datatable):
    if datatable is None: return dash.no_update
    df = pd.DataFrame(datatable)
    # print("filter_fysm_id_and_undo_delete: datatbl")
    # print(df)
    res = df[df['fsym_id'] == selected]
    return res.to_dict('records'), [{'name': i, 'id':i} for i in res.columns]

@app.callback(
    Output('basic-info', 'children'),
    Output('comparison-data-table', 'data'),
    Output('comparison-data-table', 'columns'),
    # Output('fsym-id-data-table', 'data'),
    # Output('fsym-id-data-table', 'columns'),
    Output('fsym-id-graph', 'figure'),
    Output('comparison-msg', 'children'),
    # Output('payment-exist-msg', 'children'),
    # Output('load-data-msg', 'value'),
    # Input('fsym-id-dropdown', 'value'),
    Input('fsym-id-data-table', 'data'),
    State('div-date-picker', 'date'))
def get_basic_info(new_data, update_date):
    new_data = pd.DataFrame(new_data)
    fsym_id = new_data['fsym_id'].values[0]
    print('get_basic_info')
    # print(new_data.dtypes)
    # print(fsym_id)
    global check_exist
    check_exist = check_existence(fsym_id)
    basic_info_str = basic_info(fsym_id, new_data)
    payment_exist_msg = ''
    if not check_exist:
        comparison_msg = "This is a newly added."
        # new = new_data[new_data['fsym_id']==fsym_id].copy()
        factset = factset_new_data(fsym_id)
        comparison = compare_new_data_with_factset(fsym_id, update_date, new_data, factset)
        fig = plot_dividend_data_comparison(factset, new_data)
        # display(HTML(comparison.to_html()))
    else:
        (last_cur, fstest_cur) = dividend_currency(new_data)
        # with outs2:
        comparison = compare_new_data_with_factset(fsym_id, update_date, new_data)
        if fstest_cur != last_cur:
            comparison_msg = f'Possible currency change. Last payment:{last_cur}. Factset payment:{fstest_cur}'
            # display(HTML(df.to_html()))
        # with outs:
        #     clear_output()
        # comparison = compare_new_data_with_factset(fsym_id, update_date, new_data)
        # new = new_data[new_data['fsym_id']==fsym_id].copy()
        new_data['listing_currency'] = fstest_cur
        new_data['payment_currency'] = last_cur
        comparison_msg = 'New Dividend Data'
        # display(HTML(new.to_html()))
        fig, payment_exist_msg = plot_dividend_data(fsym_id, new_data)
        if payment_exist_msg != '': comparison_msg = payment_exist_msg 
    return basic_info_str, comparison.to_dict('records'), [{'name': i, 'id':i} for i in comparison.columns],\
             fig, comparison_msg
         # new.to_dict('records'), [{'name': i, 'id':i} for i in new.columns],\

@app.callback(
    Output('factset-graph', 'figure'),
    # Output('load-data-msg', 'value'),
    Output('factset-card', 'style'),
    Output('factset-warning-msg', 'is_open'),
    Output('factset-warning-msg', 'children'),
    # Input('fsym-id-dropdown', 'value'),
    Input('fsym-id-data-table', 'data'))
def plot_comparison(new_data):
    bbg = pd.DataFrame(new_data)
    # bbg = new_data[new_data['fsym_id'] == fsym_id].copy()
    fsym_id = bbg['fsym_id'].values[0]
    query = f"select * from fstest.dbo.bg_div where fsym_id ='{fsym_id}'"

    try:
        bg = data_importer_dash.load_data(query)
    except Exception as e:
        warning_msg = f'Error while loading data from DB: {e}'
        fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'}))
        return fig, {'display': 'none'}, True, warning_msg
    fig = plot_dividend_data_comparison(bg, bbg)
    return fig, {}, False, ''


@app.callback(
    Output('bbg-graph', 'figure'),
    # Output('bbg-data-table', 'data'),
    # Output('bbg-data-table', 'columns'),
    # Output('load-data-msg', 'value'),
    # Input('fsym-id-dropdown', 'value'),
    # State('new-data-data-table', 'data'),
    Input('fsym-id-data-table', 'data'))
def plot_bbg(new_data):
    new_data = pd.DataFrame(new_data)
    df = prepare_bbg_data(new_data)
    
    # with outs2:
    #     clear_output()
    fig = plot_generic_dividend_data(df)
    return fig
# , df.to_dict('records'), [{'name': i, 'id':i} for i in df.columns]

@app.callback(
    Output('bg-db-graph', 'figure'),
    Output('bg-db-data-table', 'data'),
    Output('bg-db-data-table', 'columns'),
    Output('bg-db-card', 'style'),
    Output('bg-db-warning-msg', 'is_open'),
    Output('bg-db-warning-msg', 'children'),
    # Output('bg-card', 'is_open'),

    # Output('bbg-data-table', 'data'),
    # Output('bbg-data-table', 'columns'),
    # Output('load-data-msg', 'value'),
    Input('fsym-id-dropdown', 'value'))
def plot_db(fsym_id):
    query = f"select * from fstest.dbo.bg_div where fsym_id ='{fsym_id}'"
    try:
        df = data_importer_dash.load_data(query)
        df.sort_values(by=['exdate'], ascending=False, inplace=True)
        fig = plot_generic_dividend_data(df)
    except Exception as e:
        warning_msg = f'Error while loading data from DB: {e}'
        fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'}))
        return fig, {}, [], {'display': 'none'}, warning_msg
    
    if df is None:
        has_data = False
        warning_msg = 'This holding is not in the DB.' 
        cols = []
        df = {}
    else:
        has_data = True
        warning_msg = ''
        cols = [{'name': i, 'id':i} for i in df.columns]
        df = df.to_dict('records')        
    return fig, df, cols, {'display': 'none'} if not has_data else {}, has_data, warning_msg


@app.callback(
    Output('split-data-table', 'data'),
    Output('split-data-table', 'columns'),
    # Output('load-data-msg', 'value'),
    Input('fsym-id-dropdown', 'value'))
def check_split_history(fsym_id):
    query = f"""select 
                fsym_id,p_split_date,p_split_factor, 
                exp(sum(log(p_split_factor))  OVER (ORDER BY p_split_date desc)) cum_split_factor 
                from fstest.fp_v2.fp_basic_splits where fsym_id= '{fsym_id}'
                order by p_split_date
            """
    df = data_importer_dash.load_data(query)
    df.sort_values(by=['p_split_date'], ascending=False, inplace=True)
    data = df.to_dict('records') if df is not None else []
    cols = [{'name': i, 'id':i} for i in df.columns] if df is not None else []
    return data, cols

  
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
        #          dbc.Alert(id="bg-db-warning-msg", color="info", is_open=False)])
    ]
)

comparison_panel = html.Div([row_2, dbc.Row(dbc.Col(html.Br())), row_3])
print(modify_data)
def div_editor():
    return html.Div([
    html.Div(dbc.Button(
        "Edit entries",
        id="collapse-button",
        className="mb-3",
        color="primary",
        n_clicks=0,
    ), id='collapse-button-div'),
    
    dbc.Collapse(dbc.Card(
        dbc.CardBody([
        
        # html.Br(),
        # dbc.Row(html.H1("Dividend Entry Editor"), justify='center'),
        # html.Br(),
        dbc.Row(dbc.Col(html.H3("Data editor")), justify='start'),

        # dbc.Row(dbc.Col(dbc.Label('Select a fsym id'), width=10)),
        # dbc.Row(dbc.Col(dcc.Dropdown(
        #             id="fsym-id-dropdown",
        #             options=[{"label": 'All', "value": 'All'}] + [{"label": i, "value": i} for i in fsym_id],
        #             value=fsym_id[0],
        #         ), width=10), justify='center'),
        
        # Hidden datatable for storing data
        
        
        # dbc.Row(dbc.Col(dbc.Label('Select a fsym id'), width=10)),
        dbc.Row(dbc.Col(dcc.Dropdown(
                    id="editor-fsym-id-dropdown",
                    # options=[{"label": 'All', "value": 'All'}] + [{"label": i, "value": i} for i in modify_data['fsym_id']],
                )), justify='center'),
        
        # Hidden datatable for storing data
        html.Div([dash_table.DataTable(
            id='edit-data-table',
            columns=[{'name': i, 'id':i} for i in modify_data.columns],
            editable=True,
            data=modify_data.to_dict('records')
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
            # page_size=20,
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
        
            # Workaround for bug regarding display row dropdown with Boostrap
            css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}]
        )), justify='center'),
        html.Br(),
        html.Br(),  
        dbc.Row(html.Div(id='table-dropdown-container')),
    
        
        html.Br(),
        html.Br(),  
        dbc.Row(html.H3("Edit History"), justify='start'),
    
        dbc.Row(dbc.Col(dash_table.DataTable(
            id='modified-data-rows',
            data=[],
            # filter_action="native",
            sort_action="native",
            sort_mode="multi",
            page_action="native",
            page_current= 0,
            page_size= 10,
            # style_data_conditional=highlight_special_case_row(),
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
        )), justify='center'),
        
        html.Br(),
        dbc.Row(dbc.Col(dbc.Button(id="upload-modified-button", n_clicks=0, 
                                   children='Upload modified to DB', color='success'), 
                        width=2), justify='end'),
        html.Br(),
        dbc.Row(dbc.Col(dbc.Alert(id="save-modified-msg", color="info", is_open=False, duration=600), width=10), justify='end'),
 
        ]),             
        className="w-85"),
        id="collapse-editor",
        is_open=False,
    )
])

def core_functionalities():
        return dbc.Card(
            dbc.CardBody([
                dbc.Row(dbc.Col([])),
                dbc.Row(dbc.Col([])),
                dbc.Row(dbc.Col(dcc.Markdown(id="basic-info")), justify='center'),
                html.Br(),
                # dbc.Row(dbc.Col(dbc.Label('Comparison table')), justify='center'),
                html.Div(id='payment-exist-msg'), 
                html.Br(),
                html.Div([
                    dbc.Row(dbc.Col(dbc.Alert(id="comparison-msg", color="info")), justify='center'),
                    dash_table.DataTable(
                        id='comparison-data-table',
                        # columns=[{}],
                        # data={}
                )]),
                html.Br(),
                dbc.Row(dbc.Col(dash_table.DataTable(id='fsym-id-data-table'))),
                # html.Div([dash_table.DataTable(
                #     id='fsym-id-data-table',
                #     # columns=[{}],
                #     # data={}
                # )]),
                html.Br(),
                # div_editor(),
                dcc.Graph(id='fsym-id-graph'),
                comparison_panel,


            ]))

def fsym_id_selection():
    return html.Div([
                dbc.Row(dbc.Alert(id="add-to-editor-msg", children='Saved for modifying later', color="success", duration=500, is_open=False)),
                dbc.Row(dbc.Alert(id='warning-end-dropdown', color="warning", duration=700, is_open=False)),
                dbc.Row([dbc.Col(dbc.Button(id="prev-button", n_clicks=0, 
                                    children='Prev'), 
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
    
        
@app.callback(
        Output('modify-list', 'data'),
        Output('add-to-editor-msg', 'is_open'),
        Input('modify-button', 'n_clicks'),
        State('fsym-id-dropdown', 'value'),
        State('modify-list', 'data')
        )
def update_modify_list(n_clicks, cur_fsym_id, modify_list):
    if n_clicks:
        modify_list.append({'name': cur_fsym_id, 'id': cur_fsym_id})
        # print('update_modify_list')
        # print(modify_list)
        return modify_list, True
             
@app.callback(
        Output('fsym-id-dropdown', 'value'),
        Output('warning-end-dropdown', 'children'),
        Output('warning-end-dropdown', 'is_open'),
        Input('next-button', 'n_clicks'),
        Input('prev-button', 'n_clicks'),
        Input('fsym-id-dropdown', 'value'),
        State('view-type-radio', 'value'),
        State('mismatch-list', 'data')
        )
def go_to_next_prev(prev_clicks, next_clicks, cur_fsym_id, view_type, view_type_lst):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]    
    lst = [row[view_type] for row in view_type_lst]
    if lst[-1] is None:
        lst = lst[:-1] ,'', False
    lst.sort()
    if not prev_clicks and not next_clicks:
        return lst[0]
    if prev_clicks or next_clicks:
        idx = lst.index(cur_fsym_id)-1 if 'prev-button.n_clicks' == changed_id else lst.index(cur_fsym_id)+1
        beg_end_msg = 'first' if idx < 0 else 'last'
        if idx >= len(lst) or idx < 0:
            end_dropdown_msg = f'This is the {beg_end_msg} Fsym Id.'
            return dash.no_update, end_dropdown_msg, True
        return lst[idx], '', False


main_panel = html.Div([
    dbc.Alert(id="no-data-msg", color="info", is_open=False),
    
    dbc.Card(
    id = 'main-panel',
    children = [dbc.CardBody([
            
        # Hidden datatable for storing data
        html.Div([dash_table.DataTable(
            id='data-table',
            editable=True,
        )], style= {'display': 'none'}),
        

        html.Div([dash_table.DataTable(
            id='new-data-data-table',
            # columns=[{'name': i, 'id':i} for i in data.columns],
            # data=data.to_dict('records')
        )], style= {'display': 'none'}),

        
        fsym_id_selection(),
        # dcc.Loading(id="is-loading-data", children=[fsym_id_selection()], type="default"),
        core_functionalities(),
        
        
        html.Br(),#TODO
        dbc.Row(dbc.Col(dbc.Button(id="upload-button", n_clicks=0, 
                                   children='Upload original to DB', color='success'), 
                        width=2), justify='end'),
        dbc.Row(dbc.Col(dbc.Alert(id="save-msg", color="info", is_open=False, duration=600), width=10), justify='end'),               
        ])])], id='main-panel-div')

data_view_type_selection = html.Div(id='view-type-div', children=[dbc.Row(dbc.Col(dbc.Label('Select the type of data'), width=10)),
            dbc.RadioItems(
                id='view-type-radio',
                options=[
                    {'label': 'All Goods', 'value': 'all_goods'},
                    {'label': 'Mismatched', 'value': 'mismatch'},
                    {'label': 'Skipped', 'value': 'skipped'}
                ],
                # value='mismatch',
                value='mismatch',
                inline=True)])
def top_select_panel():
    return dbc.Card(
        dbc.CardBody([
            html.Br(),
            dbc.Row(dbc.Col(html.H1("Dividend Entry Uploader", className="card-title"))),
            html.Br(),
            dcc.DatePickerSingle(
                id='div-date-picker',
                min_date_allowed=date(2000, 8, 5),
                max_date_allowed=date.today(),
                # initial_visible_month=date(2017, 8, 5),
                date = date(2021, 11, 30),
                # date=(datetime.today() + pd.offsets.MonthEnd(0)),
                # disabled_days=,
                # display_format='YYYYMMDD',
                clearable =True),
            html.Br(),
            html.Br(),
            dbc.Row(dbc.Col(dbc.Label('Getting index members only?'), width=10)),
            dbc.RadioItems(# TODO needs to fix the radio
                id='index-only-radio',
                options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False}
                ],
                value=False,
                labelStyle={'display': 'inline-block'}),
            html.Div(id='progress-div', children=[html.Progress(id="progress_bar")]),
            html.Hr(),
            html.Br(),
            data_view_type_selection,
            html.Hr(),
            ]), className='mt-3')


def div_uploader():
   return html.Div([
        main_panel,
       ], id='uploader')


# App Layout
app.layout = dbc.Container([
    # dcc.Store(id='all-goods-list'),
    dcc.Store(id='mismatch-list'),
    dcc.Store(id='modify-list', data=[]),

    top_select_panel(),
    html.Br(), 
    # dcc.Loading(id="is-loading-top-panel", children=[top_select_panel()], type="default"),
    # dcc.Loading(id="is-loading-data", children=[dbc.Alert(id="is-loading-msg")], type="default"),
    # dcc.Loading(id="is-loading-div-uploader", children=[div_uploader()], type="default"),
    div_uploader(),
    # div_uploader(),
    html.Br(), 
    div_editor()
    ], fluid=True)


# @app.callback(
#     # Output('data-table', 'data'),
#     Output('modified-data-rows', 'data'),
#     # Input('data-table', 'data_timestamp'),
#     Input('fsym-id-data-table', 'data_previous'),
#     State('fsym-id-data-table', 'data'),
#     State('modified-data-rows', 'data'))
# def update_modified_data_table(rows_prev, rows, modified_rows):
#     len_rows = len(rows) if rows is not None else 0
#     len_rows_prev = len(rows_prev) if rows_prev is not None else 0
#     if rows_prev is None: return []
#     if len_rows == len_rows_prev:
#         modified_rows = [i for i in rows if i not in rows_prev] if modified_rows is None else modified_rows + [i for i in rows if i not in rows_prev]
#         modified_rows[-1]['action'] = 'update'
#         modified_rows= modified_rows + [i for i in rows_prev if i not in rows]
#         modified_rows[-1]['action'] = 'original'
#     if len_rows < len_rows_prev:
#          modified_rows = [i for i in rows_prev if i not in rows] if modified_rows is None else modified_rows + [i for i in rows_prev if i not in rows]
#          modified_rows[-1]['action'] = 'delete'
#     # if (len(rows) == len(rows_prev)):ss
#     #     modified_rows = [i.update({'action': 'update'}) for i in rows if i not in rows_prev] if modified_rows is None else modified_rows + [i.update({'action': 'update'}) for i in rows if i not in rows_prev]
#     #     modified_rows= modified_rows + [i.update({'action': 'original'}) for i in rows_prev if i not in rows]
#     # if (len(rows) < len(rows_prev)):
#     #      modified_rows = modified_rows + [i.update({'action': 'delete'}) for i in rows_prev if i not in rows] 
#     idx = 0
#     for row in modified_rows:
#         row.update({'id': idx})
#         idx = idx + 1
#     return modified_rows


 

@app.callback(
    Output('output-data-table', 'data'),
    Input('editor-fsym-id-dropdown', 'value'))
    # State('edit-data-table', 'data'))
def filter_fysm_id_editor(selected):
    # print('filter_fysm_id_editor____________')
    if selected == 'All':
        return modify_data.to_dict('records')
    # print(modify_data)
    return modify_data[modify_data['fsym_id'] == selected].to_dict('records')

@app.callback(
    Output('edit-data-table', 'data'),
    # State('edit-data-table', 'data'),
    Input('modified-data-rows', 'data'),
    State('modified-data-rows', 'data_previous'),
    State('editor-fsym-id-dropdown', 'value'),
    State('output-data-table', 'data')
)
def update_changed_data_table(rows, rows_prev, fsym_id, modified_datatable):
    global modify_data
    # global i
    # df = modify_data.copy()
    modified_df = pd.DataFrame(modified_datatable) 
    # print(f'update_changed_data_table: ____________{i}')
    # print('modified_df: ')
    # print(modified_df)
    # print('modify_data: ')
    modify_data = modify_data[~(modify_data['fsym_id'] == fsym_id)]
    # print(modify_data)
    # df[df['fsym_id'] == fsym_id] = modified_df
    modify_data = pd.concat([modify_data, modified_df])
    # print('updated_row:')
    # print(modify_data)
    res = modify_data.to_dict('records')
    undo_delete_row = [row for row in rows_prev if row not in rows] if (rows is not None and rows_prev is not None) and len(rows_prev) > len(rows) else []
    # print('rows')
    # print(pd.DataFrame(rows))
    # print('rows_prev')
    # print(pd.DataFrame(rows_prev))
    # print('undo_delete_row:')
    # print(pd.DataFrame(undo_delete_row))
    # res  = res +  undo_delete_row if undo_delete_row is not None else res
    # pd.DataFrame(undo_delete_row).drop(columns=['action'], inplace=True)
    modify_data = pd.concat([modify_data, pd.DataFrame(undo_delete_row)])
    # print('update_changed_data_table:')
    # print(modify_data)
    # i=i+1
    return res


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
    idx = 0
    for row in modified_rows:
        row.update({'id': idx})
        idx = idx + 1
    return modified_rows

#TODO
@app.callback(
        Output("save-msg", "children"),
        Output("save-msg", "is_open"),
        Input("upload-button", "n_clicks"),
        Input('modify-list', 'data'),
        State("new-data-data-table", "data")
        )
def export_not_modified_data(nclicks, modify_lst, modified_data): 
    if nclicks == 0:
        raise PreventUpdate
    else:
        df = pd.DataFrame(modified_data)
        lst = pd.DataFrame(modify_lst)
        lst = lst['name'] #TODO Use dictionary for loops instead
        print(df)
        df = df[~(df['fsym_id'].isin(lst))]
        datatypes = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
        df = df.astype(datatypes)
        # print(df.dtypes)
        df.sort_values(by=['fsym_id', 'exdate'], ascending=False)
        df.to_csv('not_edited_data')
        return 'Data saved to DB', True

@app.callback(
        Output("save-modified-msg", "children"),
        Output("save-modified-msg", "is_open"),
        Input("upload-modified-button", "n_clicks"),
        )
def export_modified_data(nclicks): 
    if nclicks == 0:
        raise PreventUpdate
    else:
        df = modify_data.copy()
        datatypes = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
        df = df.astype(datatypes)
        print(df)
        # print(df.dtypes)
        df.sort_values(by=['fsym_id', 'exdate'], ascending=False)
        df.to_csv('edited_div_data')
        return 'Data saved to DB', True

# Running the server
if __name__ == "__main__":
    # View the app at http://192.168.2.77:8080/ or, in general,
    #   http://[host computer's IP address]:8080/
    # app.run_server(debug=False, host='0.0.0.0', port = 8080)
    app.run_server(debug=True, port=80, dev_tools_silence_routes_logging = False)
    # app.run_server(debug=True, port=8080, use_reloader=False)
    # app.run_server(debug=False, port=8080, use_reloader=False)