# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:07:28 2022

@author: Chang.Liu
"""
import dash
from dash import callback_context, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
# from dash_extensions.enrich import Trigger, FileSystemCache

from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import dash_table
import pandas as pd
import numpy as np
import json
from typing import List, Tuple
from datetime import date
from bg_data_uploader import *

from timeit import default_timer as timer

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", 
                "content": "width=device-width, initial-scale=1"}],
    # long_callback_manager=long_callback_manager,
    # prevent_initial_callbacks=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
    
app.layout = dbc.Container([
    html.Div(dcc.Store(id='data-table'), style={'display': 'none'}),
    dcc.DatePickerSingle('div-date-picker',
                         date = date(2021, 12, 31),
                         clearable=True), 
    dcc.Dropdown(id="fsym-id-dropdown"),

    # dbc.Label('lst dict'),
    dcc.Store(id='filter-lst'),
    
    # dbc.Label('fsym id-data-table'),
    dcc.Store(id='filter-pd'),  



    # dash_table.DataTable(id='fsym-id-data-table'),
    dcc.Store(id='filter_pd_json-data-table'),

    dcc.Store(id='filter_json-data-table'),


    ], fluid=True)


@app.callback(
        Output('data-table', 'data'),
        Output('fsym-id-dropdown', 'options'),
        Output('fsym-id-dropdown', 'value'),
        # Output('filter-pd', 'columns'),
        # Output('filter-lst', 'columns'),
        Input('div-date-picker', 'date'), 
    )
def load_data_to_dash(update_date):
    print(f'update_date:{update_date}')
    f_date = update_date.replace('-','')
    new_data = pd.read_parquet(rf'\\bgndc\Analysts\Scheduled_Jobs\output\new_dvd_data_{f_date}.parquet')
    update_list = pd.read_csv(rf'\\bgndc\Analysts\Scheduled_Jobs\input\sec_list_{f_date}.csv')
    
    (new_data, skipped, pro_rata, splits) = \
        preprocess_bbg_data(new_data, update_list, False, update_date)
    new_data = bbg_data_single_security(new_data)

    seclist = new_data['fsym_id'].unique()
    dropdown = [{"label": i, "value": i} for i in seclist]
    print('end of load_data_to_dash')
    new_data = load_bg_div_data(seclist)
    cols = [{'name': i, 'id':i} for i in new_data.columns]
    
    print('load_data_to_dash')
    print(new_data)
    print(new_data.shape)
    return new_data.to_json(orient='split', date_format='iso'),\
        dropdown, new_data['fsym_id'].unique()[0],\
        # cols, cols
    # (manual_list, all_goods) = bulk_upload(new_data, update_date, factset_data)



# @app.callback(
#     Output('filter-lst', 'data'),
#     Input('fsym-id-dropdown', 'value'),
#     State('data-table', 'data'))
# def filter_lst_dict(selected, datatable):
#     filter_lst_dict_1 = timer()
#     # print('filter_lst_dict')
#     lst = [row for row in datatable if row['fsym_id'] == selected]
#     # print(lst)
#     dash.callback_context.record_timing('filter_lst_dict', timer() - filter_lst_dict_1, 'filter_lst_dict')
#     return lst


# @app.callback(
#     Output('filter-pd', 'data'),
#     Input('fsym-id-dropdown', 'value'),
#     State('data-table', 'data'))
# def filter_pd(selected, datatable):
#     filter_pd_1 = timer()
#     df = pd.DataFrame(datatable)
#     # print('filter_pd_default')
#     # print(df)
#     df = df[df['fsym_id']==selected]
#     res = df.to_dict('records')
    
#     dash.callback_context.record_timing('filter_pd_1', timer() - filter_pd_1, 'filter_pd_1')
#     return res


@app.callback(
    Output('filter_pd_json-data-table', 'data'),
    Input('fsym-id-dropdown', 'value'),
    State('data-table', 'data'))
def filter_pd_json(selected, datatable):
    filter_pd_json_1 = timer()
    df = pd.read_json(datatable, orient='split')
    df = df[df['fsym_id']==selected]
    res = df.to_json(orient='split', date_format='iso')
    # res = json.dumps(df.to_json(orient='records', date_format='iso'))
    # res = '{}'.format(df)
    
    dash.callback_context.record_timing('filter_pd_json', timer() - filter_pd_json_1, 'filter_pd_json')
    return res


@app.callback(
    Output('filter_json-data-table', 'data'),
    Input('fsym-id-dropdown', 'value'),
    State('data-table', 'data'))
def filter_json(selected, jsonified_data):
    filter_lst_dict_1 = timer()
    data = json.loads(jsonified_data)
    # print('filter_json')
    # print(jsonified_data)
    lst = [row for row in data['data'] if row[0] == selected]
    res = '{}'.format(lst)

    dash.callback_context.record_timing('filter_json', timer() - filter_lst_dict_1, 'filter_json')
    return res


if __name__ == "__main__":
    # View the app at http://192.168.2.77:8080/ or, in general,
    #   http://[host computer's IP address]:8080/
    

    app.config.suppress_callback_exceptions = True
    # app.run_server(debug=False, host='0.0.0.0', port=8080, use_reloader=False)
    app.run_server(debug=True, port=80, dev_tools_silence_routes_logging = True)
          
