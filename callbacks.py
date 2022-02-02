# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:41:16 2022

@author: Chang.Liu
"""
from dash import callback_context, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
# from dash_extensions.enrich import Trigger, FileSystemCache

import pandas as pd
import numpy as np

from typing import List, Tuple

from bg_data_uploader import *


def register_callbacks(app, long_callback_manager, data_importer_dash) -> None:
    """
    Registers the callbacks to the app
    """

    def compare_new_data_with_factset(secid, update_date, bbg, factset):
        # if factset is None:
        #     factset = factset_new_data(secid, last_exdate)
        if check_exist:
            factset = factset[factset['exdate'] <= update_date]##TODO inorporate into laod_data step
        # print('compare_new_data_with_factset')
        # print(factset)
        # print(bbg)
        bbg['exdate'] = pd.to_datetime(bbg['exdate'], format='%Y-%m-%d')
        new_data_comparison = pd.merge(bbg, factset, how='outer', 
                                       on=['fsym_id','exdate','div_type'],
                                       suffixes=('_bbg','_factset'))
        new_data_comparison = new_data_comparison.sort_values(['exdate'])
        new_data_comparison = new_data_comparison.reset_index(drop=True)
        new_data_comparison = new_data_comparison[
            new_data_comparison.filter(
                regex='fsym_id|exdate|payment_date|amount'
                ).columns]
        new_data_comparison['check_amount'] = np.where(
            abs(
                new_data_comparison['payment_amount_factset']-\
                    new_data_comparison['payment_amount_bbg'])>0.001,
                'Mismatch', 'Good')
        new_data_comparison['check_payment_date'] = np.where(
            new_data_comparison['payment_date_factset']!=\
                new_data_comparison['payment_date_bbg'],
            'Mismatch', 'Good')
        return new_data_comparison

    @app.callback(
        Output("collapse-editor", "is_open"),
        # Output('edit-data-table', 'data'),
        Output('editor-fsym-id-dropdown', 'options'),
        Output('editor-fsym-id-dropdown', 'value'),
        Input("collapse-button", "n_clicks"),
        Input('modify-list', 'data'),
        State("collapse-editor", "is_open"),
        State('new-data-data-table', 'data'),
        # running=[
        #     (Output("collapse-button", "disabled"), True, False),
        # ],
        # manager=long_callback_manager,
        )
    def get_editor_data(n, modify_lst, is_open, datatable):
        df = pd.DataFrame(datatable)
        lst = [row['name'] for row in modify_lst]
        global modify_data 
        modify_data = df[df['fsym_id'].isin(lst)]
        fsym_ids = [{"label": i, "value": i} for i in modify_data['fsym_id']]
        # fsym_ids = [{"label": 'All', "value": 'All'}] + \
        #[{"label": i, "value": i} for i in modify_data['fsym_id']]
    
        if n:
            return not is_open, fsym_ids , fsym_ids[0]['value']
        return is_open, fsym_ids , fsym_ids[0]['value']
    
    @app.callback(
            Output('modify-list', 'data'),
            Output('add-to-editor-msg', 'is_open'),
            Input('modify-button', 'n_clicks'),
            State('fsym-id-dropdown', 'value'),
            State('modify-list', 'data'))
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
            State('fsym-id-dropdown', 'value'),
            State('view-type-radio', 'value'),
            State('view-type-list', 'data'))
    def go_to_next_prev(prev_clicks, next_clicks, cur_fsym_id, view_type, view_type_lst):
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]    
        lst = [row[view_type] for row in view_type_lst]
        if lst[-1] is None:
            lst = lst[:-1]
        lst.sort()
        if not prev_clicks and not next_clicks:
            return lst[0], '', False
        if prev_clicks or next_clicks:
            idx = lst.index(cur_fsym_id)-1 \
                if 'prev-button.n_clicks' == changed_id \
                else lst.index(cur_fsym_id)+1
            beg_end_msg = 'first' if idx < 0 else 'last'
            if idx >= len(lst) or idx < 0:
                end_dropdown_msg = f'This is the {beg_end_msg} Fsym Id.'
                return no_update, end_dropdown_msg, True
            return lst[idx], '', False
    
    @app.callback(
        Output('modified-data-rows', 'style_data_conditional'),
        Input('modified-data-rows', 'data'))
    def highlight_changed_cell(data):
        df = pd.DataFrame(data)
        lst_idx, lst_changed_cell = find_changed_cell(df)
        return [
            {
                'if': {
                    'filter_query': '{{id}} = {}'.format(i),
                    # 'row_index': i, # More efficient but not for multipage df
                    'column_id': col
                },
                'backgroundColor': 'DodgerBlue',
                'color': 'white'
            }
            for i, col in zip(lst_idx, lst_changed_cell)
        ]

    def find_changed_cell(df: pd.DataFrame) -> Tuple[List, List]:
        """
        Find the cell that the user modified

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of the modified history

        Returns
        -------
        Tuple[List, List]
            A list of the index and column name for the changed cell.

        """
        df = df[~(df['action'] =='delete')]
        cols = df.columns[1:12].tolist()
        lst_changed_cell = []
        lst_idx = []
        for idx, row in df.iterrows():
            if row['action'] == 'update':
                next_row = df.loc[idx+1]
    
                for col in cols:
                    both_nan = (row[col] != row[col]) and (next_row[col] != next_row[col])
                    if (row[col] != next_row[col]) and not both_nan:
                        lst_idx = lst_idx + [idx, idx+1]
                        lst_changed_cell = lst_changed_cell + [col, col]
        return lst_idx, lst_changed_cell   
  
    
  
    
    @app.callback(
      Output('path-warning-msg1', 'children'),
      Output('path-warning-msg2', 'children'),
      Output('div-data-path', 'valid'),
      Output('seclist-path', 'valid'),
      Output('path-warning-msg1', 'is_open'),
      Output('path-warning-msg2', 'is_open'),
      Input('div-data-path', 'value'),
      Input('seclist-path', 'value'),
      )
    def check_path_validity(new_data_path=None, update_list_path=None):
        import_warning_msg_1 = ''
        import_warning_msg_2 = ''       
        
        div_path_valid = False
        seclist_path_valid = False
        if new_data_path:
            try:
                new_data = pd.read_parquet(new_data_path)
                div_path_valid = True
            except Exception as e:
                import_warning_msg_1 = f'Error for new dvd path: {e}. \nUsing default path.'
        if update_list_path:
            try:
                update_list = pd.read_csv(update_list_path)
                seclist_path_valid = True
            except Exception as e:
                import_warning_msg_2 = f'Error for seclist path: {e}. \nUsing default path.'
                
        return import_warning_msg_1, import_warning_msg_2,\
            div_path_valid, seclist_path_valid,\
                not div_path_valid and new_data_path, not seclist_path_valid and update_list_path
                
    # @app.callback(
    #   Output('path-warning-msg', 'children'),
    #   Output('path-warning-msg', 'valid'),
    #   Input('div-data-path', 'value'),
    #   Input('seclist-path', 'value'),
    #   )
    # def submit_path(new_data_path=None, update_list_path=None): 
    #     if nclicks == 0:
    #         raise PreventUpdate
        
            
        
    
    # fsc = FileSystemCache("cache_dir")
    # fsc.set("progress", None)
        
       

    # @app.callback(Output("progress", "children"), Trigger("interval", "n_intervals"))
    # def update_progress():
    #     value = fsc.get("progress")  # get progress
    #     if value is None:
    #         raise PreventUpdate
    #     return "Progress is {:.0f}%".format(float(fsc.get("progress")) * 100)

    @app.callback(
        Output('data-table', 'data'),
        # Output('data-table', 'columns'),
        Output('view-type-list', 'data'),
        # Output('progress-div', 'style'),
        Output('bg-div-data-table', 'data'),
        # Output('bg-div-data-table', 'columns'),
        Output('split-db-data-table', 'data'),
        # Output('split-db-data-table', 'columns'),
        Output('basic-info-data-table', 'data'),
        Output('factset-data-table', 'data'),
        # Output('factset-data-table', 'valid'),
        Input('div-date-picker', 'date'), 
        Input('index-only-radio', 'value'),
        Input('submit-path-button', 'n_clicks'),
        State('div-data-path', 'value'),
        State('seclist-path', 'value'),
        State('div-data-path', 'valid'),
        State('seclist-path', 'valid'),
        # running=[
        #     (Output("div-date-picker", "disabled"), True, False),
        #     (Output("collapse-button-div", "style"), {'display': 'none'}, {}),
        #     (Output("main-panel-div", "style"), {'display': 'none'}, {}),
        #     (Output("view-type-div", "style"), {'display': 'none'}, {}),
        # ],
        # manager=long_callback_manager,
        # progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
    )
    # @functools.lru_cache(maxsize=5)
    def load_data_to_dash(update_date, index_flag, path_btn_clicks,
                          new_data_path, update_list_path,
                          div_path_valid, seclist_path_valid
                          ):
        # monthend_date = (datetime.today() + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        # if update_date == "":
        #     update_date = monthend_date
        print(f'update_date:{update_date}')
        f_date = update_date.replace('-','')
        
        new_data = pd.read_parquet(rf'\\bgndc\Analysts\Scheduled_Jobs\output\new_dvd_data_{f_date}.parquet')
        update_list = pd.read_csv(rf'\\bgndc\Analysts\Scheduled_Jobs\input\sec_list_{f_date}.csv')
        if path_btn_clicks  > 0: 
            if not div_path_valid and not seclist_path_valid:
                return dash.no_update
            if div_path_valid:
                new_data = pd.read_parquet(new_data_path)
            if seclist_path_valid:
                update_list = pd.read_csv(update_list_path)
                
        print('load_data_to_dash')

        # i=0
        (new_data, skipped, pro_rata, splits) = \
            preprocess_bbg_data(new_data, update_list, index_flag, update_date)
        
        # set_progress((str(i + 1), str(total)))
        new_data = bbg_data_single_security(new_data)
        # set_progress((str(i + 1), str(total)))
    
        skipped = process_skipped(skipped)
        # set_progress((str(i + 1), str(total)))
        
        seclist = sorted(list(new_data['fsym_id'].unique()))
        # total=8
        bg_div_data = load_bg_div_data(seclist)
        # set_progress((str(i + 1), str(total)))
        split_data = load_split_data(seclist)
        # set_progress((str(i + 1), str(total)))

        basic_info_data = load_basic_info_data(seclist)
        # set_progress((str(i + 1), str(total)))
        
        factset_data = load_factset_data(seclist)

        factset_data = factset_data_single_security(factset_data)

        # set_progress((str(i + 1), str(total)))

        (manual_list, all_goods) = bulk_upload(new_data, update_date, factset_data)
        print('bulk_upload')
        # set_progress((str(i + 1), str(total)))
     
        # print(new_data.dtypes)
        # print('Loaded')
        # print(all_goods)
        # print(manual_list)
        # print(skipped)
        
        all_goods_lst = sorted(all_goods['fsym_id'].to_list()) if all_goods is not None else []
        mismatch_lst = sorted(manual_list) if manual_list is not None else []
        skipped_lst = skipped['fsym_id'].unique() if skipped is not None else []
        # print(all_goods_lst)
        # print(mismatch_lst)
        # print(skipped_lst)
      
        # fysm_id_lst = [*all_goods_lst, *mismatch_lst, *skipped_lst]
        # print(fysm_id_lst)
        print('after lst')

        # print(factset_data)
        view_type_ids = pd.DataFrame({
            'all_goods': pd.Series(all_goods_lst),
            'mismatch': pd.Series(mismatch_lst),   
            'skipped': pd.Series(skipped_lst)})
        print('after view_type_ids')
           
        # print(dropdown_options)
        return new_data.to_dict('records'),\
                view_type_ids.to_dict('records'),\
                bg_div_data.to_dict('records'),\
                split_data.to_dict('records'), basic_info_data.to_dict('records'), factset_data.to_dict('records')
                # [{'name': i, 'id':i} for i in split_data.columns]
    
    @app.callback(
        Output('new-data-data-table', 'data'),
        Output('new-data-data-table', 'columns'),
        Output('fsym-id-dropdown', 'options'),
        Output('no-data-msg', 'children'),
        Output('no-data-msg', 'is_open'),
        Output('main-panel', 'style'),
        # Output('modified-data-rows', 'columns'),
        Input('view-type-radio', 'value'),
        Input('data-table', 'data'),
        Input('view-type-list', 'data'))
    def load_selected_data(selected_review_option, datatable, view_type_lst):
        df = pd.DataFrame(datatable)
        df_selection = pd.DataFrame(view_type_lst)
        # print('load_selected_data')
        # print(df_selection)
        selected_ids = df_selection[selected_review_option].unique()    
        if selected_ids[-1] is None:
            selected_ids = selected_ids[:-1] 
        has_data = selected_ids.size > 0
        fsym_id_dropdown_options = [{"label": i, "value": i} \
                                    for i in selected_ids] if has_data else []
    
        selected_data = df[df['fsym_id'].isin(selected_ids)] if has_data else pd.DataFrame([])
        no_data_msg = f'There is no entry to be reviewed for {selected_review_option}'\
            if not has_data else 'Data loaded'
        display_option = {'display': 'none'} if not has_data else {}
   
        return selected_data.to_dict('records'),\
            [{'name': i, 'id':i} for i in selected_data.columns],\
           fsym_id_dropdown_options,\
           no_data_msg, not has_data, display_option,\
           # [{'name': 'action', 'id':'action'}] + [{'name': i, 'id':i} for i in df.columns]
    
    
    @app.callback(
        Output('fsym-id-data-table', 'data'),
        Output('fsym-id-data-table', 'columns'),
        Output('div-selected-data-table', 'data'),
        Output('split-selected-data-table', 'data'),
        Output('split-selected-data-table', 'columns'),
        # Output('no-', 'is_open'),        
        Output('split-warning-msg', 'is_open'),
        Output('split-warning-msg', 'children'),
        Output('split-content', 'style'),
        Input('fsym-id-dropdown', 'value'),
        State('new-data-data-table', 'data'),
        State('bg-div-data-table', 'data'),
        State('split-db-data-table', 'data'))
    def filter_fysm_id_data(selected, datatable, div_datatable, split_datatable):
        if datatable is None: return no_update
        df = pd.DataFrame(datatable)
        bg_div_df = pd.DataFrame(div_datatable)
        split_df = pd.DataFrame(split_datatable)
        # print('filter_fysm_id_data')
        # print(bg_div_df)
        # print(split_df)
        res = df[df['fsym_id'] == selected]
        
        div_selected = bg_div_df[bg_div_df['fsym_id']==selected] if bg_div_df.shape[0] != 0 else bg_div_df

        if split_df.shape[0] != 0:
            split_df = split_df[split_df['fsym_id']==selected]

        return res.to_dict('records'), [{'name': i, 'id':i} for i in res.columns],\
            div_selected.to_dict('records'), split_df.to_dict('records'),\
            [{'name': i, 'id':i} for i in split_df.columns], split_df.shape[0] == 0,\
            f'There is no split data for {selected}',\
                {} if split_df.shape[0] != 0 else  {'display': 'none'}
            
    @app.callback(
        Output('basic-info', 'children'),
        Output('comparison-data-table', 'data'),
        Output('comparison-data-table', 'columns'),
        Output('fsym-id-graph', 'figure'),
        Output('comparison-msg', 'children'),
        Input('fsym-id-data-table', 'data'),
        State('div-selected-data-table', 'data'),
        State('div-date-picker', 'date'),
        State('basic-info-data-table', 'data'),
        State('factset-data-table', 'data'),
        )
    def get_basic_info(new_data, bg_div_data, update_date, 
                       basic_info_datatable, factset_datatable):
        # print('get_basic_info')
        new_data = pd.DataFrame(new_data)
        # print(new_data)

        fsym_id = new_data['fsym_id'].values[0]
        bg_div_df = pd.DataFrame(bg_div_data)
        info_df = pd.DataFrame(basic_info_datatable)
        # print(info_df)

        info_df = info_df[info_df['fsym_id']==fsym_id]
        factset_df = pd.DataFrame(factset_datatable)
        # print(factset_df)
        if factset_df.shape[0] != 0:
            factset_df = factset_df[factset_df['fsym_id']==fsym_id]
        # print(new_data.dtypes)
        global check_exist
        check_exist = bg_div_df.shape[0] != 0
        basic_info_str = basic_info(fsym_id, info_df, new_data)
        payment_exist_msg = ''
        if not check_exist:
            # last_exdate = None
            comparison_msg = "This is a newly added."
            comparison = compare_new_data_with_factset(fsym_id, update_date,
                                                       new_data,  factset_df)
            fig = plot_dividend_data_comparison(factset_df, new_data)
        else:
            # last_exdate = last_payment_exdate(fsym_id, bg_div_df['exdate'])
            (last_cur, fstest_cur) = dividend_currency(new_data)
            comparison = compare_new_data_with_factset(fsym_id, update_date, new_data, factset_df)
            if fstest_cur != last_cur:
                comparison_msg = f'Possible currency change. Last payment:{last_cur}. Factset payment:{fstest_cur}'
            new_data['listing_currency'] = fstest_cur
            new_data['payment_currency'] = last_cur
            comparison_msg = 'New Dividend Data'
            fig, payment_exist_msg = plot_dividend_data(fsym_id, new_data, bg_div_df)
            if payment_exist_msg != '': comparison_msg = payment_exist_msg 
        return basic_info_str, comparison.to_dict('records'),\
            [{'name': i, 'id':i} for i in comparison.columns],\
                 fig, comparison_msg
    
    @app.callback(
        Output('factset-graph', 'figure'),
        Output('facset-content', 'style'),
        Output('factset-warning-msg', 'is_open'),
        Output('factset-warning-msg', 'children'),
        Input('fsym-id-data-table', 'data'),
        State('div-selected-data-table', 'data'))
    def plot_comparison(new_data, div_selected_datatable):
        bbg = pd.DataFrame(new_data)
        # fsym_id = bbg['fsym_id'].values[0]
        # query = f"select * from fstest.dbo.bg_div where fsym_id ='{fsym_id}'"
    
        # try:
        bg = pd.DataFrame(div_selected_datatable)
        # except Exception as e:#TODO
        #     warning_msg = f'Error while loading data from DB: {e}'
        #     fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'}))
        #     return fig, {'display': 'none'}, True, warning_msg
        if len(bg) == 0:
            has_data = False
            warning_msg = 'No need to compare since this holding is not in the bg_div table.' 
            fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'})) 
            display_option = {'display': 'none'}
        else:
            has_data = True
            warning_msg = '' 
            fig = plot_dividend_data_comparison(bg, bbg)
            display_option = {}
        return fig, display_option, not has_data, warning_msg   
    
    @app.callback(
        Output('bbg-graph', 'figure'),
        Input('fsym-id-data-table', 'data'))
    def plot_bbg(new_data):
        new_data = pd.DataFrame(new_data)
        df = prepare_bbg_data(new_data)
        fig = plot_generic_dividend_data(df)
        return fig
    
    @app.callback(
        Output('bg-db-graph', 'figure'),
        Output('bg-db-data-table', 'data'),
        Output('bg-db-data-table', 'columns'),
        Output('bg-content', 'style'),
        Output('bg-db-warning-msg', 'is_open'),
        Output('bg-db-warning-msg', 'children'),
        Input('div-selected-data-table', 'data'))
    def plot_db(div_selected_datatable):
        # query = f"select * from fstest.dbo.bg_div where fsym_id ='{fsym_id}'"
        # try:
            # df = data_importer_dash.load_data(query)
        # print('plot_db')
        # print(pd.DataFrame(div_selected_datatable))

        # except Exception as e:
        #     warning_msg = f'Error while loading data from DB: {e}'
        #     fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'}))
        #     return fig, {}, [], {'display': 'none'}, warning_msg
        
        if len(div_selected_datatable) == 0:
            has_data = False
            warning_msg = 'This holding is not in the bg_div table.' 
            fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'}))
            cols = []
            df = []
        else:
            has_data = True
            warning_msg = ''
            df = pd.DataFrame(div_selected_datatable)
            df.sort_values(by=['exdate'], ascending=False, inplace=True)
            fig = plot_generic_dividend_data(df)
            cols = [{'name': i, 'id':i} for i in df.columns]
            df = df.to_dict('records')        
        return fig, df, cols, {'display': 'none'} if not has_data else {}, not has_data, warning_msg
    
    # @app.callback(
    #     Output('split-data-table', 'data'),
    #     Output('split-data-table', 'columns'),
    #     # Output('load-data-msg', 'value'),
    #     Input('fsym-id-dropdown', 'value'),
    #     Input('split-data-table', 'data'))
    # def check_split_history(fsym_id, split_data):
    #     # query = f"""select 
    #     #             fsym_id,p_split_date,p_split_factor, 
    #     #             exp(sum(log(p_split_factor))  OVER (ORDER BY p_split_date desc)) cum_split_factor 
    #     #             from fstest.fp_v2.fp_basic_splits where fsym_id= '{fsym_id}'
    #     #             order by p_split_date
    #     #         """
    #     # df = data_importer_dash.load_data(query)
    #     # df.sort_values(by=['p_split_date'], ascending=False, inplace=True)
    #     split_df = pd.DataFrame(split_data)
    #     df = split_df[split_df==fsym_id]
    #     data = df.to_dict('records') if df is not None else []
    #     cols = [{'name': i, 'id':i} for i in df.columns] if df is not None else []
    #     return data, cols
    # @app.callback(
    #     Output('split-data-table', 'data'),
    #     Output('split-data-table', 'columns'),
    #     # Output('load-data-msg', 'value'),
    #     Input('fsym-id-dropdown', 'value'),
    #     Input('split-selected-data-table', 'data'))
    # def check_split_history(fsym_id, split_data):
    #     # query = f"""select 
    #     #             fsym_id,p_split_date,p_split_factor, 
    #     #             exp(sum(log(p_split_factor))  OVER (ORDER BY p_split_date desc)) cum_split_factor 
    #     #             from fstest.fp_v2.fp_basic_splits where fsym_id= '{fsym_id}'
    #     #             order by p_split_date
    #     #         """
    #     # df = data_importer_dash.load_data(query)
    #     # df.sort_values(by=['p_split_date'], ascending=False, inplace=True)
    #     split_df = pd.DataFrame(split_data)
    #     df = split_df[split_df==fsym_id]
    #     data = df.to_dict('records') if df is not None else []
    #     cols = [{'name': i, 'id':i} for i in df.columns] if df is not None else []
    #     return data, cols
    
    @app.callback(
        Output('output-data-table', 'data'),
        Input('editor-fsym-id-dropdown', 'value'))
        # State('edit-data-table', 'data'))
    def filter_fysm_id_editor(selected):
        if selected == 'All':
            return modify_data.to_dict('records')
        return modify_data[modify_data['fsym_id'] == selected].to_dict('records')
    
    @app.callback(
        Output('edit-data-table', 'data'),
        # State('edit-data-table', 'data'),
        Input('modified-data-rows', 'data'),
        State('modified-data-rows', 'data_previous'),
        State('editor-fsym-id-dropdown', 'value'),
        State('output-data-table', 'data'))
    def update_changed_data_table(rows, rows_prev, fsym_id, modified_datatable):
        global modify_data

        modified_df = pd.DataFrame(modified_datatable) 
        # print(f'update_changed_data_table: ____________{i}')
        # print('modified_df: ')
        # print(modified_df)
        modify_data = modify_data[~(modify_data['fsym_id'] == fsym_id)]
        # print(modify_data)
        # df[df['fsym_id'] == fsym_id] = modified_df
        modify_data = pd.concat([modify_data, modified_df])
    
        res = modify_data.to_dict('records')
        undo_delete_row = [row for row in rows_prev if row not in rows] if \
            (rows is not None and rows_prev is not None) and \
                len(rows_prev) > len(rows) else []
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
        Output('modified-data-rows', 'data'),
        Input('output-data-table', 'data_previous'),
        State('output-data-table', 'data'),
        State('modified-data-rows', 'data'))
    def update_modified_data_table(rows_prev, rows, modified_rows):
        if (len(rows) == len(rows_prev)):
            modified_rows = [i for i in rows if i not in rows_prev] \
                if modified_rows is None \
                    else modified_rows + [i for i in rows if i not in rows_prev]
            modified_rows[-1]['action'] = 'update'
            modified_rows= modified_rows + [i for i in rows_prev if i not in rows]
            modified_rows[-1]['action'] = 'original'
        if (len(rows) < len(rows_prev)):
             modified_rows = modified_rows + [i for i in rows_prev if i not in rows]
             modified_rows[-1]['action'] = 'delete'
    
        # Add index for modified data table    
        idx = 0
        for row in modified_rows:
            row.update({'id': idx})
            idx = idx + 1
        return modified_rows

    @app.callback(
            Output("save-msg", "children"),
            Output("save-msg", "is_open"),
            Input("upload-button", "n_clicks"),
            Input('modify-list', 'data'),
            State("new-data-data-table", "data"))
    def export_not_modified_data(nclicks, modify_lst, modified_data): 
        if nclicks == 0:
            raise PreventUpdate
        else:
            df = pd.DataFrame(modified_data)
            lst = [row['name'] for row in modify_lst]
            # print(df)
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