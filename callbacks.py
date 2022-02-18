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
    
    debug_mode = False
    def print_callback(debug_mode: bool) -> None:
        """
        Function that prints callback to help debugging

        Parameters
        ----------
        debug_mode : Bool
            True for debugging mode.

        Returns
        -------
        None

        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if debug_mode:
                    print(f"==== Function called: {func.__name__}")
                    print(f"Triggered by {callback_context.triggered[0]['prop_id']}")
                result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator
    
    def compare_new_data_with_factset(secid: str, update_date, bbg,
                                      factset, check_exist) -> pd.DataFrame:
        if check_exist:
            factset = factset[factset['exdate'] <= update_date] 
            ##TODO inorporate into laod_data step
        factset = factset.copy()
        bbg['exdate'] = pd.to_datetime(bbg['exdate'], format='%Y-%m-%d')
        factset['exdate'] = pd.to_datetime(factset['exdate'], format='%Y-%m-%d')
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

# =============================================================================
#   Uploader
# =============================================================================
    
    @app.callback(
      Output('path-warning-msg1', 'children'),
      Output('path-warning-msg2', 'children'),
      Output('div-data-path', 'valid'),
      Output('seclist-path', 'valid'),
      Output('path-warning-msg1', 'is_open'),
      Output('path-warning-msg2', 'is_open'),
      Output('div-data-path', 'placeholder'),
      Output('seclist-path', 'placeholder'),
      Input('div-date-picker', 'date'), 
      Input('div-data-path', 'value'),
      Input('seclist-path', 'value'),
      )
    def check_path_validity(update_date, new_data_path=None, 
                            update_list_path=None):
        if update_date is None: return no_update
        import_warning_msg_1 = ''
        import_warning_msg_2 = ''       
        
        div_path_valid = False
        seclist_path_valid = False
        if new_data_path:
            try:
                new_data = pd.read_parquet(new_data_path)
                div_path_valid = True
            except Exception as e:
                import_warning_msg_1 = f"""Error for new dvd path: {e}.
                                            Using default path."""
        if update_list_path:
            try:
                update_list = pd.read_csv(update_list_path)
                seclist_path_valid = True
            except Exception as e:
                import_warning_msg_2 = f"""Error for seclist path: {e}.
                                            Using default path."""
        
        f_date = update_date.replace('-','')
        new_data_default_path = f"""\\bgndc\Analysts\Scheduled_Jobs\output\\new_dvd_data_{f_date}.parquet"""
        update_list_default_path = f"""\\bgndc\Analysts\Scheduled_Jobs\input\sec_list_{f_date}.csv"""
        return import_warning_msg_1, import_warning_msg_2,\
            div_path_valid, seclist_path_valid,\
            not div_path_valid and new_data_path,\
            not seclist_path_valid and update_list_path,\
            new_data_default_path, update_list_default_path
                
    def last_day_of_month(date: datetime) -> datetime:
        """
        Get the last day of the month of a given date

        Parameters
        ----------
        date : datetime
            A given date that may or may not be the last day of the month.

        Returns
        -------
        datetime
            The last day of the month.

        """
        if date.month == 12:
            return date.replace(day=31)
        return date.replace(month=date.month+1, day=1) - timedelta(days=1)

    @app.callback(
        Output('data-table', 'data'),
        Output('view-type-list', 'data'),
        Output('bg-div-data-table', 'data'),
        Output('split-db-data-table', 'data'),
        Output('basic-info-data-table', 'data'),
        Output('factset-data-table', 'data'),
        Output('skipped-data-table', 'data'),
        Output('no-file-warning-msg', 'children'),
        Output('no-file-warning-msg', 'is_open'),
        Input('div-date-picker', 'date'), 
        Input('index-only-radio', 'value'),
        Input('submit-path-button', 'n_clicks'),
        State('div-data-path', 'value'),
        State('seclist-path', 'value'),
        State('div-data-path', 'valid'),
        State('seclist-path', 'valid'),
    )
    # @functools.lru_cache(maxsize=5)
    def load_data_to_dash(update_date, index_flag, path_btn_clicks,
                          new_data_path, update_list_path,
                          div_path_valid, seclist_path_valid
                          ):
        if update_date is None: return no_update
        update_date = last_day_of_month(datetime.strptime(update_date, '%Y-%m-%d'))
        f_date = update_date.strftime('%Y-%m-%d').replace('-','')
        new_data_path = rf"""\\bgndc\Analysts\Scheduled_Jobs\output\new_dvd_data_{f_date}.parquet"""
        update_list_path = rf"""\\bgndc\Analysts\Scheduled_Jobs\input\sec_list_{f_date}.csv"""
        
        # Invalid default path
        try:
            new_data = pd.read_parquet(new_data_path)
            update_list = pd.read_csv(update_list_path)
        except Exception as e:
            display_date = update_date.strftime('%Y-%m-%d')
            import_warning_msg = f"""The file may not exist for {display_date}. 
                                    Error for seclist path: {e}."""
            return no_update, no_update, no_update, no_update, no_update,\
                no_update, no_update, import_warning_msg, True
        
        # Use user submitted path
        if path_btn_clicks  > 0: 
            if not div_path_valid and not seclist_path_valid:
                return no_update
            if div_path_valid:
                new_data = pd.read_parquet(new_data_path)
            if seclist_path_valid:
                update_list = pd.read_csv(update_list_path)

        (new_data, skipped, pro_rata) = \
            preprocess_bbg_data(new_data, update_list, index_flag)
        
        new_data = bbg_data_single_security(new_data)
    
        skipped = process_skipped(skipped)
        
        # Load data
        seclist = list(new_data['fsym_id'].unique()) +\
            list(skipped['fsym_id'].unique())
    
        bg_div_data = load_bg_div_data(seclist)
        split_data = load_split_data(seclist)
        split_data['p_split_date'] = pd.DatetimeIndex(
            split_data['p_split_date']).strftime("%Y-%m-%d")
        basic_info_data = load_basic_info_data(seclist)
        factset_data = load_factset_data(seclist)
        
        if factset_data.shape[0] != 0:
            factset_data = factset_data_single_security(factset_data)
            
        (manual_list, all_goods) = bulk_upload(new_data, update_date, factset_data)
        if all_goods is not None:
            all_goods_lst = sorted(all_goods['fsym_id'].to_list())
        else: all_goods_lst = []
        
        if manual_list is not None:
            mismatch_lst = sorted(manual_list) 
        else: mismatch_lst = []
        
        if skipped is not None:
            skipped_lst = sorted(skipped['fsym_id'].unique())
        else: skipped_lst = []
        # fysm_id_lst = [*all_goods_lst, *mismatch_lst, *skipped_lst]

        view_type_ids = pd.DataFrame({
            'all_goods': pd.Series(all_goods_lst),
            'mismatch': pd.Series(mismatch_lst),   
            'skipped': pd.Series(skipped_lst)})
        new_data.to_csv('new_data_processed')
        skipped.to_csv('skipped_processed')
        return new_data.to_dict('records'),\
                view_type_ids.to_dict('records'),\
                bg_div_data.to_dict('records'),\
                split_data.to_dict('records'), basic_info_data.to_dict('records'),\
                    factset_data.to_dict('records'), skipped.to_dict('records'),\
                    no_update, no_update
    
    @app.callback(
        Output('new-data-data-table', 'data'),
        Output('fsym-id-dropdown', 'options'),
        Output('no-data-msg', 'children'),
        Output('no-data-msg', 'is_open'),
        Output('main-panel', 'style'),
        Output('modified-data-rows', 'columns'),
        Output('next-button', 'n_clicks'),
        Output('prev-button', 'n_clicks'),
        Output('no-overall-data-msg', 'is_open'),
        Input('view-type-radio', 'value'),
        Input('data-table', 'data'),
        State('view-type-list', 'data'),
        State('skipped-data-table', 'data'))
    def load_selected_data(selected_review_option, datatable, view_type_data, skipped):
        # df_selection = pd.DataFrame(view_type_data)
        display_option = {'display': 'none'}
        if selected_review_option != 'skipped':
            df = pd.DataFrame(datatable)
        else:
            df = pd.DataFrame(skipped)
        if view_type_data is None:
            return no_update, no_update, no_update, no_update, display_option,\
                no_update, no_update, no_update, True
            
        selected_ids = [row[selected_review_option] for row in view_type_data]

        selected_ids = sorted(list(filter(None, selected_ids)))
        has_data = len(selected_ids) > 0
        if has_data:
            no_data_msg = ''
            fsym_id_dropdown_options = [{"label": i, "value": i} 
                                        for i in selected_ids]
            selected_data = df[df['fsym_id'].isin(selected_ids)]
            display_option = {}
        else:
            fsym_id_dropdown_options = []
            selected_data = pd.DataFrame([])
            no_data_msg = f'There is no entry to be reviewed for {selected_review_option}'

        return selected_data.to_dict('records'),\
           fsym_id_dropdown_options,\
           no_data_msg, not has_data, display_option,\
            [{'name': 'action', 'id':'action'}] + [{'name': i, 'id':i} 
                                                   for i in df.columns],\
                0, 0, False
        
    @app.callback(
        Output('fsym-id-data-table', 'data'),
        Output('fsym-id-data-table', 'columns'),
        Output('div-selected-data-table', 'data'),
        Output('split-selected-data-table', 'data'),
        Output('split-selected-data-table', 'columns'),
        Output('split-warning-msg', 'is_open'),
        Output('split-warning-msg', 'children'),
        Output('split-content', 'style'),
        Input('fsym-id-dropdown', 'value'),
        State('new-data-data-table', 'data'),
        State('bg-div-data-table', 'data'),
        State('split-db-data-table', 'data'))
    def filter_fysm_id_data(selected, datatable, div_datatable, split_datatable):
        if datatable is None: return no_update
        new_data_filtered = [row for row in datatable if row['fsym_id'] == selected]
        
        for col in ['declared_date' , 'exdate', 'payment_date', 'record_date']:
            for row in new_data_filtered:
                row[col] = pd.to_datetime(row[col], format='%Y-%m-%d')\
                    .strftime("%Y-%m-%d") if row[col] else row[col]

        new_data_col = [{'name': i, 'id':i} for i in datatable[0].keys()]\
            if len(datatable) else []
        div_selected = [row for row in div_datatable 
                        if row['fsym_id'] == selected]\
            if len(div_datatable) else []
        split_selected = [row for row in split_datatable 
                          if row['fsym_id'] == selected]\
            if len(split_datatable) else []
        split_cols = [{'name': i, 'id':i} for i in split_datatable[0].keys()]\
            if len(split_datatable) else []

        return new_data_filtered, new_data_col,\
            div_selected, split_selected,\
        split_cols, not len(split_selected),\
            f'There is no split data for {selected}',\
                {} if len(split_selected) else  {'display': 'none'}
            
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
        if not new_data: return no_update
        new_data = pd.DataFrame(new_data)

        fsym_id = new_data['fsym_id'].values[0]
        bg_div_df = pd.DataFrame(bg_div_data)
        info_df = pd.DataFrame(basic_info_datatable)

        info_df = info_df[info_df['fsym_id']==fsym_id]
        factset_df = pd.DataFrame(factset_datatable)
        if factset_df.shape[0] != 0:
            factset_df = factset_df[factset_df['fsym_id']==fsym_id]

        check_exist = bg_div_df.shape[0] != 0
        basic_info_str = basic_info(fsym_id, info_df, new_data)
        payment_exist_msg = ''
        if not check_exist:
            comparison_msg = "This is a newly added."
            comparison = compare_new_data_with_factset(fsym_id, update_date,
                                                       new_data,  factset_df,
                                                       check_exist)
            fig = plot_dividend_data_comparison(factset_df, new_data)
        else:
            (last_cur, fstest_cur) = dividend_currency(new_data)
            comparison = compare_new_data_with_factset(fsym_id, update_date, 
                                                       new_data, factset_df, 
                                                       check_exist)

            new_data['listing_currency'] = fstest_cur
            new_data['payment_currency'] = last_cur
            comparison_msg = 'New Dividend Data'
            fig, payment_exist_msg = plot_dividend_data(fsym_id, new_data,
                                                        bg_div_df)
            if payment_exist_msg != '': comparison_msg = payment_exist_msg
            has_mismatch = not (comparison[['check_amount', 
                                            'check_payment_date']]=='Good')\
                .all().all()
            if has_mismatch:
                comparison_msg = f'Mismatch found for {fsym_id}.'
            if fstest_cur != last_cur:
                comparison_msg = f"""Possible currency change. 
                                Last payment: {last_cur}. 
                                Factset payment: {fstest_cur}"""

        for col in ['exdate' , 'payment_date_bbg', 'payment_date_factset']:
            comparison[col] = pd.DatetimeIndex(comparison[col])\
                                            .strftime("%Y-%m-%d")
        return basic_info_str, comparison.to_dict('records'),\
            [{'name': i, 'id':i} for i in comparison.columns],\
            fig, comparison_msg
    
    @app.callback(
        Output('factset-graph', 'figure'),
        Output('facset-content', 'style'),
        Output('factset-warning-msg', 'is_open'),
        Output('factset-warning-msg', 'children'),
        Input('fsym-id-data-table', 'data'),
        Input('div-selected-data-table', 'data'))
    def plot_comparison(new_data, div_selected_datatable):
        bbg = pd.DataFrame(new_data)
        bg = pd.DataFrame(div_selected_datatable)
        if len(bg) == 0:
            has_data = False
            warning_msg = """No need to compare since this holding
                            is not in the bg_div table."""
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
        if not new_data: return no_update
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
    def plot_db(div_selected_datatable: List[Dict]):     
        """
        Plot graph for dividend data from the database.

        Parameters
        ----------
        div_selected_datatable : List[Dict]
            Dividend datatable for the selected fsym_id.

        """     
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
            df['exdate'] = pd.DatetimeIndex(df['exdate']).strftime("%Y-%m-%d")
            df.sort_values(by=['exdate'], ascending=False, inplace=True)
            fig = plot_generic_dividend_data(df)
            cols = [{'name': i, 'id':i} for i in df.columns]
            df = df.to_dict('records')        
        return fig, df, cols, {'display': 'none'} if not has_data else {},\
            not has_data, warning_msg
    
    @app.callback(
        Output('output-data-table', 'data'),
        Input('editor-fsym-id-dropdown', 'value'))
    @print_callback(debug_mode)
    def filter_fysm_id_editor(selected: str) -> List[Dict]:
        print('filter_fysm_id_editor')
        print(modify_data)
        if modify_data.shape[0] == 0: return no_update
        output_df = modify_data.copy()
        for col in ['declared_date' , 'exdate', 'payment_date', 'record_date']:
            output_df[col] = pd.DatetimeIndex(output_df[col]).strftime("%Y-%m-%d")
        print(modify_data)

        if selected == 'All':
            return output_df.to_dict('records')
        return output_df[output_df['fsym_id'] == selected].to_dict('records')
     
    @app.callback(
        Output('modify-button', 'disabled'),
        Input('modify-switch', 'value'))
    def show_collapse_button(is_switch_on: bool) -> bool:
        if is_switch_on:
            return True
        return False
    
    # @app.callback(
    #     Output('modify-switch', 'value'),
    #     Input('view-type-button', 'value'))
    # def clear_modify_button_after_switch_view_type(view_type_button):
    #     return False

    @app.callback(
            Output('modify-list', 'data'),
            Output('add-to-editor-msg', 'is_open'),
            Input('modify-button', 'n_clicks'),
            Input('view-type-radio', 'value'),
            State('fsym-id-dropdown', 'value'),
            State('modify-list', 'data'))
    @print_callback(debug_mode)
    def update_modify_list(n_clicks: int, view_type_radio: str,
                           cur_fsym_id: str,
                           modify_list: List) -> Tuple[List[str], bool]:
        
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]    
        if changed_id == 'view-type-radio.value':
            return [], False
        if not n_clicks:
            return no_update
        modify_list.append({'name': cur_fsym_id, 'id': cur_fsym_id})
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
    def go_to_next_prev(prev_clicks: int, next_clicks: int, cur_fsym_id: str,
                        view_type: str, view_type_lst: List):
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]    
        lst = [row[view_type] for row in view_type_lst]
        lst = sorted(list(filter(None, lst)))

        # For giving a default value
        if not cur_fsym_id:
            return lst[0], '', False
        if not prev_clicks and not next_clicks:
            return no_update
        if not cur_fsym_id in lst:
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
        if not data: return []
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
        Output('editor-fsym-id-dropdown', 'options'),
        Output('editor-fsym-id-dropdown', 'value'),
        Input('modify-list', 'data'),
        Input('modify-switch', 'value'),
        State('new-data-data-table', 'data'),
        )
    @print_callback(debug_mode)
    def get_editor_data(modify_lst, is_switch_on, datatable):
        print('get_editor_data')
        if not modify_lst: return no_update
        df = pd.DataFrame(datatable)
        lst = list(set([row['name'] for row in modify_lst]))
        lst.sort()
        global modify_data 
        modify_data = df[df['fsym_id'].isin(lst)]
        fsym_ids = [{"label": i, "value": i} for i in lst]
        print(lst)
        print(modify_data)
        if not is_switch_on: return no_update, no_update
        return fsym_ids , fsym_ids[0]['value']
 
    @app.callback(
        Output('editor-open-msg', 'is_open'),
        Output('editor-main', 'style'),
        Input('modify-switch', 'value'),
        )   
    def show_editor_info_msg(is_switch_on):
        if not is_switch_on:
            return True, {'display': 'none'}
        return False, {}
        
    @app.callback(
        Output('edit-data-table', 'data'),
        Input('modified-data-rows', 'data'),
        State('modified-data-rows', 'data_previous'),
        State('editor-fsym-id-dropdown', 'value'),
        State('output-data-table', 'data'))
    @print_callback(debug_mode)
    def update_changed_data_table(rows: List[Dict], rows_prev: List[Dict],
                                  fsym_id: str, 
                                  modified_datatable: List[Dict]) -> List[Dict]:
        print('update_changed_data_table')
        print(rows)
        if not len(rows): return no_update
        global modify_data
        print(modify_data)
        # if modified_datatable:
        modified_df = pd.DataFrame(modified_datatable)
        print(modified_df)
        modify_data = modify_data[~(modify_data['fsym_id'] == fsym_id)]
        modify_data = pd.concat([modify_data, modified_df])
    
        res = modify_data.to_dict('records')
        
        undo_delete_row = [row for row in rows_prev if row not in rows] if \
            (rows is not None and rows_prev is not None) and \
                len(rows_prev) > len(rows) else []
        # res  = res +  undo_delete_row if undo_delete_row is not None else res
        # pd.DataFrame(undo_delete_row).drop(columns=['action'], inplace=True)
        modify_data = pd.concat([modify_data, pd.DataFrame(undo_delete_row)])
        print(modify_data)
        return res
    
    @app.callback(
        Output('modified-data-rows', 'data'),
        Input('modify-switch', 'value'),
        Input('output-data-table', 'data_previous'),
        State('output-data-table', 'data'),
        State('modified-data-rows', 'data'))
    @print_callback(debug_mode)
    def update_modified_data_table(is_switch_on, rows_prev, rows, modified_rows):
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'modify-switch.value' == changed_id:
            return []
        if rows == None and rows_prev == None:
            return no_update
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
            df = df[~(df['fsym_id'].isin(lst))]
            datatypes = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
            df = df.astype(datatypes)
            # for col in ['declared_date', 'exdate', 'payment_date', 'record_date']:
            #     df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
            df.sort_values(by=['fsym_id', 'exdate'], ascending=False)
            df.reset_index(inplace=True, drop=True)
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
            for col in ['declared_date', 'exdate', 'payment_date', 'record_date']:
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%dT%H:%M:%S')
            df.sort_values(by=['fsym_id', 'exdate'], ascending=False)
            df.reset_index(inplace=True, drop=True)
            df.to_csv('edited_div_data')
            return 'Data saved to DB', True