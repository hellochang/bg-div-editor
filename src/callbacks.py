# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:41:16 2022

@author: Chang.Liu
"""
from dash import callback_context, no_update, dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
# from dash_extensions.enrich import Trigger, FileSystemCache

import pandas as pd
import numpy as np

from typing import Optional, List, Tuple, Dict

from bg_data_uploader import *


def register_callbacks(app: dash.Dash) -> None:
    """
    Registers the callbacks to the app    

    Parameters
    ----------
    app : dash.Dash
        Dash app instance.

    Returns
    -------
    None
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
    def check_path_validity(update_date: str, new_data_path: Optional[str]=None, 
                            update_list_path: Optional[str]=None
                            ) -> Tuple[str, str, bool, bool,
                                       str, str, str, str]:
        """
        Checks whether the user input paths are valid and 
        returns path placeholder based on selected date with the datepicker

        Parameters
        ----------
        update_date : str
            Date selected by the Datepicker in step 1.
        new_data_path : Optional[str], optional
            User input path for Bloomberg parquet file. The default is None.
        update_list_path : Optional[str], optional
            User input path for update list csv file. The default is None.

        Returns
        -------
        Tuple[str, str, bool, bool,                                       
              str, str, str, str]
            Warning message if the paths are invalid.
            Paths and default placeholder for the paths
        """
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
    def load_data_to_dash(update_date: str, index_flag: bool, 
                          path_btn_clicks: int,
                          new_data_path: str, update_list_path: str,
                          div_path_valid: bool, seclist_path_valid: bool
                          ) -> Tuple[List[Dict], List[Dict], List[Dict],
                                     List[Dict], List[Dict], List[Dict],
                                     List[Dict], str, bool]:
        """
        Load data to Dash from DB and local files

        Parameters
        ----------
        update_date : str
            The date selected in the datepicker in step 1.
        index_flag : bool
            Whether we're viewing data from indices (SP 500) or not from step 2.
        path_btn_clicks : int
            Submit button for user input path.
        new_data_path : str
            Path for Bloomberg data parquet file.
        update_list_path : str
            Path for update list csv file.
        div_path_valid : bool
            If new_data_path is valid.
        seclist_path_valid : bool
            If update_list_path is valid.

        Returns
        -------
        Tuple[List[Dict], List[Dict], List[Dict],                                     
              List[Dict], List[Dict], List[Dict],                                     
              List[Dict], str, bool]
            Populated data
            Warning message if file path is invalid (hence no data is populated)

        """
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
                
        # Process data
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
    def load_selected_data(selected_review_option: str, datatable: List[Dict],
                           view_type_data: List[Dict], skipped: List[Dict]
                           ) -> Tuple[List[Dict], List[Dict], str, bool, Dict,
                                      List[Dict], int, int, bool]:
        """
        Load data based on whether All Goods, Mismatch or Skipped is selected

        Parameters
        ----------
        selected_review_option : str
            The option user selected. One of All Goods, Mismatch or Skipped.
        datatable : List[Dict]
            Bloomberg data from the parquet file.
        view_type_data : List[Dict]
            List of secids for each of All Goods, Mismatch and Skipped.
        skipped : List[Dict]
            Skipped data.
            
        """
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
    def filter_fsym_id_data(selected: str, datatable: List[Dict], 
                            div_datatable: List[Dict], 
                            split_datatable: List[Dict]
                            ) -> Tuple[List[Dict], List[Dict],
                                       List[Dict], List[Dict],
                                       List[Dict], bool, str, Dict]:
        """
        Filter datatable based on the selected fsym_id

        Parameters
        ----------
        selected : str
            Selected secid from the dropdown.
        datatable : List[Dict]
            Bloomberg datatable
        div_datatable : List[Dict]
            Bg_div datatable .
        split_datatable : List[Dict]
            Split datatable.

        Returns
        -------
        Tuple[List[Dict], List[Dict],                                       
              List[Dict], List[Dict],                                       
              List[Dict], bool, str, Dict]
            Filtered datatable for the selected fsym_id
            Warning message for splits if there's no data
        
        """
        if not datatable: return no_update
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
    def get_basic_and_comparison_info(new_data: List[Dict], 
                                      bg_div_data: List[Dict], 
                                      update_date: str,
                                      basic_info_datatable: List[Dict],
                                      factset_datatable: List[Dict]
                                      ) -> Tuple[str, List[Dict], 
                                                 List[Dict], go.Figure, str]:
        """
        Get basic information and comparison dataframe for the selected fsym_id

        Parameters
        ----------
        new_data : List[Dict]
            Bloomberg data.
        bg_div_data : List[Dict]
            bg_div data from DB.
        update_date : str
            Date selected in datepicker in step 1.
        basic_info_datatable : List[Dict]
            Datatable contain basic information for the current secid dropdown.
        factset_datatable : List[Dict]
            Datatable for factset data for the current secid dropdown.

        Returns
        -------
        Tuple[str, List[Dict],List[Dict], go.Figure, str]
            Basic information string for the current secid
            Comparison dataframe and columns for the current secid
            Graph for the current secid
            Message outlining the issue/info of BBG data for the current secid

        """
        if not new_data: return no_update
        new_data = pd.DataFrame(new_data)

        fsym_id = new_data['fsym_id'].values[0]
        bg_div_df = pd.DataFrame(bg_div_data)
        info_df = pd.DataFrame(basic_info_datatable)
        info_df = info_df[info_df['fsym_id']==fsym_id]
        basic_info_str = basic_info(info_df, new_data)
        factset_df = pd.DataFrame(factset_datatable)
        if factset_df.shape[0] != 0:
            factset_df = factset_df[factset_df['fsym_id']==fsym_id]

        check_exist = bg_div_df.shape[0] != 0
        payment_exist_msg = ''
        if not check_exist:
            comparison_msg = "This is a newly added."
            comparison = compare_new_data_with_factset(update_date,
                                                       new_data,  factset_df,
                                                       check_exist)
            fig = plot_dividend_data_comparison(factset_df, new_data)
        else:
            (last_cur, fstest_cur) = dividend_currency(new_data)
            comparison = compare_new_data_with_factset(update_date, 
                                                       new_data, factset_df, 
                                                       check_exist)

            new_data['listing_currency'] = fstest_cur
            new_data['payment_currency'] = last_cur
            comparison_msg = 'New Dividend Data'
            fig, payment_exist_msg = plot_dividend_data(new_data, bg_div_df)
            has_mismatch = not (comparison[['check_amount', 
                                            'check_payment_date']]=='Good')\
                .all().all()
            if has_mismatch:
                comparison_msg = f'Mismatch found for {fsym_id}.'
            if fstest_cur != last_cur:
                comparison_msg = f"""Possible currency change. 
                                Last payment: {last_cur}. 
                                Factset payment: {fstest_cur}"""
            if payment_exist_msg != '': comparison_msg = payment_exist_msg

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
    def plot_comparison(new_data: List[Dict],
                        div_selected_datatable: List[Dict]
                        ) -> Tuple[go.Figure, List[Dict], bool, str]:
        """
        Plot graph that compares Bloomberg and bg_div data

        Parameters
        ----------
        new_data : List[Dict]
            Bloomberg data.
        div_selected_datatable : List[Dict]
            bg_div data.

        Returns
        -------
        fig : go.Figure
            Plotted graph.
        display_option : List[Dict]
            Should the factset graph be showed (by using css style).
        is_open: bool
            If the warning message should be displayed.
        warning_msg : str
            Content of the warning message.

        """
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
    def plot_bbg(new_data: List[Dict]) -> go.Figure:
        """
        Plot Bloomberg graph.  

        Parameters
        ----------
        new_data : List[Dict]
            Bloomberg data.

        Returns
        -------
        fig : go.Figure
            Bloomberg graph.

        """
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
    def filter_fsym_id_editor(selected: str) -> List[Dict]:
        """
        Filter data based on secid selected in the dropdown for step 5 editor

        Parameters
        ----------
        selected : str
            Selected secid in the drop down.

        Returns
        -------
        List[Dict]
            Filtered data for the selected secid.

        """
        if modify_data.shape[0] == 0: return no_update
        output_df = modify_data.copy()
        for col in ['declared_date' , 'exdate', 'payment_date', 'record_date']:
            output_df[col] = pd.DatetimeIndex(output_df[col]).strftime("%Y-%m-%d")

        if selected == 'All':
            return output_df.to_dict('records')
        return output_df[output_df['fsym_id'] == selected].to_dict('records')
     
    @app.callback(
        Output('modify-button', 'disabled'),
        Input('modify-switch', 'value'))
    def disable_modify_button(is_switch_on: bool) -> bool:
        """
        Disable modify button (that adds current secid for being edited later)
        if switch(that indicates all secids for editing is added) is on

        Parameters
        ----------
        is_switch_on : bool
            The switch indicates that the user finished adding secids 
            for the editor.

        Returns
        -------
        bool
            Whether the button for adding secid is disabled.

        """
        if is_switch_on:
            return True
        return False
    
    @app.callback(
            Output('modify-list', 'data'),
            Output('add-to-editor-msg', 'is_open'),
            Input('modify-button', 'n_clicks'),
            Input('view-type-radio', 'value'),
            Input('div-date-picker', 'date'),
            Input('index-only-radio', 'value'),
            State('fsym-id-dropdown', 'value'),
            State('modify-list', 'data'))
    @print_callback(debug_mode)
    def update_modify_list(n_clicks: int, view_type_radio: str,
                           update_date: str, index_only: bool, cur_fsym_id: str,
                           modify_list: List[Dict]) -> Tuple[List[str], bool]:
        """
        Add the current secid to the list that can be edited by the editor and
        revert the list to blank if 1) the date is switched, 2) the view type
        (all_goods, mismatched, skipped) is switched or 3) the index member 
        only option is changed

        Parameters
        ----------
        n_clicks : int
            Number of clicks for the button of adding current secid.
        view_type_radio : str
            All_goods, mismatched or skipped.
        update_date : str
            Date selected in the date picker.
        index_only : bool
            Whether to choose only index members.
        cur_fsym_id : str
            Current secid in the secid dropdown for step 4 the data viewer.
        
        : List[Dict]
            List of secids for data that's to be edited in the 
            editor in step 5.

        Returns
        -------
        Tuple[List[str], bool]
            List of secids for data that's to be edited in step 5.
            Message letting the user know that the current secid is added
            in the editor.

        """
        changed_id = [p['prop_id'] for p in callback_context.triggered][0] 
        if changed_id == 'view-type-radio.value' or\
        changed_id == 'div-date-picker.date' or\
        changed_id == 'index-only-radio.value':
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
                        view_type: str, view_type_lst: List[Dict]
                        ) -> Tuple[str, str, bool]:
        """
        Go to the next secid or the previous secid

        Parameters
        ----------
        prev_clicks : int
            The number of clicks for the previous button.
        next_clicks : int
            The number of clicks for the next button.
        cur_fsym_id : str
            Current selected secid.
        view_type : str
            All_goods, mismatched or skipped.
        view_type_lst : List[Dict]
            List of secids for the options of all_goods, mismatched or skipped.

        Returns
        -------
        Tuple[str, str, bool]
            Secid after tapping the next or prev button.
            Warning message if this is the last/first secid.
            Whether the warning message is open.

        """
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
    def highlight_changed_cell(data: List[Dict]) -> List[Dict]:
        """
        Highlight the cell modified in editor history

        Parameters
        ----------
        data : List[Dict]
            Editor history.

        Returns
        -------
        List[Dict]
            Dash styling format.

        """
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
    def get_editor_data(modify_lst: List[Dict], is_switch_on: bool,
                        datatable: List[Dict]) -> Tuple[List[Dict], str]:
        """
        Get the data for the editor and populate the editor dropdown

        Parameters
        ----------
        modify_lst : List[Dict]
            List of secid for data that needs to be edited.
        is_switch_on : bool
            The switch indicates that the user finished adding secids 
            for the editor.
        datatable : List[Dict]
            Datatable storing data for the current selection.

        Returns
        -------
        Tuple[List[Dict], str]
            The dropdown for the editor and the default value for the dropdown.

        """
        if not modify_lst: return no_update
        df = pd.DataFrame(datatable)
        lst = list(set([row['name'] for row in modify_lst]))
        lst.sort()
        global modify_data 
        modify_data = df[df['fsym_id'].isin(lst)]
        fsym_ids = [{"label": 'All', "value": 'All'}] +\
            [{"label": i, "value": i} for i in lst]
        if not is_switch_on: return no_update, no_update
        return fsym_ids , fsym_ids[0]['value']
 
    @app.callback(
        Output('editor-open-msg', 'is_open'),
        Output('editor-main', 'style'),
        Input('modify-switch', 'value'),
        )   
    def show_editor_info_msg(is_switch_on: bool) -> Tuple[bool, Dict]:
        """
        Shows an info message if the editor switch is not turned on

        Parameters
        ----------
        is_switch_on : bool
            The switch for finishing adding secids for the editor.

        Returns
        -------
        Tuple[bool, Dict]
            Info message letting the user know that the editor switch needs
                to be turned on for the editor to open.
            Style for the editor (whether displays the editor or not)

        """
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
                                  selected: str, 
                                  modified_datatable: List[Dict]) -> List[Dict]:
        """
        

        Parameters
        ----------
        rows : List[Dict]
            The current copy of rows in the edit history.
        rows_prev : List[Dict]
            The previous copy of rows in the edit history.
        selected : str
            The current selected secid from the secid dropdown for editors.
        modified_datatable : List[Dict]
            The editable datatable.

        Returns
        -------
        List[Dict]
            The storage for the editable datatable (unused at the moment unless
            found a way to avoid the circular callback).

        """
        if not rows: return no_update
        global modify_data
        modified_df = pd.DataFrame(modified_datatable)
        if selected != 'All':
            modify_data = modify_data[~(modify_data['fsym_id'] == selected)]
            modify_data = pd.concat([modify_data, modified_df])
        else:
            modify_data = modified_df
        
        res = modify_data.to_dict('records')
        undo_delete_row = [row for row in rows_prev if row not in rows] if \
            (rows is not None and rows_prev is not None) and \
                len(rows_prev) > len(rows) else []
        # res  = res +  undo_delete_row if undo_delete_row is not None else res
        # pd.DataFrame(undo_delete_row).drop(columns=['action'], inplace=True)
        modify_data = pd.concat([modify_data, pd.DataFrame(undo_delete_row)])
        return res
    
    @app.callback(
        Output('modified-data-rows', 'data'),
        Input('modify-switch', 'value'),
        Input('output-data-table', 'data_previous'),
        State('output-data-table', 'data'),
        State('modified-data-rows', 'data'))
    @print_callback(debug_mode)
    def update_modified_data_table(is_switch_on: bool, rows_prev: List[Dict],
                                   rows: List[Dict], 
                                   modified_rows: List[Dict]) -> List[Dict]:
        """
        Populate edit history and add index for reference for highlighting

        Parameters
        ----------
        is_switch_on : bool
            The switch for finished adding the sec list.
        rows_prev : List[Dict]
            The previous copy of rows in the editable datatable.
        rows : List[Dict]
            The current copy of rows in the editable datatable.
        modified_rows : List[Dict]
            Edit history (the previous copy).

        Returns
        -------
        List[Dict]
            Edit history.

        """
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'modify-switch.value' == changed_id:
            return []
        if not rows and not rows_prev:
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
    def export_not_modified_data(nclicks: int, modify_lst: List[Dict],
                                 new_data: List[Dict]
                                 ) -> Tuple[str, bool]:
        """
        Export data as it is, that haven't been modified by editor

        Parameters
        ----------
        nclicks : int
            Save button is clicked.
        modify_lst : List[Dict]
            List of the secids that have been added to the editor.
        new_data : List[Dict]
            Processed Bloomberg data.

        Raises
        ------
        PreventUpdate
            No update for the non-clicked case, prevent callback from firing.

        Returns
        -------
        Tuple[str, bool]
            A message letting user know that the button has been clicked.

        """
        
        if nclicks == 0:
            raise PreventUpdate
        else:
            df = pd.DataFrame(new_data)
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
    def export_modified_data(nclicks: int) -> Tuple[str, bool]: 
        """
        Export data after being modified by editor

        Parameters
        ----------
        nclicks : int
            Save button is clicked.

        Raises
        ------
        PreventUpdate
            No update for the non-clicked case, prevent callback from firing.

        Returns
        -------
        Tuple[str, bool]
            A message letting user know that the button has been clicked.

        """
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