# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:32:36 2021

@author: Todd.Liu
"""
from pathlib import Path
#%%
import functools
from typing import Optional, List, Tuple, Dict
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')
from bg_data_importer import DataImporter
from datetime import datetime, date, timedelta
data_importer_uploader = DataImporter(verbose=False)

# Plotting
import plotly.express as px
import plotly.graph_objects as go

#%%

def load_factset_data(fysm_id_lst: List) -> pd.DataFrame:
    """
    Load factset data from basic_dividend table in the database

    Parameters
    ----------
    fysm_id_lst : List
        List of fsym_id for the data we want to import.

    Returns
    -------
    df : pd.DataFrame
        DataFrame of our imported data.

    """
    if not len(fysm_id_lst):
        return
    query = """SELECT 
			bd.fsym_id, currency,p_divs_exdate,p_divs_recdatec,p_divs_paydatec, 
				p_divs_pd,p_divs_s_pd FROM fstest.fp_v2.fp_basic_dividends bd
			 
			  LEFT JOIN (select fsym_id, max(exdate) as exdate 
                        from fstest.dbo.bg_div div 
                        where fsym_id IN {} GROUP BY fsym_id) AS div
			  ON bd.fsym_id  = div.fsym_id

			  where bd.fsym_id IN {}
			  AND p_divs_exdate >  exdate""".\
                  format(str(tuple(fysm_id_lst)), str(tuple(fysm_id_lst)))
    df = data_importer_uploader.load_data(query)
    return df
    
def load_basic_info_data(fysm_id_lst: List) -> pd.DataFrame:
    """
    Load basic information data for the holdings we want to show

    Parameters
    ----------
    fysm_id_lst : List
        List of fsym_id for the data we want to import.

    Returns
    -------
    df : pd.DataFrame
        DataFrame of our imported data.

    """
    if not len(fysm_id_lst):
        return
    query = """select 
			sc.fsym_id, bbg.bbg_id, proper_name, tr.ticker_region 'ticker'
            from fstest.sym_v1.sym_coverage sc
            left join fstest.sym_v1.sym_ticker_region tr on tr.fsym_id = sc.fsym_id
			left join fstest.sym_v1.sym_bbg bbg on bbg.fsym_id = sc.fsym_id
            where sc.fsym_id IN {}""".format(str(tuple(fysm_id_lst)))
    df = data_importer_uploader.load_data(query)
    return df

def load_bg_div_data(fysm_id_lst: List) -> pd.DataFrame:
    """
    Load dividend data from the database

    Parameters
    ----------
    fysm_id_lst : List
        List of fsym_id for the data we want to import.

    Returns
    -------
    df : pd.DataFrame
        DataFrame of our imported data.

    """
    if not len(fysm_id_lst):
        return
    query = """SELECT fsym_id, exdate, div_freq, div_initiation, payment_amount,
            div_type FROM fstest.dbo.bg_div WHERE fsym_id IN {}""".\
                format(str(tuple(fysm_id_lst)))
    df = data_importer_uploader.load_data(query)
    return df

def load_split_data(fysm_id_lst: List) -> pd.DataFrame:
    """
    Load split data from the database

    Parameters
    ----------
    fysm_id_lst : List
        List of fsym_id for the data we want to import.

    Returns
    -------
    df : pd.DataFrame
        DataFrame of our imported data.

    """
    if not len(fysm_id_lst):
        return

    query = """SELECT fsym_id,p_split_date,p_split_factor,
                exp(sum(log(p_split_factor)) OVER (ORDER BY p_split_date desc)) 
                cum_split_factor FROM fstest.fp_v2.fp_basic_splits
                WHERE fsym_id IN {} order by p_split_date""".\
                    format(str(tuple(fysm_id_lst)))
    df = data_importer_uploader.load_data(query)
    if df.shape[0] != 0:
        df.sort_values(by=['p_split_date'], ascending=False, inplace=True)
    return df

# =============================================================================
# Plots
# =============================================================================

def plot_dividend_data(new_data: pd.DataFrame, bg_data: pd.DataFrame
                       ) -> Tuple[go.Figure, str]:
    """
    Plot dividend graph for the case when data already exists in DB

    Parameters
    ----------
    new_data : pd.DataFrame
        DESCRIPTION.
    bg_data : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    fig : go.Figure
        Dividend graph plotted on input data.
    msg : str
        Message if payments already exists.

    """
    msg = ''
    fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'},
                                     yaxis=dict(title='Amount'),
                                     yaxis2=dict(title='Freq', overlaying='y',
                                                 side='right', range=[0, 14])))
    new = new_data.copy()
    new['div_type'] = new['div_type'].str.replace(' ', '')
    new_regular = new[new['div_type'] != 'special']
    new_special = new[new['div_type'] == 'special']
    
    bg = bg_data.copy()
    bg['div_type'] = bg['div_type'].str.replace(' ', '')
    bg_regular = bg[bg['div_type'] != 'special']
    bg_special = bg[bg['div_type'] == 'special']
    
    bg['exdate'] = pd.to_datetime(bg['exdate'], format='%Y-%m-%d')
    new['exdate'] = pd.to_datetime(new['exdate'], format='%Y-%m-%d')
    already_exist = list(set(bg['exdate']).intersection(set(new['exdate'])))
    if len(already_exist) > 0:
        if len(already_exist) == 1:
            already_exist = already_exist[0].strftime('%Y-%m-%d')
        else:    
            already_exist = [already_exist_date.strftime('%Y-%m-%d')
                             for already_exist_date in already_exist]
        msg = f'Payments on {already_exist} already exists.'

        return fig, msg
    fig.add_trace(
        go.Scatter(x=new_regular['exdate'], y=new_regular['payment_amount'],
                   mode='lines+markers', name='New Regular', 
                   line=dict(color='orchid')))
    fig.add_trace(
        go.Scatter(x=new_special['exdate'], y=new_special['payment_amount'],
                   mode='markers',name='New Special', line=dict(color='red')))
    fig.add_trace(
        go.Scatter(x=new_special['exdate'], y=new_special['div_freq'],
                   mode='markers', name='Div Freq', 
                   line=dict(color='green'), yaxis='y2'))
    fig.add_trace(
        go.Scatter(x=bg_regular['exdate'], y=bg_regular['payment_amount'],
                   mode='lines+markers',name='BG Regular',
                   line=dict(color='grey')))
    fig.add_trace(
        go.Scatter(x=bg_regular['exdate'], y=bg_regular['div_freq'], 
                   mode='markers', name='Div Freq',
                   line=dict(color='green'), yaxis='y2'))
    if sum(bg_regular['div_initiation']) > 0:
        fig.add_trace(
            go.Scatter(x=bg_regular['exdate'],
                       y=bg_regular['div_initiation'].astype(int).replace(0, np.nan),
                       mode='markers',name='BG Init', 
                       line=dict(color='dodgerblue')))
    fig.add_trace(
        go.Scatter(x=bg_special['exdate'], y=bg_special['payment_amount'],
                   mode='markers',name='BG Special', line=dict(color='black')))

    if new.shape[0]>1:
        fig.update_layout(shapes=[dict(type="rect", xref="x", 
                                       yref="paper",x0=new['exdate'].min(),
                                       y0=0,x1=new['exdate'].max(),
                                       y1=1,fillcolor="LightSalmon",opacity=0.5,
                                       layer="below",line_width=0)])
    return fig, msg


def plot_generic_dividend_data(df: pd.DataFrame) -> go.Figure:
    """
    Plot dividend graph for one df

    Parameters
    ----------
    df : pd.DataFrame
        Dividend data we want to plot
    
    Returns
    -------
    fig : go.Figure
        Dividend graph based on input data.

    """
    fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'}))
    df['div_type'] = df['div_type'].str.replace(' ', '')
    df_regular = df[df['div_type'] == 'regular']
    df_special = df[df['div_type'] == 'special']
    fig.add_trace(go.Scatter(x=df_regular['exdate'], 
                             y=df_regular['payment_amount'], 
                             mode='lines+markers', name='Regular',
                             line=dict(color='orchid')))
    fig.add_trace(go.Scatter(x=df_special['exdate'], 
                             y=df_special['payment_amount'],
                             mode='markers',name='Special', 
                             line=dict(color='black')))
    if df_regular['div_initiation'].sum() > 0:
        fig.add_trace(go.Scatter(x=df_regular['exdate'], 
                                 y=df_regular['div_initiation'].\
                                     astype(int).replace(0, np.nan),
                                 mode='markers', 
                                 name='BG Init', line=dict(color='dodgerblue')))
    if 'suspension' in df['div_type'].values:
        df_suspension = df[df['div_type'] == 'suspension']
        fig.add_trace(go.Scatter(x=df_suspension['exdate'], 
                                 y=[df_regular['payment_amount'].max()]*\
                                     df_suspension.shape[0], mode='markers',
                                     name='BG Suspension',
                                     line=dict(color='Red')))
    return fig
        

        
def plot_dividend_data_comparison(bg: pd.DataFrame, bbg: pd.DataFrame
                                  ) -> go.Figure:
    """
    Plot dividend comparison graph

    Parameters
    ----------
    bg : pd.DataFrame
        Dividend data from DB
    bbg : pd.DataFrame
        Bloomberg data.

    Returns
    -------
    fig : go.Figure
        Dividend comparison graph.

    """
    fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'}))

    bbg['div_type'] = bbg['div_type'].str.replace(' ', '')
    bbg_regular = bbg[bbg['div_type'] == 'regular']
    bbg_special = bbg[bbg['div_type'] == 'special']
    
    bg['div_type'] = bg['div_type'].str.replace(' ', '')
    bg_regular = bg[bg['div_type'] == 'regular']
    bg_special = bg[bg['div_type'] == 'special']
    
    fig.add_trace(go.Scatter(x=bg_regular['exdate'], 
                         y=bg_regular['payment_amount'],
                         mode='lines+markers', name='BG Regular',
                         line=dict(color='orchid')))
    fig.add_trace(go.Scatter(x=bg_special['exdate'], 
                             y=bg_special['payment_amount'], 
                             mode='markers',name='BG Special', 
                             line=dict(color='red')))
    fig.add_trace(go.Scatter(x=bbg_regular['exdate'], 
                             y=bbg_regular['payment_amount'],
                             mode='lines+markers', name='BBG Regular',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=bbg_special['exdate'], 
                         y=bbg_special['payment_amount'],
                         mode='markers',name='BBG Special',
                         line=dict(color='black')))
    return fig
    
#%% Functions
def switch_freq_to_number(freq: str) -> int:
    """
    Convert dividend frequenty from word to number
    of months correspondingly

    Parameters
    ----------
    freq : str
        Dividend frequency (Quarter, Monthly, Annual, Semi-Anl, Irreg, None).

    Returns
    -------
    int
        Number of month for each dividend payment such as 4, 12, 2 etc.

    """
    if freq == 'Quarter' : return 4
    if freq == 'Monthly' : return 12
    if freq == 'Semi-Anl' : return 2
    if freq == 'Annual' : return 1
    if freq == 'None' : return 0
    if freq == 'Irreg' : return -1

def convert_div_type(div_type: str) -> str:
    """
    Convert dividend type to standard usage for BG

    Parameters
    ----------
    div_type : str
        Type of dividend such as Regular Cash, Special Cash,
        Ommited, Cancelled.

    Returns
    -------
    str
        Type of dividend such as regular, special, etc.

    """
    if div_type == 'Regular Cash': return 'regular'
    if div_type == 'Special Cash': return 'special'
    if div_type in ['Omitted', 'Cancelled']: return div_type.lower()
    return div_type

def prepare_bbg_data(new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust Bloomberg data's currency before plotting

    Parameters
    ----------
    new_data : pd.DataFrame
        Bloomberg data.

    Returns
    -------
    new_data : pd.DataFrame
        Bloomberg data after adjusted for the plot.

    """
    (last_cur, fstest_cur) = dividend_currency(new_data)
    if last_cur is None:
        last_cur = fstest_cur
        new_data['listing_currency'] = fstest_cur
    else:
        new_data['payment_currency'] = last_cur
    return new_data

def preprocess_bbg_data(df_: pd.DataFrame, bbgid: pd.DataFrame, 
                        index_flag: bool) -> Tuple[pd.DataFrame, 
                                                   pd.DataFrame, pd.DataFrame]:
    """
    Clean and rename columns for factset data  

    Parameters
    ----------
    df_ : pd.DataFrame
        Data before processing.
    bbgid : pd.DataFrame
        Update list from the csv.
    index_flag : bool
        Whether to only include index constituents.

    Returns
    -------
    df : pd.DataFrame
        Data after processing.
    skipped : pd.DataFrame
        Skipped data after processing.
    pro_rata : pd.DataFrame
        Pro rata data after processing.

    """
    
    df = df_.copy()
    dates_cols = df.filter(like='Date').columns
    df[dates_cols] = df[dates_cols].apply(pd.to_datetime, axis=1)
    df.columns=[name.lower().replace(' ','_').replace('-','')
                for name in df.columns]
    df = df.rename(columns={'security' : 'bbg_id'})
    df['bbg_id'] = df['bbg_id'].str.replace('/bbgid/', '')
    bbgid = bbgid[['bbg_id','fsym_id', 'in_index', 'listing_currency']]
    df = pd.merge(df, bbgid, on='bbg_id', how='left')
    df.sort_values(['fsym_id','exdate'], inplace=True)
    df.rename(columns={'payable_date':'payment_date',
                       'dividend_amount':'payment_amount',
                       'dividend_type':'div_type',
                       'dividend_frequency':'div_freq',
                       'dvd_crncy':'payment_currency'},
              inplace=True)

    if index_flag:
        df = df[df['in_index'] == 1]
    else:
        df = df[df['in_index'] == 0]
    df = df[~df['fsym_id'].isna()]

    df['div_freq'] = df['div_freq'].apply(switch_freq_to_number)
    pro_rata = df[df['div_type'].isin(['Pro Rata'])].copy()
    skipped = df[df['div_type'].isin(['Omitted', 'Cancelled',
                                      'Discontinued'])].copy()
    skipped['dividend_amount'] = 0
    # splits = df[df['div_type']=='Stock Split'].copy()
    df = df[df['div_type'].isin(['Regular Cash', 'Special Cash'])]
    df['div_type'] = df['div_type'].apply(convert_div_type)
    df['div_freq'] = np.where(df['div_type'] == 'special', np.nan, df['div_freq'])
    df = bbg_data_single_security(df)
    return (df, skipped, pro_rata)

def factset_data_single_security(df: pd.DataFrame, skip_flag: Optional[bool]=False
                                 ) -> pd.DataFrame:
    """
    Clean and rename columns for factset data

    Parameters
    ----------
    df : pd.DataFrame
        Data before processing
    skip_flag : bool, optional
        Whether the dividend is skipped. The default is False.

    Returns
    -------
    df : pd.DataFrame
        Data after processing.
    """
    df.rename(columns={ 'p_divs_exdate':'exdate',
                        'p_divs_pd':'payment_amount',
                        'p_divs_paydatec':'payment_date',
                        'p_divs_recdatec':'record_date',
                        'currency':'listing_currency'},
              inplace=True)
    df['div_initiation'] = 0
    df['skipped'] = skip_flag
    df['div_type'] = np.where(df['p_divs_s_pd']==1,'special','regular')
    df['div_freq'] = np.nan
    del df['p_divs_s_pd']
    return df


def bbg_data_single_security(df_: pd.DataFrame, 
                             skip_flag: Optional[bool]=False, 
                             ) -> pd.DataFrame:
    """
    Clean and rename columns for data (such as BBG data)

    Parameters
    ----------
    df_ : pd.DataFrame
        Data before processing
    skip_flag : bool, optional
        Whether the dividend is skipped. The default is False.

    Returns
    -------
    df : pd.DataFrame
        Data after processing.

    """
    df = df_.copy()
    keep_cols = ['fsym_id','listing_currency', 'payment_currency','declared_date',
                 'exdate', 'record_date', 'payment_date','payment_amount',
                 'div_type','div_freq']
    df = df[keep_cols].copy()
    df['div_initiation'] = 0
    df['skipped'] = skip_flag
    df['div_type'] = df['div_type'].apply(convert_div_type)
    df['div_freq'] = np.where(df['div_type']=='special', np.nan, df['div_freq'])
    return df

def dividend_currency(crncy: Optional[pd.DataFrame]=None) -> Tuple[str, str]:
    """
    Get the listing and payment currency of the given data

    Parameters
    ----------
    crncy : Optional[pd.DataFrame], optional
        The given data containing currency. The default is None.

    Returns
    -------
    Tuple[str, str]
        Last (payment) currency and listing currency.

    """
    last_cur = crncy.iloc[0]['payment_currency']
    fstest_cur = crncy.iloc[0]['listing_currency']
    return (last_cur, fstest_cur)


def check_before_upload(df: pd.DataFrame) -> bool:
    """
    Verify data is properly processed before moving to the next step

    Parameters
    ----------
    df : pd.DataFrame
        Data that needs to be verified.

    Returns
    -------
    bool
        Whether the data is processed properly.

    """
    secid = df['fsym_id'].iloc[0]
    if 'Pro Rata' in df['div_type']:
        return False
    for x in list(df['div_type'].unique()):
        if x not in ['regular','special', 'mixed','suspension']:
            print(secid, end=' | ')
            print('Div Type: ' + x)
            return False
    regulars = df[df['div_type']=='regular']
    specials = df[df['div_type']=='special']
    skipped = df[df['skipped'] == True]
    # suspension = df[df['div_suspension'] == True]
    for x in list(regulars['div_freq'].unique()):
        if x not in [1, 2, 4, 12]:
            print(secid, end=' | ')
            print('Regular: ' + str(x))
            return False
    for x in list(specials['div_freq'].unique()):
        if ~np.isnan(x):
            print(secid, end=' | ')
            print('Special: ' + str(x))
            return False
    if skipped.shape[0] == 0:
        return True
    if ('regular' not in skipped['div_type'].values) and \
        ('suspension' not in skipped['div_type'].values):
        print('Skipped does not have proper div_type')
        return False
    return True

def basic_info(info: pd.DataFrame, new_data_: pd.DataFrame) -> str:
    """
    Get basic information about the current stock

    Parameters
    ----------
    info : pd.DataFrame
        Information df by querying from the database.
    new_data_ : pd.DataFrame
        New Bloomberg data.

    Returns
    -------
    str
        Formatted string containing basic information.

    """
    secid = info.iloc[0]['fsym_id']
    name = info.iloc[0]['proper_name']
    bbg_id = info.iloc[0]['bbg_id']
    ticker = info.iloc[0]['ticker']
    fs_cur = new_data_['listing_currency'].values[0]
    bbg_cur = new_data_['payment_currency'].values[0]

    return f"""
    >
    **Fsym Id**: {secid}      |      **BBG Id**: {bbg_id}   |      
    **Company name**: {name}             
    **Ticker**: {ticker}      |        **BBG Crncy**: {bbg_cur}        |
    **FS Crncy**: {fs_cur}
    >
    """

def compare_new_data_with_factset(update_date: datetime,
                                  bbg: pd.DataFrame,
                                  factset: pd.DataFrame, check_exist: bool
                                  ) -> pd.DataFrame:
    """
    Compare Bloomberg data with factset data and identify descrepancy in date
    and price

    Parameters
    ----------
    update_date : datetime
        Selected last day of the month for the month we need to check.
    bbg : pd.DataFrame
        Bloomberg data.
    factset : pd.DataFrame
        Factset data.
    check_exist : bool
        Whether the data exists.

    Returns
    -------
    new_data_comparison : pd.DataFrame
        Comparison df that outlines mismatch in date and price of the data.

    """
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


def bulk_upload(df: pd.DataFrame, update_date: datetime, 
                factset_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """
    Find the entries that needs to be checked manually
    and can be directly uploaded

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned new data from the parquet file.
    update_date : datetime
        Selected last day of the month for the month we need to check.
    factset_df : pd.DataFrame
        Factset data loaded from DB.

    Returns
    -------
    List[str]
        List of secids for data that needs to be checked manually.
    pd.DataFrame
        Data that does not need to be checked manually.

    """
    success_list = []
    checked_list = []
    seclist = sorted(list(df['fsym_id'].unique()))
    for secid in seclist:
        new = df[df['fsym_id']==secid].copy()
        (last_cur, fstest_cur) = dividend_currency(new)
        if fstest_cur != last_cur:
            continue       
        factset = factset_df[factset_df['fsym_id']==secid]
        comp = compare_new_data_with_factset(update_date, new,
                                             factset,
                                             factset.shape[0] != 0)

        if not (comp[['check_amount', 'check_payment_date']]=='Good').all().all():
            continue
        
        check_flag = check_before_upload(new)
        if not check_flag:
            continue
        checked_list.append(secid)
        success_list.append(new)
    if len(success_list) == 0:
        return (list(set(seclist) ^ set(checked_list)), None)
    df = pd.concat(success_list)
    return (list(set(seclist) ^ set(df['fsym_id'].tolist())), df)

def process_skipped(skipped_: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and rename columns for skipped data

    Parameters
    ----------
    skipped_ : pd.DataFrame
        Dataframe for raw skipped data.

    Returns
    -------
    skipped : pd.DataFrame
        Dataframe for skipped data after processing.

    """
    skipped = skipped_.copy()
    skipped = bbg_data_single_security(skipped)
    skipped['div_type'] = np.where(skipped['div_type']=='Discontinued',
                                   'suspension',
                                   'regular')
    skipped['skipped'] = np.where(skipped['div_type']=='suspension', False, True)
    skipped['div_freq'] = np.where(skipped['div_freq']==-1,
                                   np.nan,
                                   skipped['div_freq'])
    skipped['payment_amount'] = 0
    return skipped

