# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:32:36 2021

@author: Todd.Liu
"""
from pathlib import Path
#%%
import functools
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')
from bg_data_importer import DataImporter
from datetime import datetime
data_importer_uploader = DataImporter(verbose=False)#TODO changed

# Plotting
import plotly.express as px
import plotly.graph_objects as go

#%%
def plot_dividend_data(fsym_id, new_data, bg_data, alt_cur=None):
    msg = ''
    fig = go.Figure(layout=go.Layout(xaxis={'type': 'date'},
                                     yaxis=dict(title='Amount'),
                                     yaxis2=dict(title='Freq', overlaying='y',
                                                 side='right', range=[0, 14])))
    new = new_data[new_data['fsym_id'] == fsym_id].copy()
    new['div_type'] = new['div_type'].str.replace(' ', '')
    new_regular = new[new['div_type'] != 'special']
    new_special = new[new['div_type'] == 'special']
    
    bg = bg_data.copy()
    bg['div_type'] = bg['div_type'].str.replace(' ', '')
    bg_regular = bg[bg['div_type'] != 'special']
    bg_special = bg[bg['div_type'] == 'special']
    
    already_exist = list(set(bg['exdate']).intersection(set(new['exdate'])))
    if len(already_exist) > 0:
        already_exist = [already_exist_date.strftime('%Y/%m/%d/')
                         for already_exist_date in already_exist],
        msg = f'Payments on {already_exist} already exists.'

        return fig, msg
    fig.add_trace(
        go.Scatter(x=new_regular['exdate'], y=new_regular['payment_amount'],
                   mode='lines+markers', name='New Regular', line=dict(color='orchid')))
    fig.add_trace(
        go.Scatter(x=new_special['exdate'], y=new_special['payment_amount'],
                   mode='markers',name='New Special', line=dict(color='red')))
    fig.add_trace(
        go.Scatter(x=new_special['exdate'], y=new_special['div_freq'],
                   mode='markers', name='Div Freq', line=dict(color='green'), yaxis='y2'))
    fig.add_trace(
        go.Scatter(x=bg_regular['exdate'], y=bg_regular['payment_amount'],
                   mode='lines+markers',name='BG Regular', line=dict(color='grey')))
    fig.add_trace(
        go.Scatter(x=bg_regular['exdate'], y=bg_regular['div_freq'], 
                   mode='markers', name='Div Freq', line=dict(color='green'), yaxis='y2'))
    if sum(bg_regular['div_initiation']) > 0:
        fig.add_trace(
            go.Scatter(x=bg_regular['exdate'],
                       y=bg_regular['div_initiation'].astype(int).replace(0, np.nan),
                       mode='markers',name='BG Init', line=dict(color='dodgerblue')))
    fig.add_trace(
        go.Scatter(x=bg_special['exdate'], y=bg_special['payment_amount'],
                   mode='markers',name='BG Special', line=dict(color='black')))
    
    if alt_cur is not None:
        fig.add_trace(
            go.Scatter(x=alt_cur['exdate'], y=alt_cur['payment_amount'],
                       mode='lines+markers',name='Alt Cur', line=dict(color='purple')))
    
    if new.shape[0]>1:
        fig.update_layout(shapes=[dict(type="rect", xref="x", 
                                       yref="paper",x0=new['exdate'].min(),
                                       y0=0,x1=new['exdate'].max(),
                                       y1=1,fillcolor="LightSalmon",opacity=0.5,
                                       layer="below",line_width=0)])
    return fig, msg


#TODO changed func sig
def prepare_bbg_data(new_data, alt_cur=''):
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
                                     name='BG Suspension', line=dict(color='Red')))
    return fig
        

        
def plot_dividend_data_comparison(bg, bbg):
    # print('plot_dividend_data_comparison')
    # print('bg')

    # print(bg)
    # print('bbg')
    # print(bbg)

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
def switch_freq_to_number(freq):
    if freq == 'Quarter' : return 4
    if freq == 'Monthly' : return 12
    if freq == 'Semi-Anl' : return 2
    if freq == 'Annual' : return 1
    if freq == 'None' : return 0
    if freq == 'Irreg' : return -1

def convert_div_type(div_type):
    if div_type == 'Regular Cash': return 'regular'
    if div_type == 'Special Cash': return 'special'
    if div_type in ['Omitted', 'Cancelled']: return div_type.lower()
    return div_type

# @functools.lru_cache(maxsize=5)
def preprocess_bbg_data(df_, bbgid, index_flag, rdate):
    df = df_.copy()
    dates_cols = df.filter(like='Date').columns
    # print('preprocess_bbg_data')
    # print(dates_cols)
    df[dates_cols] = df[dates_cols].apply(pd.to_datetime, axis=1)
    # print(df.dtypes)
    df.columns=[name.lower().replace(' ','_').replace('-','')
                for name in df.columns]
    df = df.rename(columns={'security' : 'bbg_id'})
    df['bbg_id'] = df['bbg_id'].str.replace('/bbgid/', '')
    rdate = rdate.replace('-', '')
    bbgid = pd.read_csv(rf'\\bgndc\Analysts\Scheduled_Jobs\input\sec_list_{rdate}.csv')
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
    skipped = df[df['div_type'].isin(['Omitted', 'Cancelled','Discontinued'])].copy()
    skipped['dividend_amount'] = 0
    splits = df[df['div_type']=='Stock Split'].copy()
    df = df[df['div_type'].isin(['Regular Cash', 'Special Cash'])]
    df['div_type'] = df['div_type'].apply(convert_div_type)
    df['div_freq'] = np.where(df['div_type'] == 'special', np.nan, df['div_freq'])
    df = bbg_data_single_security(df)
    return (df, skipped, pro_rata, splits)

def factset_data_single_security(df, skip_flag=False):
    df.rename(columns={ 'p_divs_exdate':'exdate',
                        'p_divs_pd':'payment_amount',
                        'p_divs_paydatec':'payment_date',
                        'p_divs_recdatec':'record_date',
                        'currency':'listing_currency'},
              inplace=True)
    # print('factset_data_single_security:' )
    # print(df)
    df['div_initiation'] = 0
    df['skipped'] = skip_flag
    df['div_type'] = np.where(df['p_divs_s_pd']==1,'special','regular')
    df['div_freq'] = np.nan
    del df['p_divs_s_pd']
    return df

# def factset_new_data(secid, last_payment_exdate):
#     if secid is None or secid == '':
#         return
#     exdate_condition = ''
#     if last_payment_exdate is not None:
#         exdate_condition = f"and p_divs_exdate > '{last_payment_exdate}'"
#     query = f"""
#                 select bd.fsym_id, bbg.bbg_id,currency,p_divs_exdate,p_divs_recdatec,p_divs_paydatec,
#                     p_divs_pd,p_divs_s_pd from fstest.fp_v2.fp_basic_dividends bd
#               left join fstest.sym_v1.sym_bbg bbg on bbg.fsym_id = bd.fsym_id
# 			  where bd.fsym_id = '{secid}' {exdate_condition}
#               """
#     df = data_importer_uploader.load_data(query)
#     # print('factset_new_data import: ')
#     # print(df)
#     df = factset_data_single_security(df)
#     # print(df.dtypes)
#     return df

def load_factset_data(fysm_id_lst):
    if fysm_id_lst is None or fysm_id_lst == []:
        return
    print('load_factset_data')
    # print(tuple(fysm_id_lst))
    # print(str(tuple(fysm_id_lst)))
    query = """select 
			bd.fsym_id, currency,p_divs_exdate,p_divs_recdatec,p_divs_paydatec, 
				p_divs_pd,p_divs_s_pd from fstest.fp_v2.fp_basic_dividends bd
			 
			  LEFT JOIN (select fsym_id, max(exdate) as exdate 
                        from fstest.dbo.bg_div div 
                        where fsym_id IN {} GROUP BY fsym_id) AS div
			  ON bd.fsym_id  = div.fsym_id

			  where bd.fsym_id IN {}
			  AND p_divs_exdate >  exdate""".\
                  format(str(tuple(fysm_id_lst)), str(tuple(fysm_id_lst)))
        
    # print(query)
    df = data_importer_uploader.load_data(query)
    # print(df)
    return df
    
def load_basic_info_data(fysm_id_lst):
    if fysm_id_lst is None or fysm_id_lst == []:
        return
    # print(fysm_id_lst)
    # print(tuple(fysm_id_lst))
    # print(str(tuple(fysm_id_lst)))
    query = """select 
			sc.fsym_id, bbg.bbg_id, proper_name, tr.ticker_region 'ticker'
            from fstest.sym_v1.sym_coverage sc
            left join fstest.sym_v1.sym_ticker_region tr on tr.fsym_id = sc.fsym_id
			left join fstest.sym_v1.sym_bbg bbg on bbg.fsym_id = sc.fsym_id
            where sc.fsym_id IN %s""" %\
        str(tuple(fysm_id_lst))
    # print(query)
    df = data_importer_uploader.load_data(query)
    # print(df)
    return df

def load_bg_div_data(fysm_id_lst):
    if fysm_id_lst is None or fysm_id_lst == []:
        return
    # print(fysm_id_lst)
    # print(tuple(fysm_id_lst))
    # print(str(tuple(fysm_id_lst)))
    query = """SELECT fsym_id, exdate,div_freq, div_initiation,payment_amount, div_type 
                from fstest.dbo.bg_div where fsym_id IN %s""" %\
        str(tuple(fysm_id_lst))
    # print(query)
    df = data_importer_uploader.load_data(query)
    # print(df)
    return df

# def load_bg_div_exdates(fysm_id_lst):
#     if fysm_id_lst is None or fysm_id_lst == []:
#         return

#     query = "select fsym_id, max(exdate) from fstest.dbo.bg_div where fsym_id IN %s GROUP BY fsym_id" %\
#         str(tuple(fysm_id_lst))
#     df = data_importer_uploader.load_data(query)
#     print('load_bg_div_exdates')
#     # print(df)
#     return df

def load_split_data(fysm_id_lst):
    if fysm_id_lst is None or fysm_id_lst == []:
        return

    query = """select fsym_id,p_split_date,p_split_factor, exp(sum(log(p_split_factor))
                OVER (ORDER BY p_split_date desc)) cum_split_factor from fstest.fp_v2.fp_basic_splits
                where fsym_id IN %s order by p_split_date""" %\
        str(tuple(fysm_id_lst))
    df = data_importer_uploader.load_data(query)
    print('load_split_data')
    # print(query)
    # print(df)
    df.sort_values(by=['p_split_date'], ascending=False, inplace=True)
    return df

def bbg_data_single_security(df_, skip_flag=False, suspension=False):
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

# def last_payment_exdate(secid):
#     query = f"""
#                 select max(exdate) from fstest.dbo.bg_div where fsym_id = '{secid}'
#             """
#     last_date = data_importer_uploader.load_data(query)
#     print(last_date)
#     if pd.isna(last_date.iloc[0,0]):
#         return None
#     return last_date.iloc[0,0].strftime('%Y-%m-%d')

def last_payment_exdate(secid, exdate_col):
    # query = f"""
    #             select max(exdate) from fstest.dbo.bg_div where fsym_id = '{secid}'
    #         """
    last_date = exdate_col.max()
    if pd.isna(last_date):
        return None
    last_date = datetime.strptime(last_date, '%Y-%m-%dT%H:%M:%S')
    print(last_date)
    return last_date.strftime('%Y-%m-%d')

#TODO
def dividend_currency(crncy=None):
    # crncy = new_data_[new_data_['fsym_id']==secid]
    last_cur = crncy.iloc[0]['payment_currency']
    fstest_cur = crncy.iloc[0]['listing_currency']
    return (last_cur, fstest_cur)

def dividend_currency_bulk_upload(secid, new_data_=None):
    crncy = new_data_[new_data_['fsym_id']==secid]
    last_cur = crncy.iloc[0]['payment_currency']
    fstest_cur = crncy.iloc[0]['listing_currency']
    return (last_cur, fstest_cur)

# def check_existence(secid):
#     query = f"""
#              select 1 from fstest.dbo.bg_div where fsym_id = '{secid}'
#     """
#     df = data_importer_uploader.load_data(query)
#     if df.shape[0] == 0:
#         return False
#     return True

def check_before_upload(df):
    # flag =True
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

def basic_info(secid, info, new_data_):

    # query = f"""
    #         select proper_name, currency, tr.ticker_region 'ticker'
    #         from fstest.sym_v1.sym_coverage sc
    #         left join fstest.sym_v1.sym_ticker_region tr on tr.fsym_id = sc.fsym_id
    #         where sc.fsym_id = '{secid}'
    #         """
    # info = data_importer_uploader.load_data(query)
    name = info.iloc[0]['proper_name']
    bbg_id = info.iloc[0]['bbg_id']
    ticker = info.iloc[0]['ticker']
    fs_cur = new_data_['listing_currency'].values[0]
    bbg_cur = new_data_['payment_currency'].values[0]

    return f'''
    >
    >**Fsym Id**: {secid}      |      **BBG Id**: {bbg_id}   |      **Company name**: {name}             
    >**Ticker**: {ticker}      |        **BBG Crncy**: {bbg_cur}        |
    >**FS Crncy**: {fs_cur}
    >
    '''

def compare_new_data_with_factset(secid, update_date, new_data, factset):
    # factset = factset_new_data(secid, last_exdate)
    if factset.shape[0] != 0:
        factset = factset[factset['exdate'] <= update_date]
    bbg = new_data[new_data['fsym_id']==secid].copy()
    comp = pd.merge(bbg,
                    factset,
                    how='outer',
                    on=['fsym_id','exdate', 'div_type'],
                    suffixes=('_bbg','_factset'))
    comp = comp.sort_values(['exdate'])
    # print('compare_new_data_with_factset')
    # print(comp)
    comp = comp.reset_index(drop=True)
    comp = comp[comp.filter(regex='fsym_id|exdate|payment_date|amount').columns]
    comp['check_amount'] = np.where(abs(comp['payment_amount_factset'] -\
                                        comp['payment_amount_bbg'])>0.001,
                                    'Mismatch',
                                    'Good')
    comp['check_payment_date'] = np.where(
        comp['payment_date_factset']!=comp['payment_date_bbg'],
        'Mismatch',
        'Good')
    return comp

# TODO changed function signature, added new_data as an arg
def bulk_upload(df, update_date, factset_df):
    success_list = []
    checked_list = []
    seclist = sorted(list(df['fsym_id'].unique()))
    # div_df = load_bg_div_exdates(seclist)
    print('bulk_upload')
    # total=len(seclist)
    # i=0
    for secid in seclist:
        print(secid)
        # set_progress((str(i + 1), str(total)))
        new = df[df['fsym_id']==secid].copy()
        (last_cur, fstest_cur) = dividend_currency_bulk_upload(secid, df)
        if fstest_cur != last_cur:
            # print(secid + ' | Possible currency change')
            continue
        new['listing_currency'] = fstest_cur
        new['payment_currency'] = fstest_cur
        # print('After cur check')
       
        # last_exdate_df = div_df[div_df['fsym_id'] == secid]
        # if last_exdate_df.shape[0] != 0:
        #     last_exdate = last_exdate_df.iloc[0,1]
        #     # print(last_exdate)
        #     last_exdate = last_exdate.strftime('%Y-%m-%d')
        #     print(last_exdate)

        #     last_exdate = last_exdate if not pd.isna(last_exdate) else None
        # else:
        #     last_exdate = None
        # print(factset_df[factset_df['fsym_id']==secid])
            
        comp = compare_new_data_with_factset(secid, update_date, df, factset_df[factset_df['fsym_id']==secid])
        # print('After compare check')

        if not (comp[['check_amount', 'check_payment_date']]=='Good').all().all():
            print(secid + ' | Mismatch found.')
            continue
        # print('B4 flag check')
        
        check_flag = check_before_upload(new)
        if not check_flag:
            continue
        # print('After flag check')
        checked_list.append(secid)
        success_list.append(new)
        # print(f'Finshed {secid} check')
    if len(success_list) == 0:
        # print('No stocks to be updated')
        # print('before return')
        return (list(set(seclist) ^ set(checked_list)), None)
    df = pd.concat(success_list)
    # print('Prepare to upload')
    print('Uploaded')
    return (list(set(seclist) ^ set(df['fsym_id'].tolist())), df)

def process_skipped(skipped_):
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
#%% Main
