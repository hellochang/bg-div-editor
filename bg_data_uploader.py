# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:32:36 2021

@author: Todd.Liu
"""
from pathlib import Path
#%%
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')
from bg_data_importer import DataImporter
from datetime import datetime
data = DataImporter(verbose=False)
bg_div_columns = ['fsym_id', 'listing_currency', 'payment_currency',
                  'declared_date', 'exdate', 'record_date', 'payment_date',
                  'payment_amount', 'div_type', 'div_freq', 'div_initiation',
                  'skipped']
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

def preprocess_bbg_data(df_, bbgid, index_flag, rdate):
    df = df_.copy()
    dates_cols = df.filter(like='Date').columns
    df[dates_cols] = df[dates_cols].apply(pd.to_datetime, axis=1)
    df.columns=[name.lower().replace(' ','_').replace('-','')
                for name in df.columns]
    df = df.rename(columns={'security' : 'bbg_id'})
    df['bbg_id'] = df['bbg_id'].str.replace('/bbgid/', '')
    rdate = rdate.replace('-', '')
    # bbgid = pd.read_csv(rf'S:\Shared\Scheduled_Jobs\input\sec_list_{rdate}.csv')
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
    df['div_initiation'] = 0
    df['skipped'] = skip_flag
    df['div_type'] = np.where(df['p_divs_s_pd']==1,'special','regular')
    df['div_freq'] = np.nan
    del df['p_divs_s_pd']
    return df

def factset_new_data(secid):
    if secid is None or secid == '':
        return
    last_exdate = last_payment_exdate(secid)
    exdate_condition = ''
    if last_exdate is not None:
        exdate_condition = f"and p_divs_exdate > '{last_exdate}'"
    query = f"""
                select bd.fsym_id, bbg.bbg_id,currency,p_divs_exdate,p_divs_recdatec,p_divs_paydatec,
                    p_divs_pd,p_divs_s_pd from fstest.fp_v2.fp_basic_dividends bd
              left join fstest.sym_v1.sym_bbg bbg on bbg.fsym_id = bd.fsym_id
			  where bd.fsym_id = '{secid}' {exdate_condition}
              """
    df = data.load_data(query)
    df = factset_data_single_security(df)
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

def get_bg_div_data(secid):
    query = f"select * from fstest.dbo.bg_div where fsym_id = '{secid}'"
    bg_data = data.load_data(query)
    return bg_data

def last_payment_exdate(secid):
    query = f"""
                select max(exdate) from fstest.dbo.bg_div where fsym_id = '{secid}'
            """
    last_date = data.load_data(query)
    if pd.isna(last_date.iloc[0,0]):
        return None
    return last_date.iloc[0,0].strftime('%Y-%m-%d')

def dividend_currency(secid, new_data_=None):
    crncy = new_data_[new_data_['fsym_id']==secid]
    last_cur = crncy.iloc[0]['payment_currency']
    fstest_cur = crncy.iloc[0]['listing_currency']
    return (last_cur, fstest_cur)

def check_existence(secid):
    query = f"""
             select 1 from fstest.dbo.bg_div where fsym_id = '{secid}'
    """
    df = data.load_data(query)
    if df.shape[0] == 0:
        return False
    return True

def alternative_currency(secid, show_all=False):
    exdate_condition = ' '
    if not show_all:
        last_exdate = last_payment_exdate(secid)
        # print(last_exdate)
        exdate_condition = f"and p_divs_exdate > '{last_exdate}'"
    query = f"""
                select
                    scc.fsym_id 'alt_fsym_id', scc.currency, scc.fref_listing_exchange,
                    p_divs_exdate, p_divs_paydatec, p_divs_recdatec, p_divs_pd, p_divs_s_pd
                    --, div.payment_amount
            	from FSTest.sym_v1.sym_coverage  sc
            	left join fstest.sym_v1.sym_coverage scc on sc.fsym_security_id = scc.fsym_security_id
                                                        and scc.regional_flag = 1
            	join fstest.fp_v2.fp_basic_dividends bd on bd.fsym_id = scc.fsym_id
            	left join fstest.dbo.BG_Div div on (div.exdate = bd.p_divs_exdate or
                                                    div.payment_date = bd.p_divs_paydatec)
                                                   and div.fsym _id = sc.fsym_id
            	where sc.fsym_id = '{secid}' and scc.currency <> sc.currency
            	and bd.p_divs_s_spinoff = 0 and scc.currency in ('CAD', 'USD') {exdate_condition}
            """

    try:
        df = data.load_data(query)
    except Exception as e:
        print("No Data")
        return None
    df = factset_data_single_security(df)
    return df

def upload_to_db(df):
    print('Uploaded')


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

def basic_info(secid, new_data_):

    query = f"""
            select proper_name, currency, tr.ticker_region 'ticker'
            from fstest.sym_v1.sym_coverage sc
            left join fstest.sym_v1.sym_ticker_region tr on tr.fsym_id = sc.fsym_id
            where sc.fsym_id = '{secid}'
            """
    info = data.load_data(query)
    name = info.iloc[0]['proper_name']
    ticker = info.iloc[0]['ticker']
    fs_cur = new_data_.loc[new_data_['fsym_id']==secid, 'listing_currency'].values[0]
    bbg_cur = new_data_.loc[new_data_['fsym_id']==secid, 'payment_currency'].values[0]

    return f"{secid} | {name} | {ticker} |BBG Crncy: {bbg_cur} | FS Crncy: {fs_cur}"

def compare_new_data_with_factset(secid, update_date, new_data):
    factset = factset_new_data(secid)
    factset = factset[factset['exdate'] <= update_date]
    bbg = new_data[new_data['fsym_id']==secid].copy()
    comp = pd.merge(bbg,
                    factset,
                    how='outer',
                    on=['fsym_id','exdate', 'div_type'],
                    suffixes=('_bbg','_factset'))
    comp = comp.sort_values(['exdate'])
    comp = comp.reset_index(drop=True)
    comp = comp[comp.filter(regex='fsym_id|exdate|payment_date|amount').columns]
    comp['check_amount'] = np.where(abs(comp['payment_amount_factset'] - comp['payment_amount_bbg'])>0.001,
                                    'Mismatch',
                                    'Good')
    comp['check_payment_date'] = np.where(comp['payment_date_factset']!=comp['payment_date_bbg'],
                                          'Mismatch',
                                          'Good')
    return comp

# TODO changed function signature, added new_data as an arg
def bulk_upload(df, update_date, new_data):
    success_list = []
    checked_list = []
    seclist = sorted(list(df['fsym_id'].unique()))
    for secid in seclist:
        new = df[df['fsym_id']==secid].copy()
        (last_cur, fstest_cur) = dividend_currency(secid, df)
        if fstest_cur != last_cur:
            # print(secid + ' | Possible currency change')
            continue
        # new['listing_currency'] = fstest_cur
        # new['payment_currency'] = fstest_cur
        comp = compare_new_data_with_factset(secid, update_date, new_data)
        if not (comp[['check_amount', 'check_payment_date']]=='Good').all().all():
            # print(secid + ' | Mismatch found.')
            continue
        check_flag = check_before_upload(new)
        if not check_flag:
            continue
        checked_list.append(secid)
        success_list.append(new)
    if len(success_list) == 0:
        print('No stocks to be updated')
        return (list(set(seclist) ^ set(checked_list)), None)
    df = pd.concat(success_list)
    # print('Prepare to upload')
    # print('Uploaded')
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
