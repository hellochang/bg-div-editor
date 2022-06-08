# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:49:27 2022

@author: Chang.Liu
"""
import pandas as pd

path1 = 'S:/Scheduled_Jobs/output/new_dvd_data_20211231.parquet'
path2 = '\\bgndc\Analyst\S:\Scheduled_Jobs\output\\new_dvd_data_20211231.parquet'
# path3 = r'C:\Users\Chang'+'.Liu\Documents\dev\data_update_checker\\output\new_dvd_data_20211231.parquet'
path3 = r'C:\Users\Chang.Liu\Documents\dev\data_update_checker\output\new_dvd_data_20211231.parquet'
path4 =r'\\bgndc\Analysts\Scheduled_Jobs\output\new_dvd_data_20211231.parquet'
new_data = pd.read_parquet(path4)
print(new_data)

