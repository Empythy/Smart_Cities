# this file was converted from a jupyter notebook.  It cleans the data by imputing NAs, resamples into the correct timeframe



# Dan Herweg | 
# Smart Cities - Professor Rossi | 
# Lab 2 | 
# 1/03/2019
import gc
import statsmodels.tsa.api as ts
import pandas as pd
import missingno as msno
import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline') #artifact of notebook
import missingpy as mp
import numpy as np
import datetime as dt
import os
os.chdir('C:\\Users\\Dan Herweg\\Pycharm_Projects\\Smart_Cities')
gc.collect()

#grab all files from aggregation
df_AC = pd.read_csv('.\\Smart_Cities\\_data\\01_AC.csv')
df_AQ = pd.read_csv('.\\Smart_Cities\\_csv\\01_AQ.csv')
df_WS = pd.read_csv('.\\Smart_Cities\\_csv\\01_WS.csv')


# # Import, aggregate, average, fill NA

# ### Air quality sensors (pollutants)
#     - Fill NA with either interpolation or data imputing methods using the other pollutants as features (try to average across staion or to use only sensors within the same station)
#     - Do not smooth this

#viz of missing values
missingdata_df = df_AQ.columns[df_AQ.isnull().any()].tolist()
# msno.matrix(df_AQ[missingdata_df])


#Deal separately with hourlys and dailys
Hourlys = pd.concat([df_AQ.iloc[:,0:25],df_AQ.iloc[:,30:]], axis=1) #ALL HOURLY DATA
Hourlys.dropna(axis=0, how='all', thresh=2, inplace=True) #get rid of daily NAs
#Hourlys.columns
Hourlys.timestamp = pd.to_datetime(Hourlys.timestamp)
Hourlys.drop_duplicates(inplace=True)

gc.collect()

#redo index
start = min(df_AQ.timestamp)
end = max(df_AQ.timestamp)
new_index = pd.date_range(start=start, end=end, freq='1H', name='timestamp')
new_index = pd.DataFrame(new_index)
new_index.set_index([new_index.timestamp])
new_index.head(5)


#redo merge of hourly data to index
Hourlys = pd.merge(new_index, Hourlys, on='timestamp', how='left', sort=False,copy=True)
Hourlys.set_index(Hourlys.timestamp, inplace=True)
Hourlys = Hourlys.loc[:, Hourlys.columns != 'timestamp']

#viz of missing values
missingdata_df2 = Hourlys.columns[Hourlys.isnull().any()].tolist()
# msno.matrix(Hourlys[missingdata_df2])

gc.collect()

# predict Missing hourly pollutants from present pollutants with random forest
imputer = mp.MissForest()
polluted_imputed = imputer.fit_transform(Hourlys)
# polluted_imputed


#put imputed data in a data frame
Hourlys_cols = Hourlys.columns.values
Hourlys = pd.DataFrame(polluted_imputed, columns = Hourlys_cols, index=Hourlys.index )
Hourlys = Hourlys.reindex(sorted(Hourlys.columns), axis=1)
# Hourlys.head(3)
# Hourlys.shape

gc.collect()


# group hourly data by daily averages to merge with daily pollutants
Avg_H2D = Hourlys
Avg_H2D = Hourlys.resample('D').mean()
Avg_H2D.shape


# get daily data in daily df
Dailys = pd.concat([df_AQ.iloc[:,0],df_AQ.iloc[:,25:30]], axis=1)
Dailys.dropna(axis=0, how='all', thresh=2, inplace=True)
Dailys.timestamp = pd.to_datetime(Dailys.timestamp)
Dailys.set_index(Dailys.timestamp, inplace=True)
Dailys = Dailys.loc[:, Dailys.columns != 'timestamp']

# Dailys


#merge daily data with average daily data from hourly data
Dailys2 = pd.merge(Dailys, Avg_H2D, on='timestamp', how='left', sort=False,copy=True)

# Dailys2.head(3)
# Dailys2.shape


# imputs daiily gaps in one line this time
polluted_imputed_D = pd.DataFrame(imputer.fit_transform(Dailys2), columns=Dailys2.columns, index=Dailys2.index)
polluted_imputed_D


#put these imputations before the rest of the data
Dailys2.index = Dailys2.index - dt.timedelta(days=10000)
Dailys2.head()

gc.collect()

# putting daily values with the hourly to use them to impute, then getting rid of them
FeelsWrong = Hourlys.append(Dailys2, sort=True)
# FeelsWrong.head(3)
# FeelsWrong.shape


#impute
imputed_all = pd.DataFrame(imputer.fit_transform(FeelsWrong), columns=FeelsWrong.columns, index=FeelsWrong.index)
# imputed_all.shape


#sort so truncate function works
imputed_all.sort_index(inplace=True)
# imputed_all.shape


#drop daily values
start = Dailys.index.min()
imputes = imputed_all.truncate(before=start)
# imputes.shape


# group by pollutant averages
short_cols = [i[2:6] for i in imputes.columns]
imputes.columns = short_cols
col_list = list(imputes.columns)

del Dailys
del Dailys2
del imputed_all
del imputes
gc.collect()


#Average the pollutants over the sensors
Avg_H = pd.DataFrame()

for p in set(short_cols):
    tempdf = pd.DataFrame()
    first = col_list.index(p)
    last = len(col_list) - col_list[::-1].index(p) - 1 #https://stackoverflow.com/questions/522372/finding-first-and-last-index-of-some-value-in-a-list-in-python
    tempdf = Hourlys.iloc[:, first:last+1]
    Avg_H[p] = tempdf.mean(axis=1)

# Avg_H.head()
# Avg_H.shape

del tempdf
del Hourlys
gc.collect()


# ### Weather sensors (meteo)
#     - Fill NA with data imputing methods (use random forest -> https://pypi.org/project/missingpy/)
#     - Create also smoothed columns (e.g using exponential filtering with alpha=0.2)
# 


missingdata_df3 = df_WS.columns[df_WS.isnull().any()].tolist()
# msno.matrix(df_WS[missingdata_df3])


# get index right
df_WS.timestamp = pd.to_datetime(df_WS.timestamp)
df_WS.set_index(df_WS.timestamp, inplace=True)
df_WS = df_WS.loc[: , df_WS.columns != 'timestamp']


#impute meteorological variables on other meteorological varbailes
imputed_WS = pd.DataFrame(imputer.fit_transform(df_WS), columns=df_WS.columns, index=df_WS.index)

del df_WS
gc.collect()

imputed_WS = imputed_WS.reindex(sorted(imputed_WS.columns), axis=1)
imputed_WS.columns


# group by weather averages and add in exponentially smoothed versions
short_cols = [i[2:12] for i in imputed_WS.columns]

imputed_WS2 = imputed_WS
imputed_WS2.columns = short_cols


col_list = list(imputed_WS2.columns)
col_list
Avg_WS = pd.DataFrame()
w_vars = list(set(short_cols))

for w in w_vars:
    tempdf = pd.DataFrame()
    first = col_list.index(w)
    last = len(col_list) - col_list[::-1].index(w) - 1 #https://stackoverflow.com/questions/522372/finding-first-and-last-index-of-some-value-in-a-list-in-python
    tempdf = imputed_WS2.iloc[:, first:last+1]
    Avg_WS[w] = tempdf.mean(axis=1)
    

# Avg_WS

# make smoothed versions
for s in Avg_WS.columns:
    
  # https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html
    fit = ts.SimpleExpSmoothing(Avg_WS[s]).fit(smoothing_level=0.2, optimized=False)
    fcast = fit.predict(start=imputed_WS.index.min(), end=imputed_WS.index.max())
    smoothed_name = 'sm_' + s
    Avg_WS[smoothed_name] = fcast

# Avg_WS.columns

del imputed_WS
del imputed_WS2

# Merge in one table... they are not same length yet!!!
AQ_WS  = pd.merge(Avg_H, Avg_WS, on='timestamp')
# AQ_WS.isnull().sum()
# Avg_WS.shape
del Avg_H
del Avg_WS

gc.collect()

# ## Import traffic and vehicle data (import and merge)
#     - Count passages (eventually use sub hour resolution and then create additional colums for each sub-hour)
#     - Create also smoothed columns
#     - Merge with table containing weather and air pollutants using the datetime as key
# 



#remove annoying capital T and conver to dt
df_AC['timestamp'] = df_AC.Timestamp
df_AC.timestamp = pd.to_datetime(df_AC.timestamp)
del(df_AC['Unnamed: 0'])



df_AC.set_index(df_AC.timestamp, inplace=True)
df_AC = df_AC.loc[:,df_AC.columns != 'Timestamp']
df_AC = df_AC.loc[:,df_AC.columns != 'timestamp']




# create traffic totals

AC_2H = df_AC.Plate
AC_2H = AC_2H.resample('H').count()
AC_2H = pd.DataFrame(AC_2H)


AC_2H.columns = ['traffic']
AC_2H.index.min()



#smooth traffic totals
fit = ts.SimpleExpSmoothing(AC_2H.traffic).fit(smoothing_level=0.2, optimized=False)
fcast = fit.predict(start=AC_2H.index.min(), end=AC_2H.index.max())
smoothed_name = 'sm_traffic'
AC_2H[smoothed_name] = fcast
# AC_2H



# binary for euros, vtypes, ftypes, DFP,  Length will be 'eyeballed'
AC_H = pd.DataFrame(index=df_AC.index)
for i in range(1,5):
    for j in df_AC.iloc[:,i].unique():
        colname = str(df_AC.columns[i]) +'_'+str(j)
        AC_H[colname] = np.where(df_AC.iloc[:,i]==j,1,0)
AC_H = AC_H.resample('H').sum()
AC_H.head()



# get length distribution and decide how to split it
import matplotlib.pyplot as plt
checklength = df_AC.Length[df_AC.Length!=0]
plt.hist(x=checklength, bins=100, color='#0504aa', alpha=0.7, rwidth=0.85)
plt.title('Car Length Distribution')
plt.xlim(0,7500)
plt.savefig('.\\Smart_Cities\\_viz\\Car_Length_Distribution.png')

#these seem like good divisions between small, medium and large lengths
sm_med = 3000
med_lg = 6000


df_len = pd.DataFrame(df_AC.Length)
AC_HH = pd.DataFrame(index=df_AC.index)
AC_HH['small_len'] = np.where((df_len.Length >0) & (df_len.Length <= sm_med), 1,0)
AC_HH['med_len'] = np.where((df_len.Length > sm_med) & (df_len.Length <= med_lg),1,0)
AC_HH['lg_len'] = np.where((df_len.Length > med_lg ),1,0) #this is dropped later...

AC_HH = AC_HH.resample('H').sum()
# AC_HH.head()


#make day of week dummys, to be subbed out later but still need this
AC_DATE = pd.DataFrame(index=AC_HH.index)
AC_DATE['ts'] = AC_DATE.index#.Series.dt.dayofweek
AC_DATE['DOW'] = AC_DATE.ts.dt.dayofweek
AC_DATE

daynames = ['mon','tue','wed','thu','fri','sat','sun']

for i in range(len(daynames)):
    colname = daynames[i]
    AC_DATE[colname] = np.where(AC_DATE.DOW == i, 1,0)


# make hour dummies, these will also be dropped but time of day data needs to use these columns
AC_DATE['HOUR'] = AC_DATE.ts.dt.hour

for i in range(24):
    colname = str(i) + ':00'
    AC_DATE[colname] = np.where(AC_DATE.HOUR == i, 1,0)



# AC_['small_len'] = np.where((df_len.Length >0) & (df_len.Length <= sm_med), 1,0)

# dt.datetime.weekday(AC_HH.index[0])
# AC_DATE.new_index.dt.day_name()



#delete columns not needed
del(AC_DATE['DOW'])
del(AC_DATE['HOUR'])
del(AC_DATE['ts'])
AC_DATE



# do the 15 min thing

AC_15M = df_AC.Plate
AC_15M = AC_15M.resample('15min').count()
AC_15M = pd.DataFrame(AC_15M)

# https://stackoverflow.com/questions/25055712/pandas-every-nth-row

AC_15M_toH = pd.DataFrame() 
# quarterHours = [15, 30, 45, 60]
for i in range(4):
    colname=str(i*15)+'-'+str((i+1)*15) + 'm'
    AC_15M_toH[colname] = list(AC_15M.iloc[i::4,0])

AC_15M_toH.index = AC_H.index

AC_15M_toH.head()



# merge all AC data

# https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns
from functools import reduce
dfs = [AC_2H, AC_H, AC_HH, AC_DATE, AC_15M_toH]  #these names are awful I need to change them...
AC_FINAL = reduce(lambda left,right: pd.merge(left,right,on='timestamp'), dfs)
AC_FINAL.columns



# drop all the zero columns since they are not applicable
dropcols = ['EURO_0', 'VType_0', 'FType_0', 'DPF_0']
AC_FINAL.drop(dropcols, axis=1, inplace=True)

AC_FINAL.columns



# final merge
ALL_FEATURES = pd.merge(AQ_WS, AC_FINAL, on='timestamp', sort=True, copy=True)


ALL_FEATURES.head()



# create csv
export_LAB2 = ALL_FEATURES.to_csv('.\\Smart_Cities\\_csv\\02_ALL_FEATURES.csv')
print('Successfully cleaned, part I')

