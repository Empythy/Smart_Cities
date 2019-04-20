#!/usr/bin/env python
# coding: utf-8

# # Next week I will graduate to PyCharm...
# Dan Herweg | 
# Smart Cities - Professor Rossi | 
# Lab 1 | 
# 11/03/2019

import os
import pandas as pd


# # Area C data

# In[2]:
os.chdir('C:\\Users\\Dan\\PycharmProjects\\Smart_Cities')
# os.getcwd()
###### Area C

##Add column names
#ID, Street, Type, Lat, Long
gates = pd.read_csv('.\\Smart_Cities\\_data\\00_DatiAirQuality\\DatiAirQuality\\MI_Area_C\\gates_hdr.csv')
for i in range(len(gates.columns.values)):
    gates.columns.values[i] = gates.columns.values[i].strip(' ')

#Timestamp, Plate, Gate
transit = pd.read_csv('.\\Smart_Cities\\_data\\00_DatiAirQuality\\DatiAirQuality\\MI_Area_C\\transit_hdr.csv')
for i in range(len(transit.columns.values)):
    transit.columns.values[i] = transit.columns.values[i].strip(' ')

#Plate, EURO, VType, FType, DPF, Length
vehicles = pd.read_csv('.\\Smart_Cities\\_data\\00_DatiAirQuality\\DatiAirQuality\\MI_Area_C\\vehicles_hdr.csv')
for i in range(len(vehicles.columns.values)):
    vehicles.columns.values[i] = vehicles.columns.values[i].strip(' ')

# Join vehicles and transit data
veh_tra = pd.merge(vehicles, transit, how='outer', on= 'Plate',sort=False,copy=True)

# Create final dataframe by joining gates
df_AC = pd.merge(veh_tra, gates, how='outer', left_on= 'Gate',right_on='ID', sort=False,copy=True)


#Check
# df_AC.head()

#Export to CSV and confirm
export_AC = df_AC.to_csv('.\\Smart_Cities\\_data\\01_AC.csv')#sent to data folder bc too big for github push...
print('Traffic Data successfully aggregated and exported')






# # Air Quality data

###### Air Quality
AQ_col_names=['match','sensor','Lat','Long','Pollutant','Units','Frequency']
legend_AQ = pd.read_csv('.\\Smart_Cities\\_data\\00_DatiAirQuality\\DatiAirQuality\\MI_Air_Quality\\data\\pollution-legend-mi.csv',
                       encoding = "ISO-8859-1",
                       names = AQ_col_names,
                        header=None
                       )

# string to add to in loop
filepath_AQ = '.\\Smart_Cities\\_data\\00_DatiAirQuality\\DatiAirQuality\\MI_Air_Quality\\data\\'
poll_name = 'mi_pollution_'


#create column names for loop and tuple of all legend info for final (tuples dont remain so in colnames, not a great choice)
subset = legend_AQ[['Pollutant','Units','sensor', 'match']]
Pol_cols_initial = legend_AQ.match
Pol_cols_final= [tuple(x) for x in subset.values]

# create first df to start merging to
df_AQ = pd.read_csv(filepath_AQ+poll_name+str(Pol_cols_initial[0])+'.csv', header=None, names = ['match','timestamp','value'])
df_AQ.drop(columns=['match','value'], axis=1, inplace=True)

# loop through files matching on timestamp (since there are irregularities to be considered I did not want to remove any data yet)
for i in range(len(Pol_cols_initial)):
    temp_file = pd.read_csv(filepath_AQ+poll_name+str(Pol_cols_initial[i])+'.csv', header=None, names = ['match','timestamp','value'])
    col_name = temp_file.match[1]
    temp_file = temp_file.drop(columns = 'match')
    temp_file.columns.values[1] = col_name
    
    df_AQ = pd.merge(df_AQ, temp_file, how= 'outer', on='timestamp', sort=True,copy=True)
    
df_AQ.set_index('timestamp', inplace=True)



#make column headers contain legend info
df_AQ.columns=Pol_cols_final


#check
df_AQ.head()


#export air quality data to csv and confirm
export_AQ = df_AQ.to_csv('Smart_Cities\\_csv\\01_AQ.csv')
print('Air Quality Data successfully aggregated and exported')




###### Weather Station
WS_col_names=['match','sensor','Lat','Long','value','units']
legend_WS = pd.read_csv('Smart_Cities\\_data\\00_DatiAirQuality\\DatiAirQuality\\MI_Weather_Station_Data\\data\\mi_meteo_legend.csv',
                        encoding = "ISO-8859-1",
                        names = WS_col_names,
                        header=None
                        )

# string to add legend key to
filepath_WS = 'Smart_Cities\\_data\\00_DatiAirQuality\\DatiAirQuality\\MI_Weather_Station_Data\\data\\'
wea_name = 'mi_meteo_'


#create column names for loop and tuple of all legend info for final
subset = legend_WS[['value','units','sensor', 'match']]
Wea_cols_initial = legend_WS.match
Wea_cols_final= [tuple(x) for x in subset.values]


# create first df to start merging to
df_WS = pd.read_csv(filepath_WS+wea_name+str(Wea_cols_initial[0])+'.csv', header=None, names = ['match','timestamp','value'])
df_WS.drop(columns=['match','value'], inplace=True)

# loop through files matching on timestamp (since there are irregularities to be considered I did not want to remove any data yet)
for i in range(len(Wea_cols_initial)):
    temp_file = pd.read_csv(filepath_WS+wea_name+str(Wea_cols_initial[i])+'.csv', header=None, names = ['match','timestamp','value'])
 
    #some have headers to standardize
    if str(temp_file.match[0]).strip(' ') =='ID':
        temp_file = pd.read_csv(filepath_WS+wea_name+str(Wea_cols_initial[i])+'.csv',  header = 0, names = ['match','timestamp','value'])
    
    col_name = temp_file.match[1]
    temp_file = temp_file.drop(columns = 'match')
    temp_file.columns.values[1] = col_name
    
    df_WS = pd.merge(df_WS, temp_file, how= 'outer', on='timestamp', sort=True,copy=True)
    
df_WS.set_index('timestamp', inplace=True)

#make column headers contain legend info
df_WS.columns = Wea_cols_final


#check
df_WS.head()



#export air quality data to csv and confirm
export_WS = df_WS.to_csv('.\\Smart_Cities\\_csv\\01_WS.csv')
print('Weather Data successfully aggregated and exported')