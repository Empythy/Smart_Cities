
#This file fixes some things the professor suggested and creates the Air Quality Index target variable

import os
import statsmodels.tsa.api as ts
import pandas as pd
import matplotlib.dates as mdat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

# os.chdir(os.getcwd())
os.chdir('C:\\Users\\Dan Herweg\\Pycharm_Projects\\Smart_Cities')

data = pd.read_csv('.\\Smart_Cities\\_csv\\02_ALL_FEATURES.csv')
data.index = data.timestamp
data.index = pd.to_datetime(data.index)
data.rename(columns = {'timestamp':'delete'}, inplace=True)
# data.columns



# smooth pollutant totals
pol_cols = data.columns[1:11]
for s in pol_cols:
    # https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html
    fit = ts.SimpleExpSmoothing(data[s]).fit(smoothing_level=0.2, optimized=False)
    fcast = fit.predict(start=data.index.min(), end=data.index.max())
    smoothed_name = 'sm_' + s
    data[smoothed_name] = fcast

data.columns


# remove final variable of lengths since they are all accounted for and you only need n-1 dummy vars
data.drop(columns='lg_len', inplace=True)

# make weekend/weekday
weekend_days = ['sat', 'sun']
data['weekend'] = data[weekend_days].sum(axis=1)
# data.weekend

weekdays = ['mon', 'tue', 'wed', 'thu', 'fri']
#data['weekday'] = data[weekdays].sum(axis=1) #dont need dummy variables that sum to 1
# data.weekday

data.drop(columns=weekdays, inplace=True)
data.drop(columns=weekend_days, inplace=True)

# make morning, mid_day, evening, night
hour_cols = data.columns[49:73]
morning = hour_cols[5:11]
mid_day = hour_cols[11:15]
evening = hour_cols[15:20]
night = hour_cols[20:]
night = night.append(hour_cols[:5])
# night

data['morning'] = data[morning].sum(axis=1)
data['mid_day'] = data[mid_day].sum(axis=1)
data['evening'] = data[evening].sum(axis=1)
# data['night'] = data[night].sum(axis=1) dummy vars that sum to 1 not needed

data.drop(columns=hour_cols, inplace=True)

data.drop(columns='delete', axis=1, inplace=True)
# data.head()


# create target variable AQI
ref_PM10 = 50
ref_NO2 = 200
ref_O3 = 120
PM10_24h = []
Ozone_8h = []

for i in range(len(data.sm_Ozon)):
    try:
        MA_8h = data.Ozon[i-8:i].mean()
        Ozone_8h.append(MA_8h)
    except:
        Ozone_8h.append(np.nan)


Ozone_8h = [x/ref_O3 for x in Ozone_8h]
Ozone_8h
data['O3_8h_MA'] = Ozone_8h



data.columns
for i in range(len(data.sm_PM10)):
    try:
        MA_8h = data.PM10[i-24:i].mean()
        PM10_24h.append(MA_8h)
    except:
        PM10_24h.append(np.nan)


PM10_24h = [x/ref_PM10 for x in PM10_24h]
data['PM10_24h_MA'] = PM10_24h

NO_I = [x/ref_NO2 for x in data.sm_Nitr]
data['NO2_I'] = NO_I
# >>> df["C"] = df[["A", "B"]].max(axis=1)
data.PM10
data['AQI'] = ((data.PM10_24h_MA + data[['NO2_I','O3_8h_MA']].max(axis=1)) /2)*100


#viz of AQI
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(data.index, data['AQI'], color='maroon')

ax.xaxis_date()

myFmt = mdat.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(myFmt)  # https://stackoverflow.com/questions/14946371/editing-the-date-formatting-of-x-axis-tick-labels-in-matplotlib
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  # https://stackoverflow.com/questions/54057567/matplotlib-uneven-intervals-between-x-axis-with-datetime
## Rotate date labels automatically
fig.autofmt_xdate()

plt.title('AQI in Milan, Nov/Dec 2013')

fig.savefig('.\\Smart_Cities\\_viz\\___AQI.png')
plt.clf()
plt.cla()
plt.close()


# drop some columns not needed for prediction
data.drop(['NO2_I','O3_8h_MA','PM10_24h_MA'], axis=1, inplace=True)

# make classification target
# df.apply(lambda x : pd.cut(x,[-1,3,6,9],labels=[1,2,3]))
# https://stackoverflow.com/questions/46990467/pandas-categorize-column-values-by-range

cutoffs = [-1,50,100,150,200,np.inf]
labels = ['Good','Acceptable', 'Mediocre', 'Poor', 'Bad']
data['AQI_C'] = pd.cut(data.AQI, bins=cutoffs,labels=labels)

# drop first day bc we need MAs
data = data.iloc[24:,]

data.isnull().sum().sum()
file = data.to_csv('.\\Smart_Cities\\_csv\\03_Features_Targets.csv')




