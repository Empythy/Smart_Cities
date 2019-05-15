import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdat
import matplotlib.ticker as ticker
import pandas as pd
import os
import seaborn as sb

os.chdir(os.getcwd())
# os.chdir('C:\\Users\\Dan Herweg\\PycharmProjects\\Smart_Cities')
plt.style.use('seaborn-darkgrid')

data = pd.read_csv('.\\Smart_Cities\\_csv\\03_Features_Targets.csv')
data.set_index('timestamp', inplace=True)
data.index = pd.to_datetime(data.index)


#viz of AQI
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.xaxis_date()
ax.plot(data.index, data['AQI'], color='darkgreen')


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

#Second viz of AQI
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(data.index, data['AQI'], color='grey')

ax.xaxis_date()

myFmt = mdat.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(myFmt)  # https://stackoverflow.com/questions/14946371/editing-the-date-formatting-of-x-axis-tick-labels-in-matplotlib
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  # https://stackoverflow.com/questions/54057567/matplotlib-uneven-intervals-between-x-axis-with-datetime
## Rotate date labels automatically
fig.autofmt_xdate()

plt.title('AQI in Milan, Nov/Dec 2013')

fig.savefig('.\\Smart_Cities\\_viz\\___AQI_grey.png')
plt.clf()
plt.cla()
plt.close()




feat_smooth = pd.DataFrame(index=data.index)
for i in data.columns:
    if i[0:3] == 'sm_':
        if i[0:3] != 'AQI':
            feat_smooth[str(i)] = data[str(i)]

tar_reg = data.AQI
tar_class = data.AQI_C

# show raw data with imputed values
feat_raw = pd.DataFrame(index=data.index)
for i in data.columns:
    if i[0:3] != 'sm_':
        if i[0:3] != 'AQI':
            feat_raw[str(i)] = data[str(i)]


feat_raw.head()


for i in range(len(feat_raw.columns)):
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(feat_raw.index, feat_raw.iloc[:,i], color='darkgreen', linewidth = .5)
    ax[0].set_title(feat_raw.columns[i]+' Over Time')
    ax[0].locator_params(bins=4)

    myFmt = mdat.DateFormatter('%b-%d')
    ax[0].xaxis.set_major_formatter(myFmt)  # https://stackoverflow.com/questions/14946371/editing-the-date-formatting-of-x-axis-tick-labels-in-matplotlib
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(15))  # https://stackoverflow.com/questions/54057567/matplotlib-uneven-intervals-between-x-axis-with-datetime
    ## Rotate date labels automatically
    fig.autofmt_xdate()

    ax[1].hist(feat_raw.iloc[:,i], bins=30, color='lightgreen')
    ax[1].set_title('Distribution of '+feat_raw.columns[i])
    fig.savefig('.\\Smart_Cities\\_viz\\ts_hist_'+str(feat_raw.columns[i])+'.png')

    plt.clf()
    plt.cla()
    plt.close()



#show raw vs smoothed
for i in data.columns:
    if i[0:3] == 'sm_':
        ind = data.columns.get_loc(i)
        noisy_name = i[3:]

        fig, ax = plt.subplots()
        ax.plot(data.index, noisy_name, data=data, color='palegreen', linewidth = .5)
        ax.plot(data.index, str(i), data=data, color='darkgreen', linewidth = .5)
        ax.xaxis_date()

        myFmt = mdat.DateFormatter('%b-%d')
        ax.xaxis.set_major_formatter(myFmt) #https://stackoverflow.com/questions/14946371/editing-the-date-formatting-of-x-axis-tick-labels-in-matplotlib
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10)) #https://stackoverflow.com/questions/54057567/matplotlib-uneven-intervals-between-x-axis-with-datetime
        ## Rotate date labels automatically
        fig.autofmt_xdate()

        plt.title(noisy_name+' Raw vs. Smoothed')

        fig.savefig('.\\Smart_Cities\\_viz\\' + str(data.columns[ind]) + '.png')
        plt.clf()
        plt.cla()
        plt.close()


#show cross correlations

# within weather (scatter matrices look terrible so heatmaps)
weather = feat_raw[['Relative H', 'Precipitat', 'Wind Speed', 'Atmospheri', 'Net Radiat', 'Wind Direc', 'Global Rad', 'Temperatur',]].copy()

weather_hm = sb.heatmap(weather.corr(),
                        xticklabels=weather.columns,
                        yticklabels=weather.columns,
                        vmin=-1, vmax=1,
                        cmap='PRGn')
plt.title('Weather Correlation Heat Map')
plt.savefig('.\\Smart_Cities\\_viz\\__Weather_Heatmap.png')

plt.show()

plt.clf()
plt.cla()
plt.close()




### scatter of correlated values
verycorrelated = .9

for i in range(len(weather.corr().columns)):
    for j in range(i+1, len(weather.corr().columns)):
        if abs(weather.corr().iloc[i,j]) > verycorrelated:
            corr = round(weather.corr().iloc[i, j], 2)
            plt.scatter(weather.columns[i], weather.columns[j], data= weather, color='lightgreen')
            plt.title(weather.columns[i] +' vs. ' +weather.columns[j] + ', corr: '+ str(corr))
            plt.savefig('.\\Smart_Cities\\_viz\\_____'+ weather.columns[i] +'v' +weather.columns[j]  +'.png')
            plt.clf()
            plt.cla()
            plt.close()

### scatter of correlated values
notverycorrelated = .1

for i in range(len(weather.corr().columns)):
    for j in range(i+1, len(weather.corr().columns)):
        if abs(weather.corr().iloc[i,j]) < notverycorrelated:
            corr = round(weather.corr().iloc[i, j], 2)
            plt.scatter(weather.columns[i], weather.columns[j], data= weather, color='lightgreen')
            plt.title(weather.columns[i] +' vs. ' +weather.columns[j] + ', corr: '+ str(corr))
            plt.savefig('.\\Smart_Cities\\_viz\\_____'+ weather.columns[i] +'v' +weather.columns[j]  +'.png')
            plt.clf()
            plt.cla()
            plt.close()

# within traffic
traffic = feat_raw[['traffic',
       'EURO_3', 'EURO_2', 'EURO_4', 'EURO_5', 'EURO_1', 'EURO_6', 'EURO_7',
       'VType_4', 'VType_3', 'VType_1', 'VType_2', 'FType_1', 'FType_2',
       'FType_4', 'FType_5', 'FType_3', 'DPF_2', 'DPF_1', 'small_len',
       'med_len', '0-15m', '15-30m', '30-45m', '45-60m']].copy()

traffic_hm = sb.heatmap(traffic.corr(),
                        xticklabels=traffic.columns,
                        yticklabels=traffic.columns,
                        vmin=-1, vmax=1,
                        cmap='PRGn')
plt.title('Traffic Correlation Heat Map')
plt.savefig('.\\Smart_Cities\\_viz\\__Traffic_Heatmap.png')

plt.show()

plt.clf()
plt.cla()
plt.close()


### scatter of correlated values (these are all very correlated!
verycorrelated = .9

for i in range(len(traffic.corr().columns)):
    for j in range(i+1, len(traffic.corr().columns)):
        if abs(traffic.corr().iloc[i,j]) > verycorrelated:
            corr = round(traffic.corr().iloc[i, j], 2)
            plt.scatter(traffic.columns[i], traffic.columns[j], data= traffic, color='lightgreen')
            plt.title(traffic.columns[i] +' vs. ' +traffic.columns[j] + ', corr: '+ str(corr))
            plt.savefig('.\\Smart_Cities\\_viz\\_____'+ traffic.columns[i] +'v' +traffic.columns[j]  +'.png')
            plt.clf()
            plt.cla()
            plt.close()



### scatter of correlated values (these are all very correlated!
notverycorrelated = .1

for i in range(len(traffic.corr().columns)):
    for j in range(i+1, len(traffic.corr().columns)):
        if abs(traffic.corr().iloc[i,j]) < notverycorrelated:
            corr = round(traffic.corr().iloc[i, j], 2)
            plt.scatter(traffic.columns[i], traffic.columns[j], data= traffic, color='lightgreen')
            plt.title(traffic.columns[i] +' vs. ' +traffic.columns[j] + ', corr: '+ str(corr))
            plt.savefig('.\\Smart_Cities\\_viz\\_____'+ traffic.columns[i] +'v' +traffic.columns[j]  +'.png')
            plt.clf()
            plt.cla()
            plt.close()





# within pollution
# feat_raw.columns
pollutant = feat_raw[['Sulf', 'Ammo', 'Tota', 'Blac', 'Benz', 'Ozon', 'PM2.', 'PM10', 'Nitr',
       'Carb']].copy()

pollutant_hm = sb.heatmap(pollutant.corr(),
                        xticklabels=pollutant.columns,
                        yticklabels=pollutant.columns,
                        vmin=-1, vmax=1,
                        cmap='PRGn')
plt.title('Pollutant Correlation Heat Map')
plt.savefig('.\\Smart_Cities\\_viz\\__Pollutant_Heatmap.png')

plt.show()

plt.clf()
plt.cla()
plt.close()


### scatter of correlated values todo probably need to remove components of AQI
verycorrelated = .9

for i in range(len(pollutant.corr().columns)):
    for j in range(i+1, len(pollutant.corr().columns)):
        if abs(pollutant.corr().iloc[i,j]) > verycorrelated:
            corr = round(pollutant.corr().iloc[i, j], 2)
            plt.scatter(pollutant.columns[i], pollutant.columns[j], data= pollutant, color='lightgreen')
            plt.title(pollutant.columns[i] +' vs. ' + pollutant.columns[j] + ', corr: '+ str(corr))
            plt.savefig('.\\Smart_Cities\\_viz\\_____'+ pollutant.columns[i] +'v' + pollutant.columns[j]  +'.png')
            plt.clf()
            plt.cla()
            plt.close()


### scatter of correlated values todo probably need to remove components of AQI
notverycorrelated = .2

for i in range(len(pollutant.corr().columns)):
    for j in range(i+1, len(pollutant.corr().columns)):
        if abs(pollutant.corr().iloc[i,j]) < notverycorrelated:
            corr = round(pollutant.corr().iloc[i, j], 2)
            plt.scatter(pollutant.columns[i], pollutant.columns[j], data= pollutant, color='lightgreen')
            plt.title(pollutant.columns[i] +' vs. ' + pollutant.columns[j] + ', corr: '+ str(corr))
            plt.savefig('.\\Smart_Cities\\_viz\\_____'+ pollutant.columns[i] +'v' + pollutant.columns[j]  +'.png')
            plt.clf()
            plt.cla()
            plt.close()


# across all
all_hm = sb.heatmap(feat_raw.corr(),
                        xticklabels=feat_raw.columns,
                        yticklabels=feat_raw.columns,
                        vmin=-1, vmax=1,
                        cmap='PRGn')
plt.title('All Variable Correlation Heat Map')
plt.savefig('.\\Smart_Cities\\_viz\\__All_Heatmap.png')

plt.show()

plt.clf()
plt.cla()
plt.close()

# Make histogram of AQI classes
data.AQI_C.value_counts().plot(kind='bar', color='lightgreen')
plt.title('AQI Categorical Distribution')
plt.tight_layout()
plt.savefig('.\\Smart_Cities\\_viz\\__AQI_Categorical_DIST.png')

plt.show()

plt.clf()
plt.cla()
plt.close()
