import matplotlib.pyplot as plt
import matplotlib.dates as mdat
import matplotlib.ticker as ticker
import pandas as pd
import os
import seaborn as sb


os.chdir(os.getcwd())
# os.chdir('C:\\Users\\Dan Herweg\\PycharmProjects\\Smart_Cities')


data = pd.read_csv('.\\Smart_Cities\\_csv\\03_Features_Targets.csv')
data.set_index('timestamp', inplace=True)
data.index = pd.to_datetime(data.index)
# type(data.index)

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
    # mean = str(round(feat_raw.iloc[:,i].mean(),2))
    # sd = str(round(feat_raw.iloc[:,i].std(),2))
    # plt.title(str(feat_raw.columns[i])+', mean: '+mean+' sd: '+sd)

    ax[0].plot(feat_raw.index, feat_raw.iloc[:,i], color='maroon', linewidth = 1)
    ax[0].set_title(feat_raw.columns[i]+'Over Time')
    ax[0].locator_params(bins=4)

    myFmt = mdat.DateFormatter('%b-%d')
    ax[0].xaxis.set_major_formatter(myFmt)  # https://stackoverflow.com/questions/14946371/editing-the-date-formatting-of-x-axis-tick-labels-in-matplotlib
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(15))  # https://stackoverflow.com/questions/54057567/matplotlib-uneven-intervals-between-x-axis-with-datetime
    ## Rotate date labels automatically
    fig.autofmt_xdate()

    ax[1].hist(feat_raw.iloc[:,i], bins=30, color='lightcoral')
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
        ax.plot(data.index, noisy_name, data=data, color='grey', linewidth = 1)
        ax.plot(data.index, str(i), data=data, color='maroon', linewidth = 2)
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
                        cmap='coolwarm')
plt.title('Weather Correlation Heat Map')
plt.savefig('.\\Smart_Cities\\_viz\\__Weather_Heatmap.png')

plt.show()

plt.clf()
plt.cla()
plt.close()


## plot all together todo (colors are shit, save it)
for w in weather.columns:
    mean = weather[w].mean()
    sd = weather[w].std()
    norm = pd.DataFrame((weather[w]-mean)/sd, columns=[w], index=data.index)
    plt.plot(data.index, w, data=norm, label=w)
plt.title('Normalized Weather Variables')
plt.show()

plt.clf()
plt.cla()
plt.close()


### scatter of correlated values todo color
verycorrelated = .85

for i in range(len(weather.corr().columns)):
    for j in range(i+1, len(weather.corr().columns)):
        if weather.corr().iloc[i,j] > verycorrelated:
            corr = round(weather.corr().iloc[i, j], 2)
            plt.scatter(weather.columns[i], weather.columns[j], data= weather)
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
                        cmap='coolwarm')
plt.title('Traffic Correlation Heat Map')
plt.savefig('.\\Smart_Cities\\_viz\\__Traffic_Heatmap.png')

plt.show()

plt.clf()
plt.cla()
plt.close()


## plot all together todo (colors are shit, needs save, binary vars)
for t in traffic.columns:
    mean = traffic[t].mean()
    sd = traffic[t].std()
    norm = pd.DataFrame((traffic[t]-mean)/sd, columns=[t], index=data.index)
    plt.plot(data.index, t, data=norm, label=t)
plt.title('Normalized Traffic Variables')
plt.show()

plt.clf()
plt.cla()
plt.close()


### scatter of correlated values (these are all very correlated!
verycorrelated = .85

for i in range(len(traffic.corr().columns)):
    for j in range(i+1, len(traffic.corr().columns)):
        if traffic.corr().iloc[i,j] > verycorrelated:
            corr = round(traffic.corr().iloc[i, j], 2)
            plt.scatter(traffic.columns[i], traffic.columns[j], data= traffic)
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
                        cmap='coolwarm')
plt.title('Pollutant Correlation Heat Map')
plt.savefig('.\\Smart_Cities\\_viz\\__Pollutant_Heatmap.png')

plt.show()

plt.clf()
plt.cla()
plt.close()

## plot all together todo (colors are shit, needs save, binary vars)
for t in pollutant.columns:
    mean = pollutant[t].mean()
    sd = pollutant[t].std()
    norm = pd.DataFrame((pollutant[t]-mean)/sd, columns=[t], index=data.index)
    plt.plot(data.index, t, data=norm, label=t)
plt.title('Normalized Pollutant Variables')
plt.show()

plt.clf()
plt.cla()
plt.close()


### scatter of correlated values todo probably need to remove components of AQI
verycorrelated = .85

for i in range(len(pollutant.corr().columns)):
    for j in range(i+1, len(pollutant.corr().columns)):
        if pollutant.corr().iloc[i,j] > verycorrelated:
            corr = round(pollutant.corr().iloc[i, j], 2)
            plt.scatter(pollutant.columns[i], pollutant.columns[j], data= pollutant)
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
                        cmap='coolwarm')
plt.title('All Variable Correlation Heat Map')
plt.savefig('.\\Smart_Cities\\_viz\\__All_Heatmap.png')

plt.show()

plt.clf()
plt.cla()
plt.close()



