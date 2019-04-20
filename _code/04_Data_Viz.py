import matplotlib.pyplot as plt
import pandas as pd
import os


os.chdir('C:\\Users\\Dan\\PycharmProjects\\Smart_Cities')


data = pd.read_csv('.\\Smart_Cities\\_csv\\03_Features_Targets.csv')
data.set_index('timestamp', inplace=True)


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
    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    # mean = str(round(feat_raw.iloc[:,i].mean(),2))
    # sd = str(round(feat_raw.iloc[:,i].std(),2))
    # plt.title(str(feat_raw.columns[i])+', mean: '+mean+' sd: '+sd)

    ax[0].plot(feat_raw.index, feat_raw.iloc[:,i])
    ax[0].set_title(feat_raw.columns[i]+'Over Time')
    ax[0].locator_params(bins=4)
    ax[1].hist(feat_raw.iloc[:,i], bins=30)
    ax[1].set_title('Distribution of '+feat_raw.columns[i])
    plt.savefig('.\\Smart_Cities\\_viz\\ts_hist_'+str(feat_raw.columns[i])+'.png')

#show raw vs smoothed
for i in data.columns:
    if i[0:3] == 'sm_':
        plt.clf()
        ind = data.columns.get_loc(i)
        plt.plot(data.index, str(i), data=data)
        noisy_name = i[3:]
        plt.plot(data.index, noisy_name, data=data)
        plt.locator_params(bins=4)
        plt.savefig('.\\Smart_Cities\\_viz\\' + str(data.columns[ind]) + '.png')


#show cross correlations

