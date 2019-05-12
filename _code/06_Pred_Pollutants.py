
import numpy as np
import seaborn as sb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdat
import matplotlib.ticker as ticker
import pandas as pd
import os
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm



os.chdir(os.getcwd())
# os.chdir('C:\\Users\\Dan Herweg\\PycharmProjects\\Smart_Cities')


data = pd.read_csv('.\\Smart_Cities\\_csv\\03_Features_Targets.csv', index_col=0)
# data.set_index('timestamp', inplace=True)
data.index = pd.to_datetime(data.index)
# type(data.index)

# all feats for copy paste
#       ['Sulf', 'Ammo', 'Tota', 'Blac', 'Benz', 'Ozon', 'PM2.', 'PM10', 'Nitr',
#        'Carb', 'Relative H', 'Precipitat', 'Wind Speed', 'Atmospheri',
#        'Net Radiat', 'Wind Direc', 'Global Rad', 'Temperatur', 'sm_Relative H',
#        'sm_Precipitat', 'sm_Wind Speed', 'sm_Atmospheri', 'sm_Net Radiat',
#        'sm_Wind Direc', 'sm_Global Rad', 'sm_Temperatur', 'traffic',
#        'sm_traffic', 'EURO_3', 'EURO_2', 'EURO_4', 'EURO_5', 'EURO_1',
#        'EURO_6', 'EURO_7', 'VType_4', 'VType_3', 'VType_1', 'VType_2',
#        'FType_1', 'FType_2', 'FType_4', 'FType_5', 'FType_3', 'DPF_2', 'DPF_1',
#        'small_len', 'med_len', '0-15m', '15-30m', '30-45m', '45-60m',
#        'sm_Sulf', 'sm_Ammo', 'sm_Tota', 'sm_Blac', 'sm_Benz', 'sm_Ozon',
#        'sm_PM2.', 'sm_PM10', 'sm_Nitr', 'sm_Carb', 'weekend', 'morning',
#        'mid_day', 'evening', 'AQI', 'AQI_C']

fsm_cols = ['sm_Relative H',  'sm_Precipitat', 'sm_Wind Speed', 'sm_Atmospheri', 'sm_Net Radiat', 'sm_Wind Direc', 'sm_Global Rad', 'sm_Temperatur',
            'sm_traffic',
            '0-15m', '15-30m', '30-45m', '45-60m',
            'EURO_3', 'EURO_2', 'EURO_4', 'EURO_5', 'EURO_1', 'EURO_6', 'EURO_7',
            'VType_4', 'VType_3', 'VType_1', 'VType_2',
            'FType_1', 'FType_2', 'FType_4', 'FType_5', 'FType_3',
            'DPF_2', 'DPF_1',
            'small_len', 'med_len',
            'weekend', 'morning', 'mid_day', 'evening']

fsm = pd.DataFrame(index=data.index)

for i in fsm_cols:
    fsm[str(i)] = data[str(i)]

tpol_cols = ['sm_Sulf', 'sm_Ammo', 'sm_Tota', 'sm_Blac', 'sm_Benz', 'sm_Ozon',
                    'sm_PM2.', 'sm_PM10', 'sm_Nitr', 'sm_Carb']

pol_y = pd.DataFrame()

for i in tpol_cols:
    pol_y[str(i)] = data[str(i)]


#stratified Train, test splits

from sklearn.model_selection import StratifiedShuffleSplit


#drop timestamp index bc it messes up strat shuffle split
pol_y.reset_index(inplace=True)
pol_y.drop(columns="timestamp", inplace=True)
fsm.reset_index(inplace=True)
fsm.drop(columns="timestamp", inplace=True)
AQI_strats = pd.DataFrame(data.AQI)
AQI_strats.reset_index(inplace=True)
AQI_strats.drop(columns="timestamp", inplace=True)

labels = [1,2,3,4,5]

AQI_strats["strat"] = pd.qcut(x=AQI_strats.AQI, q=5, labels=labels)

train_pct = .80

#https://blog.usejournal.com/creating-an-unbiased-test-set-for-your-model-using-stratified-sampling-technique-672b778022d5
split = StratifiedShuffleSplit(test_size=(1-train_pct), random_state=5)

for train_idx, test_idx in split.split(AQI_strats, AQI_strats["strat"]):
    y_train = pol_y.loc[train_idx]
    y_test = pol_y.loc[test_idx]
    x_train = fsm.loc[train_idx]
    x_test = fsm.loc[test_idx]

y_train.columns

# Multiple regression for each pollutant y
from sklearn.metrics import mean_squared_error

MLR_pol_pred = pd.DataFrame(columns=pol_y.columns)
MLR_MSE = pd.DataFrame(columns=pol_y.columns)

X = x_train

for pol in y_train.columns:
    y=y_train[pol]
    reg = linear_model.LinearRegression().fit(X=X, y=y)
    predictMLR = reg.predict(np.array(x_test))
    MLR_pol_pred[pol] = predictMLR
    MLR_MSE[pol] = [mean_squared_error(y_true=y_test[pol], y_pred=predictMLR)]

MLR_MSE.transpose()

MLR_pol_pred.index = test_idx

#Turn these predictions into AQI
AQI_calc = pd.DataFrame()
ref_PM10 = 50
ref_NO2 = 200
ref_O3 = 120
PM10_24h = []
Ozone_8h = []
NO_I = []


for i in test_idx:
    for j in range(1,8):
        MA_8h=MLR_pol_pred.sm_Ozon[i]
        try:
            MA_8h = data.sm_Ozon[i-j:i].mean()
        except:
            pass
    Ozone_8h.append(MA_8h)

Ozone_8h = [x/ref_O3 for x in Ozone_8h]
Ozone_8h
AQI_calc['O3_8h_MA'] = Ozone_8h

len(Ozone_8h)
for i in test_idx:
    PM10=MLR_pol_pred.sm_PM10[i]
    PM10_24h.append(PM10)


PM10_24h = [x/ref_PM10 for x in PM10_24h]
AQI_calc['PM10_24h_MA'] = PM10_24h

for i in test_idx:
    NO_I.append(MLR_pol_pred.sm_Nitr[i])
NO_I = [x/ref_NO2 for x in NO_I]
AQI_calc['NO2_I'] = NO_I
# >>> df["C"] = df[["A", "B"]].max(axis=1)


AQI_calc['AQI'] = ((AQI_calc.PM10_24h_MA + AQI_calc[['NO2_I','O3_8h_MA']].max(axis=1)) /2)*100


# drop some columns not needed for prediction
AQI_calc.drop(['NO2_I','O3_8h_MA','PM10_24h_MA'], axis=1, inplace=True)

# make classification target
# df.apply(lambda x : pd.cut(x,[-1,3,6,9],labels=[1,2,3]))
# https://stackoverflow.com/questions/46990467/pandas-categorize-column-values-by-range

cutoffs = [-1,50,100,150,200,np.inf]
labels = ['Good','Acceptable', 'Mediocre', 'Poor', 'Bad']
AQI_calc['AQI_C'] = pd.cut(AQI_calc.AQI, bins=cutoffs,labels=labels)
AQI_calc.index = test_idx

#get accuracy stats
Accuracy = pd.DataFrame(index=MLR_Class.columns, columns=['Accuracy'])
acc = 0
for i in test_idx:
    if AQI_calc.AQI_C[i] == data.AQI_C[i]:
        acc +=1

total = len(test_idx)
acc = acc/total
acc

