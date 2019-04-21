
################ CHECK FOR APPROPRIATENESS OF DUMMY/CONTINUOUS

########### also maybe hold out components of AQI, then predict them after

######## potential regressions
# smooth predict y -> class
# smooth predict class
# smooth predict AQI components -> class (is this the same as 1?)
# sub in raw to see if better fits
# validate on an ongoing basis, last 20%, LOOCV?


import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.dates as mdat
import matplotlib.ticker as ticker
import pandas as pd
import os


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
            'sm_Sulf', 'sm_Ammo', 'sm_Tota', 'sm_Blac', 'sm_Benz', 'sm_Ozon', 'sm_PM2.', 'sm_PM10', 'sm_Nitr', 'sm_Carb',
            'weekend', 'morning', 'mid_day', 'evening']

fsm = pd.DataFrame(index=data.index)

for i in fsm_cols:
    fsm[str(i)] = data[str(i)]

reg_y = data.AQI
class_y = data.AQI_C

# show raw data with imputed values
fraw_cols= ['Sulf', 'Ammo', 'Tota', 'Blac', 'Benz', 'Ozon', 'PM2.', 'PM10', 'Nitr', 'Carb',
            'Relative H', 'Precipitat', 'Wind Speed', 'Atmospheri', 'Net Radiat', 'Wind Direc', 'Global Rad', 'Temperatur',
            'traffic',
            '0-15m', '15-30m', '30-45m', '45-60m',
            'EURO_3', 'EURO_2', 'EURO_4', 'EURO_5', 'EURO_1', 'EURO_6', 'EURO_7',
            'VType_4', 'VType_3', 'VType_1', 'VType_2',
            'FType_1', 'FType_2', 'FType_4', 'FType_5', 'FType_3',
            'DPF_2', 'DPF_1',
            'small_len', 'med_len',
            'weekend', 'morning', 'mid_day', 'evening']

fraw = pd.DataFrame(index=data.index)
for i in fraw_cols:
    fraw[str(i)] = data[str(i)]

################# Use smoothed data
x = fsm

#Train, test, validation splits
train_pct = .50
valid_set_start = train_pct + ((1-train_pct)/2)

train_stop_idx = int(len(data)*train_pct - 12)
test_start_idx = train_stop_idx+12 #accounts for 24h influence on AQI
valid_start_idx = int(len(data)*valid_set_start)

xTrain = x.iloc[:train_stop_idx,:]
xTest = x.iloc[test_start_idx:valid_start_idx,:]
xValid = x.iloc[valid_start_idx:,:]

yTrain = reg_y.iloc[:train_stop_idx]

yTest = reg_y.iloc[test_start_idx:valid_start_idx]
yTest_class = class_y.iloc[test_start_idx:valid_start_idx]

yValid = reg_y.iloc[valid_start_idx:]
yValid_class = class_y.iloc[valid_start_idx:]
yValid_class


#THis is an issue - todo only 4 values in test set, 3 in valid with 0.8
class_y.nunique()
yTest_class.nunique()
yValid_class.nunique()

# OUTLINE - for all save predictions vs classified targets
# Single regressions for baseline
from sklearn import linear_model
Sing_LR = pd.DataFrame(index=xTest.index)

for i in xTrain.columns:
    X=np.array(xTrain[i]).reshape(-1, 1)
    y=np.array(yTrain).reshape(-1, 1)
    reg = linear_model.LinearRegression().fit(X=X, y=y)
    predictSLR = reg.predict(np.array(xTest[i]).reshape(-1, 1))
    colname = 'SLR_'+str(i)
    Sing_LR[colname] = list(predictSLR)

Sing_LR.head()

# make regressions into classifications
cutoffs = [-1,50,100,150,200,np.inf]
labels = ['Good','Acceptable', 'Mediocre', 'Poor', 'Bad']

SLR_Class = Sing_LR.copy()

for i in SLR_Class.columns:
    SLR_Class[i] = pd.cut(SLR_Class[i], bins=cutoffs, labels=labels)


#get accuracy stats
SLR_acc = pd.DataFrame(index=SLR_Class.columns, columns=['Accuracy'])
for col in range(len(SLR_Class.columns)):
    acc =0
    for row in range(len(SLR_Class.iloc[col])):
        if SLR_Class.iloc[row,col] == yTest_class.iloc[row]:
            acc +=1
    total = len(SLR_Class.iloc[col])
    acc = acc/total
    SLR_acc.iloc[col,0]=acc

SLR_acc.sort_values(by='Accuracy', ascending=False, inplace=True)
SLR_acc.head(10)





# Multiple regression
    # full
Mult_LR = pd.DataFrame(index=xTest.index)

X=np.array(xTrain)
y=np.array(yTrain).reshape(-1, 1)
reg = linear_model.LinearRegression().fit(X=X, y=y)
predictMLR = reg.predict(np.array(xTest))
colname = 'MLR_ALL'
Mult_LR[colname] = predictMLR

reg = linear_model.Ridge().fit(X=X, y=y)
predictMLR = reg.predict(np.array(xTest))
colname = 'MLR_Ridge'
Mult_LR[colname] = predictMLR

reg = linear_model.Lasso().fit(X=X, y=y)
predictMLR = reg.predict(np.array(xTest))
colname = 'MLR_Lasso'
Mult_LR[colname] = predictMLR

reg = linear_model.ElasticNet().fit(X=X, y=y)
predictMLR = reg.predict(np.array(xTest))
colname = 'MLR_ElasticNet'
Mult_LR[colname] = predictMLR

Mult_LR.head()


# Make classes (are teh regression models superfluous if we can just use classificantion versions?)
MLR_Class = Mult_LR.copy()
for i in MLR_Class.columns:
    MLR_Class[i] = pd.cut(MLR_Class[i], bins=cutoffs, labels=labels)

MLR_Class.head()


#get accuracy stats
MLR_acc = pd.DataFrame(index=MLR_Class.columns, columns=['Accuracy'])
for col in range(len(MLR_Class.columns)):
    acc =0
    for row in range(len(MLR_Class.iloc[col])):
        if MLR_Class.iloc[row,col] == yTest_class.iloc[row]:
            acc +=1
    total = len(MLR_Class.iloc[col])
    acc = acc/total
    MLR_acc.iloc[col,0]=acc

MLR_acc.sort_values(by='Accuracy', ascending=False, inplace=True)
MLR_acc.head(10)
yTest_class.unique().sum()
SLR_acc.head(10)

yTest_class.head(15)
MLR_Class.head(15)




#Validation?  multiple models tested but were they tuned?

# Multiple regression
    # full
vMult_LR = pd.DataFrame(index=xValid.index)

X=np.array(xTrain)
y=np.array(yTrain).reshape(-1, 1)
reg = linear_model.LinearRegression().fit(X=X, y=y)
predictMLR = reg.predict(np.array(xValid))
colname = 'MLR_ALL'
vMult_LR[colname] = predictMLR

reg = linear_model.Ridge().fit(X=X, y=y)
predictMLR = reg.predict(np.array(xValid))
colname = 'MLR_Ridge'
vMult_LR[colname] = predictMLR

reg = linear_model.Lasso().fit(X=X, y=y)
predictMLR = reg.predict(np.array(xValid))
colname = 'MLR_Lasso'
vMult_LR[colname] = predictMLR

reg = linear_model.ElasticNet().fit(X=X, y=y)
predictMLR = reg.predict(np.array(xValid))
colname = 'MLR_ElasticNet'
vMult_LR[colname] = predictMLR

# Make classes (are teh regression models superfluous if we can just use classificantion versions?)
vMLR_Class = vMult_LR.copy()
for i in vMLR_Class.columns:
    vMLR_Class[i] = pd.cut(vMLR_Class[i], bins=cutoffs, labels=labels)

vMLR_acc = pd.DataFrame(index=MLR_Class.columns, columns=['Accuracy'])

for col in range(len(vMLR_Class.columns)):
    acc =0
    for row in range(len(vMLR_Class.iloc[col])):
        if vMLR_Class.iloc[row,col] == yTest_class.iloc[row]:
            acc +=1
    total = len(vMLR_Class.iloc[col])
    acc = acc/total
    vMLR_acc.iloc[col,0]=acc

MLR_acc.head(10)

    # feature selection backward, forward, stepwise


    # PCA
    # SVD

