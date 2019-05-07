
################ CHECK FOR APPROPRIATENESS OF DUMMY/CONTINUOUS

######## potential regressions
# todo next
#   NN
#   1 subset selection-least correlated, lasso, most correlated w target, stepwise selection
#   use raw
#   predict componentsù
#   stratify train and test to balance the test set, bin variables because sklearn only does it by class
#      https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
#   get MSE on AQI



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

reg_y = data.AQI
class_y = data.AQI_C

# show raw data with imputed values
fraw_cols= ['Relative H', 'Precipitat', 'Wind Speed', 'Atmospheri', 'Net Radiat', 'Wind Direc', 'Global Rad', 'Temperatur',
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

#Train, test splits
train_pct = .70

train_stop_idx = int(len(data)*train_pct - 12)
test_start_idx = train_stop_idx+12 #accounts for 24h influence on AQI

xTrain = x.iloc[:train_stop_idx,:]
xTest = x.iloc[test_start_idx:,:]

yTrain = reg_y.iloc[:train_stop_idx]
yTrain_class = class_y.iloc[:train_stop_idx]

yTest = reg_y.iloc[test_start_idx:]
yTest_class = class_y.iloc[test_start_idx: ]


# distribution of classes in train vs test
Train_dist = {}
for i in yTrain_class.unique():
    Train_dist[i] = 0
for i in yTrain_class:
    Train_dist[i] += 1

# plot dictionary
fig, ax = plt.subplots()
plt.bar(*zip(*Train_dist.items()), color='lightgreen')  #https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary
plt.title('Distribution of Training Set Classes')
fig.savefig('.\\Smart_Cities\\_viz\\_Train_Distribution.png')
plt.clf()
plt.cla()
plt.close()

Test_dist = {}
for i in yTrain_class.unique():  #use train to get the same order
    Test_dist[i] = 0
for i in yTest_class:
    Test_dist[i] += 1

# plot dictionary
fig, ax = plt.subplots()
plt.bar(*zip(*Test_dist.items()), color='lightgreen')  #https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary
plt.title('Distribution of Testing Set Classes')
fig.savefig('.\\Smart_Cities\\_viz\\_Test_Distribution.png')
plt.clf()
plt.cla()
plt.close()

All_dist = {}
for i in yTrain_class.unique():  #use train to get the same order
    All_dist[i] = 0
for i in class_y:
    All_dist[i] += 1

# plot dictionary
fig, ax = plt.subplots()
plt.bar(*zip(*All_dist.items()), color='lightgreen')  #https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary
plt.title('Distribution of AQI Classes')
fig.savefig('.\\Smart_Cities\\_viz\\_All_Distribution.png')
plt.clf()
plt.cla()
plt.close()



#Plot of test set of AQI
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(yTest.index, yTest, color='darkgreen')

ax.xaxis_date()

myFmt = mdat.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(myFmt)  # https://stackoverflow.com/questions/14946371/editing-the-date-formatting-of-x-axis-tick-labels-in-matplotlib
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))  # https://stackoverflow.com/questions/54057567/matplotlib-uneven-intervals-between-x-axis-with-datetime
## Rotate date labels automatically
fig.autofmt_xdate()

plt.title('AQI Test Set')

fig.savefig('.\\Smart_Cities\\_viz\\___AQI_test.png')
plt.clf()
plt.cla()
plt.close()











# first get accuracy for picking the most frequent in test and applying to train
Train_most_common_class = yTrain_class.value_counts().idxmax()
Test_len = len(yTest_class)
Test_correct = len(yTest_class[yTest_class==Train_most_common_class])

naive_acc = Test_correct/Test_len
naive_acc

# OUTLINE - for all save predictions vs classified targets

# Single regressions for baseline
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

SLR_Class.to_csv('.\\Smart_Cities\\_viz\\SLR_Classes.csv')

#get accuracy stats for baseline LR
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
SLR_acc

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

MLR_Class.to_csv('.\\Smart_Cities\\_viz\\MLR_Classes.csv')
yTest_class.to_csv('.\\Smart_Cities\\_viz\\Test_Classes.csv')

#get accuracy stats
MLR_acc = pd.DataFrame(index=MLR_Class.columns, columns=['Accuracy'])
for col in range(len(MLR_Class.columns)):
    acc =0
    for row in range(len(MLR_Class.iloc[:,col])):
        if MLR_Class.iloc[row,col] == yTest_class.iloc[row]:
            acc +=1
    total = len(MLR_Class.iloc[:,col])
    acc = acc/total
    MLR_acc.iloc[col,0]=acc

MLR_acc.sort_values(by='Accuracy', ascending=False, inplace=True)
MLR_acc


#Validation?  multiple models tested but were they tuned?




# feature selection backward, forward, stepwise


# SUpport vector machines - can play with hyperparameteres if we validate

SVM_DT = pd.DataFrame(index=xTest.index)

svmodel = svm.SVC( C=1.0,
                kernel='rbf', #must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed
                # degree : int, optional (default=3)
                #   Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
                # gamma : float, optional (default=’auto’)
                #   Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                #   Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma)
                decision_function_shape='ovo' #always used for multiclass
                )

svmodel=svmodel.fit(X=xTrain, y=yTrain_class)
predSVM = svmodel.predict(np.array(xTest))

# Random Forest, Bagging Boosting
# Random Forest
rfmodel = ensemble.RandomForestClassifier().fit(X=xTrain, y = yTrain_class)
predRF = rfmodel.predict(np.array(xTest))
colname = 'RF'
SVM_DT[colname]= predRF

# Bagging (can use nontree estimators)
bagmodel = ensemble.BaggingClassifier().fit(X=xTrain, y = yTrain_class)
predBAG = bagmodel.predict(np.array(xTest))
colname = 'BAG'
SVM_DT[colname]= predBAG
#bagmodel.score(np.array(xTest), yTest_class)  todo good check but use this when you refactor



# Boosting
boostmodel = ensemble.AdaBoostClassifier().fit(X=xTrain, y = yTrain_class)
predADA = boostmodel.predict(np.array(xTest))
colname = 'ADA_BST'
SVM_DT[colname]= predADA

gboostmodel = ensemble.GradientBoostingClassifier().fit(X=xTrain, y = yTrain_class)
predGRAD = gboostmodel.predict(np.array(xTest))
colname = 'GDT_BST'
SVM_DT[colname]= predGRAD

SVM_DT.to_csv('.\\Smart_Cities\\_viz\\SVM_DT_Classes.csv')

# get accuracy stats
SVM_DT_acc = pd.DataFrame(index=SVM_DT.columns, columns=['Accuracy'])
for col in range(len(SVM_DT.columns)):
    acc = 0
    for row in range(len(SVM_DT.iloc[:, col])):
        if SVM_DT.iloc[row, col] == yTest_class.iloc[row]:
            acc += 1
    total = len(SVM_DT.iloc[:, col])
    acc = acc / total
    SVM_DT_acc.iloc[col, 0] = acc

SVM_DT_acc.sort_values(by='Accuracy', ascending=False, inplace=True)
SVM_DT_acc

# NN in next file...
# maybe redo the analysis with a subset of the data


