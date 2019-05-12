# rerunning the model using the following selected features

features = ['sm_Atmospheri',
            'sm_Wind Speed',
            'sm_Temperatur',
            'sm_Relative H',
            'sm_traffic',
            'weekend']

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

fsm = pd.DataFrame(index=data.index)


for i in features:
    fsm[str(i)] = data[str(i)]

reg_y = data.AQI
class_y = data.AQI_C


#stratified Train, test splits

from sklearn.model_selection import StratifiedShuffleSplit


#drop timestamp index bc it messes up strat shuffle split
data_i = data.copy()
data_i.reset_index(inplace=True)
data_i.drop(columns="timestamp", inplace=True)
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
    y_train = data_i.AQI.loc[train_idx]
    y_test = data_i.AQI.loc[test_idx]
    x_train = fsm.loc[train_idx]
    x_test = fsm.loc[test_idx]


from sklearn.metrics import mean_squared_error
# Multiple regression
    # full
MLR = pd.DataFrame(index=test_idx)
MLR_MSE = pd.DataFrame()
X=x_train
y=y_train
reg = linear_model.LinearRegression().fit(X=X, y=y)
predictMLR = reg.predict(np.array(x_test))
colname = 'MLR_ALL'
MLR[colname] = predictMLR
MLR_MSE[colname] = [mean_squared_error(y_true=y_test, y_pred=predictMLR)]
mlr_coeff = reg.coef_


reg = linear_model.Ridge().fit(X=X, y=y)
predictMLR = reg.predict(np.array(x_test))
colname = 'MLR_Ridge'
MLR[colname] = predictMLR
MLR_MSE[colname] = [mean_squared_error(y_true=y_test, y_pred=predictMLR)]
ridge_coeff = reg.coef_

reg = linear_model.Lasso().fit(X=X, y=y)
predictMLR = reg.predict(np.array(x_test))
colname = 'MLR_Lasso'
MLR[colname] = predictMLR
MLR_MSE[colname] = [mean_squared_error(y_true=y_test, y_pred=predictMLR)]
lasso_coeff = reg.coef_

reg = linear_model.ElasticNet().fit(X=X, y=y)
predictMLR = reg.predict(np.array(x_test))
colname = 'MLR_ElasticNet'
MLR[colname] = predictMLR
MLR_MSE[colname] = [mean_squared_error(y_true=y_test, y_pred=predictMLR)]
elastic_coeff = reg.coef_

MLR.head()
MLR_MSE
# Make classes

cutoffs = [-1,50,100,150,200,np.inf]
labels = ['Good','Acceptable', 'Mediocre', 'Poor', 'Bad']

#do same with test and training ys
y_true = pd.cut(y_test, bins=cutoffs, labels=labels)
y_train_class = pd.cut(y_train, bins=cutoffs, labels=labels)


MLR_Class = MLR.copy()
for i in MLR_Class.columns:
    MLR_Class[i] = pd.cut(MLR_Class[i], bins=cutoffs, labels=labels)

MLR_Class.to_csv('.\\Smart_Cities\\_viz\\MLR_Classes3.csv')

MLR_Class

#get accuracy stats
MLR_ACC = pd.DataFrame()
for col in MLR_Class.columns:
    acc =0
    for row in range(len(y_true)):
        if MLR_Class[col].iloc[row] == y_true.iloc[row]:
            acc +=1
    total = len(MLR_Class[col])
    acc = acc/total
    MLR_ACC[col]=[acc]

MLR_ACC.transpose()


########## CLASSIFICATION
########## CLASSIFICATION
########## CLASSIFICATION
########## CLASSIFICATION
########## CLASSIFICATION

Class_ACC = pd.DataFrame()

# SUpport vector machines
SVM_DT = pd.DataFrame()

svmodel = svm.SVC( C=1.0,
                kernel='rbf', #must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed
                # degree : int, optional (default=3)
                #   Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
                # gamma : float, optional (default=’auto’)
                #   Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                #   Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma)
                decision_function_shape='ovo' #always used for multiclass
                )

svmodel=svmodel.fit(X=x_train, y=y_train_class)
predSVM = svmodel.predict(np.array(x_test))
colname = 'SVM'
SVM_DT[colname]= predSVM
Class_ACC['SVM'] = [svmodel.score(X=x_test, y=y_true)]

# Random Forest, Bagging Boosting
# Random Forest
rfmodel = ensemble.RandomForestClassifier().fit(X=x_train, y = y_train_class)
predRF = rfmodel.predict(np.array(x_test))
colname = 'RF'
SVM_DT[colname]= predRF
Class_ACC[colname] = [rfmodel.score(X=x_test, y=y_true)]
RF_features = rfmodel.feature_importances_

# Bagging (can use nontree estimators)
bagmodel = ensemble.BaggingClassifier().fit(X=x_train, y = y_train_class)
predBAG = bagmodel.predict(np.array(x_test))
colname = 'BAG'
SVM_DT[colname]= predBAG
Class_ACC[colname] = [bagmodel.score(X=x_test, y=y_true)]


# Boosting
boostmodel = ensemble.AdaBoostClassifier().fit(X=x_train, y = y_train_class)
predADA = boostmodel.predict(np.array(x_test))
colname = 'ADA_BST'
SVM_DT[colname]= predADA
Class_ACC[colname] = [boostmodel.score(X=x_test, y=y_true)]

gboostmodel = ensemble.GradientBoostingClassifier().fit(X=x_train, y = y_train_class)
predGRAD = gboostmodel.predict(np.array(x_test))
colname = 'GDT_BST'
SVM_DT[colname]= predGRAD
Class_ACC[colname] = [gboostmodel.score(X=x_test, y=y_true)]

SVM_DT.to_csv('.\\Smart_Cities\\_viz\\SVM_DT_Classes.csv')
Class_ACC.transpose()



