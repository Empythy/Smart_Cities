
################ CHECK FOR APPROPRIATENESS OF DUMMY/CONTINUOUS
import pandas as pd
import matplotlib
import numpy as np

data = pd.read_csv('Features_Targets.csv', index_col=0)
data.head()
reg_y = data.iloc[:,50]
class_y = data.iloc[:,51]
x = data.iloc[:,:50]


#Train and test splits
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, reg_y, test_size = 0.2, random_state = 0, stratify=reg_y)

train_pct = .80
train_stop_idx = int(len(data)*train_pct)
test_start_idx = train_stop_idx+24 #accounts for 24h influence on AQI

xTrain = x.iloc[:train_stop_idx,:]
xTest = x.iloc[test_start_idx:,:]

yTrain = reg_y.iloc[:train_stop_idx]
yTest = reg_y.iloc[test_start_idx:]



# OUTLINE - for all save predictions vs classified targets
# SIngle regressions
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

SLR_Class.head()


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


    # feature selection backward, forward, stepwise


    # PCA
    # SVD

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#maybe dont do PCA because we need interpretability
pca_scaled_X = StandardScaler()
pca = decomposition.PCA().fit(X=pca_scaled_X, y=y)
dir(pca)
pca.components_

predictPCA = reg.predict(np.array(xTest))
colname = 'PCA'
Mult_LR[colname] = predictPCA






decomposition.TruncatedSVD
# https://stats.stackexchange.com/questions/82050/principal-component-analysis-and-regression-in-python
# Try using a pipeline to combine principle components analysis and linear regression:
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
#
# # Principle components regression
# steps = [
#     ('scale', StandardScaler()),
#     ('pca', PCA()),
#     ('estimator', LinearRegression())
# ]
# pipe = Pipeline(steps)
# pca = pipe.set_params(pca__n_components=3)
# pca.fit(X, y)