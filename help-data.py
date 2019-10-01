
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:11:13 2019

"""

import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



#%%
#Récupération des données
root = os.getcwd()

df1 = pd.read_csv(root + '/DB1.data',header=None,delimiter=' ')
df2 = pd.read_csv(root+ '/DB1_labels.data', header= None,delimiter=' ')
#drop the date column
df2 = df2.drop([df2.columns[1]] ,  axis='columns')

#%%
#Data Cleaning 
df1lim = int(len(df1)*0.5)
df1= df1.dropna(thresh=df1lim,axis=1)
df1 = df1.fillna(df1.mean())

scaler = preprocessing.StandardScaler()
df1 = scaler.fit_transform(df1)
df1 = pd.DataFrame(df1)

#%%
df2.replace(-1,0)
df2.hist()

#%%


#%%
#Feature Selection - Dimension Reduction
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
x_res=selector.fit_transform(df1)
x_res = pd.DataFrame(x_res)

from sklearn.feature_selection import SelectKBest, f_classif
bestFeatures = SelectKBest(score_func=f_classif,k=130)
x_res = bestFeatures.fit_transform(x_res, df2)


#%%

graph = pyplot.plot(range(0,446),np.cumsum(sorted(bestFeatures.scores_,reverse=True)))
pyplot.title('Score discriminant cumulatif en fonction du nombres de features')
pyplot.show()

#%%
#Classifier
#comparaison de plusieurs clf pour savoir lequel est le meilleur

liste_clf=[]


x_train, x_test, y_train, y_test  = train_test_split(x_res, df2, test_size=0.25, random_state=2108)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=2108)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2108, ratio = 1.0)

x_train, y_train = sm.fit_sample(x_train, y_train)

#%%

#clf1 = LogisticRegression(random_state=2108, class_weight='balanced').fit(x_train, y_train)


#%%
# 'good for precision'
#clf1  = RandomForestClassifier( n_estimators=150, criterion='gini', class_weight='balanced').fit(x_train,y_train)

#%%

#clf1 = xgb.XGBClassifier().fit(x_train, y_train)

#%%
# 'good' for recall
from sklearn.linear_model import SGDClassifier
clf1 = SVC(kernel='linear', class_weight='balanced', probability=True).fit(x_train,y_train) 

#%%
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

pred = clf1.predict(x_val)

#%%
print('accuracy',accuracy_score(y_val,pred))
print('precision',precision_score(y_val,pred))
print('recall',recall_score(y_val,pred))
print('avg prec',average_precision_score(y_val,pred))
print('roc_auc',roc_auc_score(y_val,pred))
print('F1_score',f1_score(y_val,pred))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_val,pred))

#%%

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_val,pred)


plt.step(recall, precision, color='green', alpha=0.2,label=str('validation '+str(round(average_precision_score(y_val,pred),3))))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend()
plt.title('2-class Precision-Recall curve : ')


pred = clf1.predict(x_test)

#%%
print('#####')
print('#####')
print('#####')

print('accuracy',accuracy_score(y_test,pred))
print('precision',precision_score(y_test,pred))
print('recall',recall_score(y_test,pred))
print('avg prec',average_precision_score(y_test,pred))
print('roc_auc',roc_auc_score(y_test,pred))
print('F1_score',f1_score(y_test,pred))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,pred))

#%%


