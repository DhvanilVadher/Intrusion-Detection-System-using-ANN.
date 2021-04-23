import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD,Adam
from keras.models import Sequential,load_model
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle


dataset = pd.read_csv('./CombinedWEDFRI.csv')
for idx,column in enumerate(dataset.columns):
    print(idx,column)
X = dataset.iloc[:, 0:11].values
Y = dataset.iloc[:,-1].values

X_col = dataset.iloc[:, 0:11]
Y_col = dataset.iloc[:,-1]
le = LabelEncoder()
Y = le.fit_transform(Y)
for i in range(1,len(Y)):
    if Y[i] != 0 and Y[i] != 1:
        Y[i] = 1;
# Splitting the dataset into the Training set
print(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print(2)
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, Y_train)
print(sel.get_support())
print(3)
print(sel.get_support().sum()) 
selected_feat= X_col.columns[(sel.get_support())]
len(selected_feat)
print(4)
print(selected_feat)

pd.Series(sel.estimator_.feature_importances_.ravel()).hist()
importances = sel.estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
# X is the train data used to fit the model 
pyplot.figure()
pyplot.title("Feature importances")
pyplot.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
pyplot.xticks(range(X.shape[1]), indices)
pyplot.xlim([-1, X.shape[1]])
pyplot.show()