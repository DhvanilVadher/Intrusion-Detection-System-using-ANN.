import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD
from numpy import array
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.callbacks import Callback
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
le = LabelEncoder()

# load model
model = load_model('MyModel1000Final.h5')
model.summary()
testing_data = pd.read_csv('./Test.csv')
#print(testing_data)

required_cols = [0,1,2,3,4,5,6,7,8,9,10]
X = testing_data.iloc[:, required_cols].values
Y = testing_data.iloc[:,-1].values


#Y = le.fit_transform(Y)
'''cnt = set()
for i in range(1,len(Y)):
    if Y[i] != 0 and Y[i] != 1:
        Y[i] = 1;
'''
    
#X = sc.fit_transform(X)
#X = sc.transform(X)

Y_pred = model.predict(X)
Y_pred = (Y_pred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y, Y_pred)
print(X)
print(Y)
print(cm)
print(accuracy_score(Y, Y_pred))

