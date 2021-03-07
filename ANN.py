# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD
from numpy import array
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.callbacks import Callback
from matplotlib import pyplot

test_acc = []

'''class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        test_acc.append(acc)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
'''
# Part 1 - Data Preprocessing

# Importinlrg the dataset
opt = SGD(lr = 0.005)
dataset = pd.read_csv('./CombinedWEDFRI.csv')

for idx,column in enumerate(dataset.columns):
    print(idx,column)

required_cols = [0,1,2,3,4,5,6,7,8,9,10]
X = dataset.iloc[:, required_cols].values
y = dataset.iloc[:,-1].values
# print(X)
print("something") 
print(y)

# Encoding categorical data
# Label Encoding the "Label" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
cnt = set()
for i in range(1,len(y)):
    if y[i] != 0 and y[i] != 1:
        y[i] = 1;

for i in y:
    cnt.add(i)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 15, activation='relu'))

# Adding the second hidden layer
for i in range(1,4):
    ann.add(tf.keras.layers.Dense(units=15, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=['accuracy','mse'])


# Training the ANN on the Training set
history = ann.fit(X_train, y_train, batch_size = 32, epochs = 1000)

minval=min(history.history['mse'])
for i in range(0,len(history.history['mse'])):
    if minval == history.history['mse'][i]:
        print("Min index = ")
        print(i)
        print("Min Value =")
        print(minval)
        break
#Graph for mse
print(history.history.keys())
pyplot.plot(history.history['mse'])
pyplot.title('Training Performance')
pyplot.ylabel('Mean Squared Error')
pyplot.xlabel('1000 Epochs')
pyplot.show()
# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation


# Predicting the Test set results

#saving model trained
ann.save('MyModel1000Final.h5')

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)