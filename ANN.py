# Importing the libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD,Adam
from keras.models import Sequential,load_model
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('./CombinedWEDFRI.csv')
for idx,column in enumerate(dataset.columns):
    print(idx,column)
X = dataset.iloc[:, 0:11].values
Y = dataset.iloc[:,-1].values


# Encoding categorical data

le = LabelEncoder()
Y = le.fit_transform(Y)
for i in range(1,len(Y)):
    if Y[i] != 0 and Y[i] != 1:
        Y[i] = 1;

# Splitting the dataset into the Training set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pickle.dump(sc, open('./scaler.pkl','wb'))


# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units = 11, activation='relu'))
for i in range(0,4):
    ann.add(tf.keras.layers.Dense(units=11, activation='relu'))
 
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN
# Compiling the ANN
opt = Adam() 
ann.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=['accuracy','mse'])

# Training the ANN on the Training set
history = ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)
weights = ann.get_weights()
#Graph for mse
pyplot.plot(history.history['mse'])
pyplot.title('Training Performance')
pyplot.ylabel('Mean Squared Error')
pyplot.xlabel('Epochs')
pyplot.show()


# Part 4 - Making the predictions and evaluating the model
# Predicting the result of a single observation
# Predicting the Test set results

#saving model trained
ann.save('SavedModel.h5')
ann.summary()

Y_pred = ann.predict(X_test)
Y_pred = (Y_pred > 0.5)
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
acc = accuracy_score(Y_test, Y_pred)
print(acc)
