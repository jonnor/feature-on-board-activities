#!/usr/bin/python3

#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=DeprecationWarning)

import os
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.utils import np_utils
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

NUM_EPOCHS = 10

def classify_features():
    print("\nFEATURES")
    X = np.loadtxt("../features.csv", skiprows=0)
    y = np.loadtxt("../classes.csv", skiprows=0)

    u = np.unique(y)
    y_map = {v:i for i, v in enumerate(u)}
    y = np.asarray([y_map[v] for v in y])
    print(y)

    n_classes = int(max(y) + 1)
    print("num classes=", n_classes)
    Y = np_utils.to_categorical(y, n_classes)
    print(Y)

    # Scale data to have mean 0 and variance 1 
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.5, random_state=2)

    n_features = X.shape[1]
    n_classes = Y.shape[1]

    nodes = 64

    model = Sequential(name="PAMAP2")
    model.add(Dense(nodes, input_dim=n_features, activation='relu'))
    model.add(Dense(nodes, activation='relu', name="intermediate_layer"))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='mse', 
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=5,
              epochs=NUM_EPOCHS,
              verbose=1,
              validation_data=(X_test, Y_test))
    model.summary()
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def classify_raw():
    print("\nRAW")
    accel_x = np.loadtxt("../raw_x.txt", skiprows=0)
    accel_y = np.loadtxt("../raw_y.txt", skiprows=0)
    accel_z = np.loadtxt("../raw_z.txt", skiprows=0)
    y = np.loadtxt("../classes.csv", skiprows=0)
    X = np.concatenate((accel_x, accel_y, accel_z), axis=1)

    u = np.unique(y)
    y_map = {v:i for i, v in enumerate(u)}
    y_reverse_map = {i:v for i, v in enumerate(u)}
    y = np.asarray([y_map[v] for v in y])
    print(y)

    n_classes = int(max(y) + 1)
    print("num classes=", n_classes)
    Y = np_utils.to_categorical(y, n_classes)
    print(Y)

    # Scale data to have mean 0 and variance 1 
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.5, random_state=2)

    n_features = X.shape[1]
    n_classes = Y.shape[1]

    nodes = 64

    model = Sequential(name="PAMAP2")
    model.add(Dense(nodes, input_dim=n_features, activation='relu'))
    model.add(Dense(nodes, activation='relu', name="intermediate_layer"))
    model.add(Dense(n_classes, activation='softmax'))
    # model.compile(loss='mse',
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=5,
              epochs=NUM_EPOCHS,
              verbose=1,
              validation_data=(X_test, Y_test))

    model.summary()
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # check the results on the validation set
    hypothesis = model.predict(X_test)
    hypothesis = np.argmax(hypothesis, 1)
    hypothesis = [y_reverse_map[v] for v in hypothesis]

    y_test = np.argmax(Y_test, 1)
    y_test = [y_reverse_map[v] for v in y_test]

    validation_score = f1_score(y_test, hypothesis, average="micro")
    print("validation_score=", validation_score)


def main():
#    classify_features()
    classify_raw()

if __name__ == "__main__":
    main()
