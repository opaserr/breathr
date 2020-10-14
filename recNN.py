# -*- coding: utf-8 -*-
"""
Reconstruction of time series from discretized vectors
COPYRIGHT: TU Delft, Netherlands. 2020.
LICENSE: GNU AGPL-3.0 
"""

# Import dependencies
import numpy as np
from scipy.io import loadmat
import pandas as pd
from collections import deque
from sklearn.utils import shuffle
import random
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse

# Parameters
PATIENT = 0
SEQ_LEN = 120
SEQ_OUT = 100      
BATCH_SIZE = 256


# Select PopBR model (patient=0) or any specific patient
if PATIENT==0:
    
    # PopBR - Get all signals from the dataset
    training_files = glob.glob(f"drive/Recnn/data/SimPG*")
    signals = list(map(int,[training_files[i].split("SimPG")[1].split(".m")[0]
                  for i in range(len(training_files))]))
    
    # Select subset of signals for training and validation
    N_TRAIN = 10000 
    N_VAL = N_TRAIN + 1000  
    
else:
    
    # PatBR - Get all signals from a patient
    training_files = glob.glob(f"drive/Recnn/data/SimPG{PATIENT}?.mat")
    signals =  list(map(int,[f'{PATIENT}'+f'{i}' for i in range(len(training_files))]))
    
    # Select all available data
    N_TRAIN = -5000
    N_VAL = -1000
    
    
# AUXILIARY FUNCTIONS
def preprocess_df(data):   
    """ Preprocess pandas DataFrame and output data points.
    INPUT:
    data........pandas DataFrame with signal fragments [pd.DataFrame]
    OUTPUT
    X...........training points [np.array]
    y...........target values [np.array]
    """
    # Drop NaN and initialize
    df.dropna(inplace=True)
    sequential_data = []
    prev_val = deque(maxlen=SEQ_LEN)
    
    # Slice signal position after position
    for i in data.values:
        prev_val.append([n for n in i])
        if len(prev_val) == SEQ_LEN:
            sequential_data.append([np.array(prev_val)])
     
    # Initialize empty lists
    X = []
    y = []
    
    # Shuffle patient data
    random.shuffle(sequential_data)
    
    # Normalize to [0,1]
    for seq in sequential_data:
        seq = np.array(seq[0])
        xval = seq[:,0]
        yval = seq[:,1]
        minv = min(xval)
        maxv = max(xval)
        x = (xval-minv)/(maxv-minv)
        Y = (yval-minv)/(maxv-minv)
        X.append(x)
        y.append(Y[0:SEQ_OUT])
        
    return np.array(X), np.array(y)


# DEFINE RECONSTRUCTION NETWORK
# Build NN model using Keras
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=SEQ_LEN))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(SEQ_OUT, activation='linear'))

model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-6))
filepath=f"drive/Recnn/saves/P{PATIENT}/weights{PATIENT}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose = 1)
callbacks_list = [checkpoint, es]


# TRAIN THE MODEL
# Initialize dataset arrays
x_train = np.array([]).reshape(0,SEQ_LEN)
y_train = np.array([]).reshape(0,SEQ_OUT)
x_val = np.array([]).reshape(0,SEQ_LEN)
y_val = np.array([]).reshape(0,SEQ_OUT)
x_test = np.array([]).reshape(0,SEQ_LEN)
y_test = np.array([]).reshape(0,SEQ_OUT)

# Load training data one signal at a time
for i in range(len(signals)):
    
    # Read GUH data
    y = loadmat(f"drive/Recnn/data/PG{signals[i]}.mat")
    x = loadmat(f"drive/Recnn/data/SimPG{signals[i]}.mat")
    y_values = np.array(y["x_series"])
    x_values = np.array(x["x_f"])
    time = np.array(x["t_f"])
    
    # Create pandas DataFrame
    data = np.hstack((time,x_values,y_values))
    df = pd.DataFrame(data, columns = ['time','Approx','Real'])
    df.set_index("time", inplace=True)
    
    # Select the last 5% of time values for test set
    times = sorted(df.index.values)
    last_5pct = times[-int(0.05*len(times))]
    test_df = df[(df.index>=last_5pct)]
    
    # Create train and validation data
    x_train_to_append, y_train_to_append = preprocess_df(df)
    x_test_to_append, y_test_to_append = preprocess_df(test_df)
    
    # Shuffle data
    x_train_to_append, y_train_to_append = shuffle(x_train_to_append, y_train_to_append)
    x_test_to_append, y_test_to_append = shuffle(x_test_to_append, y_test_to_append)

    x_train = np.vstack([x_train,x_train_to_append[:N_TRAIN]])
    y_train = np.vstack([y_train,y_train_to_append[:N_TRAIN]])
    x_val = np.vstack([x_val,x_train_to_append[N_TRAIN:N_VAL]])
    y_val = np.vstack([y_val,y_train_to_append[N_TRAIN:N_VAL]])
    x_test = np.vstack([x_test,x_test_to_append[-1000:]])
    y_test = np.vstack([y_test,y_test_to_append[-1000:]])
    
# Shuffle training data
x_train, y_train = shuffle(x_train, y_train)
x_val, y_val = shuffle(x_val, y_val)
x_test, y_test = shuffle(x_test, y_test)

model.fit(x_train, y_train, epochs=2000, validation_data=(x_val,y_val), batch_size = BATCH_SIZE,
          verbose=1, callbacks=callbacks_list)

# Serialize model to JSON
model_json = model.to_json()
with open(f"drive/Recnn/saves/P{PATIENT}/modelBN{PATIENT}.json", "w") as json_file:
    json_file.write(model_json)
print("MODEL SAVED")

