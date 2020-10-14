# -*- coding: utf-8 -*-
"""
Test and evaluate PatBR and PopBR with the rest of the data
COPYRIGHT: TU Delft, Netherlands. 2020.
LICENSE: GNU AGPL-3.0 
"""

# Import dependencies
from scipy.io import loadmat, savemat
import numpy as np
from tensorflow.keras.models import model_from_json
import glob
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import argparse

# Select patient used to train the model
PATIENT = 9

# Training parameters
SEQ_LEN = 120
SEQ_OUT = 100
SEQ_FORGE = 50

def reconstruct(model,x_lines,x_real=None):
    """ Reconstructs fragments of linearly interpolated signal.
    INPUT:
    model............Keras neural network model.
    x_lines..........interpolated breathing signal.
    x_real...........(optional) real signal for MSE calculation.
    """
    
    # Chain 1 gets stores half of predictions
    chain1 = []
    chain2 = []
    absolute_errors = []
    
    # Loop for slices in the signal
    for i in np.arange(SEQ_LEN,len(x_lines),SEQ_FORGE):
        
        # Get slice and normalize
        x_slide = x_lines[i-SEQ_LEN:i]
        norm_A = min(x_slide)
        norm_B = max(x_slide) - min(x_slide)
        x_norm = (x_slide - norm_A) / norm_B
        
        # Reconstruct signal
        x_rec = model.predict(x_norm.T)
        x_rec = np.reshape(x_rec, (SEQ_OUT,1))
        
        # Calculate squared error
        if i%2 == 0 and x_real is not None:
            x_true = x_real[i-SEQ_LEN:i-(SEQ_LEN-SEQ_OUT)]
            x_true = (x_true - min(x_true)) / (max(x_true) - min(x_true))
            absolute_errors.append(np.absolute(x_true - x_rec))
        
        # Invert normalization of reconstruction
        x_rec = x_rec * norm_B + norm_A
        
        # Append signals
        if i == SEQ_LEN: # If first slice, copy also first half in Chain2
            chain1.append(x_rec[0:SEQ_FORGE])
            chain2.append(x_rec[0:SEQ_FORGE])
            chain2.append(x_rec[SEQ_FORGE:SEQ_OUT])
            
        else: # Copy first half to Chain1, second halft to Chain2
            chain1.append(x_rec[0:SEQ_FORGE])
            chain2.append(x_rec[SEQ_FORGE:SEQ_OUT])
    
    # Append last half to Chain1
    chain1.append(x_rec[SEQ_FORGE:SEQ_OUT])
    
    # Convert to array, stack and average
    x1 = np.array(chain1)
    x2 = np.array(chain2)
    x_avg = np.vstack((x1.flatten(),x2.flatten()))
    x_avg = np.mean(x_avg, axis = 0)
        
    return x_avg, np.squeeze(np.array(absolute_errors))


# Load json and create model
json_file = open(f"saves/P{PATIENT}/modelBN{PATIENT}.json", 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load weights
model.load_weights(f"saves/P{PATIENT}/weights{PATIENT}.hdf5")
print("MODEL LOADED")


# Reconstruct sampled signals
# PopBR - Get all signals from the dataset
training_files = glob.glob(f"data/PE*")
signals = list(map(int,[training_files[i].split("PE")[1].split(".m")[0]
              for i in range(len(training_files))]))
signals.sort()
max_error_value = -1
error_avg = []
error_pat = 0
signals_pat = 0

# Loop through files
for i in range(len(signals)):
    
    # Get real and reconstructed files
    file = glob.glob(f"data/PE{signals[i]}.mat")
    lfile = glob.glob(f"data/SimPE{signals[i]}.mat")
        
    # Get real signal
    x = loadmat(file[0])
    true_signal = np.array(x["x_series"])
    
    # Get linear signal
    lx = loadmat(lfile[0])
    linear_signal = np.array(lx["x_f"])
    time = np.array(lx["t_f"])
    
    # Initialize final lists
    x_def = []
    error_def = []
    
    # Loop over signals in file
    for j in range(linear_signal.shape[1]):
        
        # Reconstruct and store signal
        x_lines = linear_signal[:,j].reshape((linear_signal.shape[0],1))
        x_avg, error = reconstruct(model,x_lines,true_signal)
        x_def.append(x_avg)
        
        # Store error per fragment
        error_def.append(np.mean(error,axis=1))
        
        # Store average error if reconstructing a different patient
        if PATIENT != 0 and signals[i]/PATIENT >= 10 and signals[i]/PATIENT <= 11:
            error_pat += np.sum(error)
            signals_pat += error.flatten().shape[0]
        else:
            error_avg.append(np.mean(error))
        
        # Store signal with maximum error
        if np.mean(error) > max_error_value:
            max_error_value = np.mean(error)
            max_error_signal = j+1
            max_error_patient = signals[i]
    
    # Cast to original shape and save
    x_def = np.array(x_def)
    x_def = x_def.reshape(x_def.shape[0],x_def.shape[1])
    x_def = np.transpose(x_def)
    time = time[:len(x_def[:,0])]
    savemat(f"saves/P{PATIENT}/{lfile[0][5:]}", {'X':x_def, 'T':time, 'MAE':error_def})
    print(f"Done with file {i+1} out of {len(signals)}!")
    
print(f"Max. error: {max_error_value}. Patient:{max_error_patient}. Signal:{max_error_signal}.")
print(f"Training error: {error_pat/max(signals_pat,1)}. Avg. error: {np.mean(error_avg)}")
