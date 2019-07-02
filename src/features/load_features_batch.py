"""
TODO
"""

import math
import numpy as np
import scipy.io


def cut_features_in_batches(features, model_params_file):
   
    n_files = len(features)
    tot_length = sum(len(f) for f in features)
    
    tot_features = np.concatenate(features)
    
    timestamps = np.zeros(tot_length)
    i, l = 0, 0
    for f in features:
        file_length = len(f)
        timestamps[l:l+file_length] = [i]*file_length
        i += 1
        l += file_length
    
    model_params = scipy.io.loadmat(model_params_file)
    window_length = model_params['wl'][0][0]
    window_step = model_params['ws'][0][0]
    meme = model_params['meme']
    devi = model_params['devi']
    
    tot_features = tot_features - meme
    tot_features = tot_features / devi
    
    # Make sure that the length of the data is equal to multiple of window
    # length to avoid any clipping of endings then slice the total features
    # array
    excess = tot_length % window_step
    
    if excess != 0:
        tot_features = np.concatenate((tot_features, np.zeros((window_length-excess, 24))))
        timestamps = np.concatenate((timestamps, np.full((window_length-excess), i-1)))
        tot_length = len(tot_features)
    
    n_slices = ((tot_length - window_length) // window_step) + 1
    
    k = 0
    X = np.zeros((n_slices, window_length, 24))
    X_timestamps = np.zeros((n_slices, window_length))
    for i_start in range(0, tot_length - window_length + window_step, window_step): # add window step to upper bound because otherwise stops one step before
        X[k, :, :] = tot_features[i_start:i_start + window_length, :]
        X_timestamps[k, :] = timestamps[i_start:i_start+window_length]
        k += 1
    
    X[np.isneginf(X)] = 0
    X[X == np.inf] = 0
    X[X == np.NaN] = 0

    # divide into multiple files if needed
    
    return X, X_timestamps
    
X, X_ts = cut_features_in_batches(F, "C:/Users/Médéric Carriat/Desktop/wce/models/LSTM_params_BLSTM_fourlang_60_60_augmented_dropout_v2.h5.mat")