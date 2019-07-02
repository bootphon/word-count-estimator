"""
TODO
"""

import numpy as np
import scipy.io


def cut_features(features, model_params_file):
    """
        Cut all file's features into portions of equal size that can be
        processed by the BLSTM network.
    """
    
    # join every file's features matrix together in one matrix
    tot_features = np.concatenate(features)
    tot_length = len(tot_features)
    
    # features_ids keeps track of the number of the origin file of the features
    features_ids = np.zeros(tot_length)
    file_id, l = 0, 0
    for f in features:
        n_features = len(f)
        features_ids[l:l+n_features] = [file_id]*n_features
        file_id += 1
        l += n_features
    
    model_params = scipy.io.loadmat(model_params_file)
    window_length = model_params['wl'][0][0]
    window_step = model_params['ws'][0][0]
    meme = model_params['meme']
    devi = model_params['devi']
    
    tot_features = tot_features - meme
    tot_features = tot_features / devi
    
    # add zeros if the length is not a multiple of the step's size
    excess = tot_length % window_step
    
    if excess != 0:
        tot_features = np.concatenate((tot_features,
                                       np.zeros((window_length-excess, 24))))
        features_ids = np.concatenate((features_ids,
                                       np.full((window_length-excess), i-1)))
        tot_length = len(tot_features)
    
    # slide window over total_features and add the result to X
    n_slices = ((tot_length - window_length) // window_step) + 1
    
    k = 0
    X = np.zeros((n_slices, window_length, 24))
    X_features_ids = np.zeros((n_slices, window_length))
    for i_start in range(0, tot_length - window_length + window_step, window_step):
        X[k, :, :] = tot_features[i_start:i_start + window_length, :]
        X_features_ids[k, :] = features_ids[i_start:i_start+window_length]
        k += 1
    
    X[np.isneginf(X)] = 0
    X[X == np.inf] = 0
    X[X == np.NaN] = 0

    # divide into multiple files if needed
    # TODO
    
    return X, X_features_ids
