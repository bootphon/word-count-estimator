"""
TODO
"""

import numpy as np
import scipy.io


def cut_features(feature_frames, model_params_file):
    """
        Cut all file's features into portions of equal size that can be
        processed by the BLSTM network.
    """
    
    # join every file's features matrix together in one matrix
    tot_features = np.concatenate(feature_frames)
    tot_length = len(tot_features)
    
    # timestamps keeps track of the number of the origin feature frame and the
    # timestamps of the features
    timestamps = np.zeros((tot_length, 2), dtype=int)
    frames_length = np.zeros(len(feature_frames), dtype=int)
    frame_nb = 0
    l = 0
    for f in feature_frames:
        frame_len = len(f)
        frames_length[frame_nb] = frame_len
        for i in range(l, l+frame_len):
            timestamps[i] = (frame_nb, i-l)
        frame_nb += 1
        l += frame_len
    
    # load parameters for window, meme/devi matrices and apply them
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
        timestamps = np.concatenate((timestamps,
                                       np.full((window_length-excess, 2),
                                               (frame_nb-1, -1))))
        tot_length = len(tot_features)
    
    # slide window over total_features and append it to X at each step
    n_slices = ((tot_length - window_length) // window_step) + 1
    X = np.zeros((n_slices, window_length, 24))
    X_timestamps = np.zeros((n_slices, window_length, 2), dtype=int)
    k = 0
    for i_start in range(0, tot_length-window_length+window_step, window_step):
        X[k, :, :] = tot_features[i_start:i_start + window_length, :]
        X_timestamps[k, :] = timestamps[i_start:i_start+window_length, :]
        k += 1
    
    X[np.isneginf(X)] = 0
    X[X == np.inf] = 0
    X[X == np.NaN] = 0

    # divide into multiple files if needed
    # TODO
    
    return X, X_timestamps, frames_length
