"""
TODO
"""

import numpy as np

def reconstruct_envelope(envelope_windows, timestamps, frames_length, window_length):
    """
        Recover file specific envelope frames from all envelope windows.
    """
    
    n_files = len(frames_length)
    envelope = []
    tot_sums = []
    
    for f in range(n_files):
       envelope.append(np.zeros(frames_length[f]))
       tot_sums.append(np.zeros(frames_length[f]))
    
    # add (in envelope) and count (in tot_sums) values in the envelope windows
    # which come from the same feature and frame
    n_windows = envelope_windows.shape[0]
    for w in range(n_windows):
        window_ts = timestamps[w,:,:]
        for k in range(window_length):
            frame_nb = window_ts[k][0]
            i = window_ts[k][1]
            if i > -1:
                envelope[frame_nb][i] += envelope_windows[w][k]
                tot_sums[frame_nb][i] += 1

    # compute the mean of values
    for f in range(n_files):
        envelope[f] =  envelope[f] / tot_sums[f]
    
    return envelope