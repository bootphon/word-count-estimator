"""
TODO
"""

import numpy as np

def reconstruct_envelope(envelope_windows, timestamps, frames_length):
    """
        TODO
    """
    
    n_files = len(frames_length)
    envelope = []
    
    for f in range(n_files):
       envelope.append(np.zeros(int(frames_length[f])))
    
    maxlen = int(max(frames_length))
    totsums = np.zeros((n_files, maxlen))
    
    n_windows = envelope_windows.shape[0]
    for w in range(n_windows):
        window_ts = timestamps[w,:,:].astype(int)
        for k in range(len(window_ts)):
            file_id = window_ts[k, 0]
            i = window_ts[k, 1]
            if i > -1:
                envelope[file_id][i] += envelope_windows[w, k]
                totsums[file_id, i] += 1

    envelope =  envelope / totsums
    
    return envelope


def count_syllables():
    # TODO
    pass

