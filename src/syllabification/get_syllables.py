"""
TODO
"""

import numpy as np
from syllabification import peakdetect as pd # see if it is efficient !!

def count_syllables(envelope, threshold):
    """
        Count peaks above a certain threshold to get number of syllable nucleus.
    """
    
    n_files = len(envelope)
    n_syl_nuclei = np.zeros(n_files)
    
    for k in range(n_files):
        n_syl_nuclei[k] = len(pd.peakdet(envelope[k], threshold)[0])
    
    return n_syl_nuclei

