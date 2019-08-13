import sys
import numpy as np

def peakdet(v, delta, x=None):
    """
    Peak picking algorithm.

    Converted from MATLAB script at http://billauer.co.il/peakdet.html by
    https://github.com/endolith.

    Parameters
    ----------
    v : ndarray
        1D array, input signal.
    delta : float
        Minimum value separating a maximum and its left neighbour for this
        maximum to be considered a peak.
    x : ndarray
        1D array, indices for mintab and maxtab.

    Returns
    -------
    maxtab : ndarray
        1D array containing the indices of the maximum peaks.
    mintab : ndarray
        1D array containing the indices of the minimum peaks.
    """

    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, np.NINF
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)

