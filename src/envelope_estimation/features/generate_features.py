"""
TODO
"""

from librosa import filters, core
import numpy as np
import soundfile as sf


def scaled_mel_filters_filterbank(sample_rate, n_fft, n_mel_filters, fmin, fmax):
    """
        Determines the matrix for a mel_filters-filterbank and scales it for each
        filter values to sum up to 1.
        
        :param sample_rate: sample rate of the signal
        :type sample_rate: int

        :param n_fft: number of FFT components
        :type n_fft: int

        :param n_mel_filterss: number of mel_filters bands to generate
        :type n_mel_filterss: int

        :param fmin: lowest frequency as a fraction of sample rate
        :type fmin: float

        :param fmax: highest frequency as a fraction of sample rate
        :type fmax: The second number to add
       
        :return: mel_filters transform matrix
        :rtype: list of list of float64
    """

    mel_filters = filters.mel(sample_rate, n_fft, n_mel_filters, fmin, fmax, 1)
    
    M = []
    for row in mel_filters:
        row = list(row)
        m = min(x for x in row if x > 0)
        row_normalized = row / m
        row_scaled = row_normalized / row_normalized.sum()
        M.append(row_scaled)
    mel_filters_scaled = np.array(M)
    
    # filterbank does not have first and last column in matlab code
    #mel_filters_scaled = mel_filters_scaled[:,1:-1]
    
    return mel_filters_scaled


def generate_features(audio_files, window_length=0.025, window_step=0.0125,
                      sample_rate=16000, use_spec_sub=False):
    """
        Determines the energy and MFCCs of each signal.
        
        :param audio_files: audio files to process
        :type audio_files: list of str

        :param window_length: length of the sliding window as a fraction of 
        the sample rate
        :type window_length: int

        :param window_step: step of the sliding window as a fraction of 
        the sample rate
        :type window_step: int

        :param sample_rate: sample rate the audio files should have
        :type sample_rate: int

        :param use_spec_sub: use spectral subtraction or not
        :type use_spec_sub: boolean
       
        :return: a tuple with:
            - F - list of MFCCs matrices per signal
            - E - list of energy array per signal
        :rtype: (list, list)
    """
    
    window_length = round(window_length * sample_rate)
    window_step = round(window_step * sample_rate)
    window_hamming = np.hamming(window_length)
    n_files = len(audio_files)
    F = []
    E = []

    mel_filters = scaled_mel_filters_filterbank(16000, 400, 24, 0, sample_rate/2)
    
    i = 0
    while i < n_files:
        f = audio_files[i]
        signal, f_sample_rate = sf.read(f)
        s_length = len(signal)
        
        if f_sample_rate != sample_rate:
            signal = core.resample(signal, f_sample_rate, sample_rate)
            f_sample_rate = sample_rate
        
        # add 0s at both sides of the signal
        signal = np.concatenate((signal, [0] * (window_length // 2)))
        signal = np.concatenate(([0] * (window_length // 2), signal))

        # add spectral subtraction if possible
        # TODO
        
        # slide window over signal and compute energy and MFFCs at each step
        n_windows = s_length // window_step
        signal_mfcc = np.zeros((n_windows, mel_filters.shape[0]))
        signal_energy = np.zeros(n_windows)
        
        j = 0
        for i_start in range(0, s_length, window_step):
            frames = signal[i_start:i_start + window_length] * window_hamming
            fft_magnitude = np.abs(np.fft.rfft(frames))
            mfcc = 20 * np.log10(np.matmul(mel_filters, fft_magnitude))
            
            signal_mfcc[j, :] = mfcc
            signal_energy[j] = sum(fft_magnitude)
            j += 1
        
        F.append(signal_mfcc)
        E.append(signal_energy)
        
        # add progress bar
        # TODO
        
        i += 1
        
    return F, E
