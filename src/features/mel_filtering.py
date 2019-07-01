"""
TODO
"""

from librosa import filters, core
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def scaled_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax):
    """
        Determines the matrix for a Mel-filterbank and scales it for each
        filter values to sum up to 1.
        
        :param sample_rate: Sample rate of the signal
        :type sample_rate: int

        :param n_fft: number of FFT components
        :type n_fft: int

        :param n_mels: number of Mel bands to generate
        :type n_mels: int

        :param fmin: lowest frequency as a fraction of sample rate
        :type fmin: float

        :param fmax: highest frequency as a fraction of sample rate
        :type fmax: The second number to add
       
        :return: Mel transform matrix
        :rtype: list of list of float64
    """

    mel = filters.mel(sample_rate, n_fft, n_mels, fmin, fmax, 1)
    
    M = []
    for row in mel:
        row = list(row)
        m = min(x for x in row if x > 0)
        row_normalized = row / m
        row_scaled = row_normalized / row_normalized.sum()
        M.append(row_scaled)
    mel_scaled = np.array(M)
    
    # filterbank does not have first and last column in matlab code
    #mel_scaled = mel_scaled[:,1:-1]
    
    return mel_scaled


def mel_process(audio_files, window_length=0.025, window_step=0.0125,
                sample_rate=16000, use_spec_sub=False):
    """
        Determines the matrix for a Mel-filterbank and scales it for each
        filter values to sum up to 1.
        
        :param audio_files: Sample rate of the signal
        :type audio_files: int

        :param window_length: number of FFT components
        :type window_length: int

        :param window_step: number of Mel bands to generate
        :type window_step: int

        :param sample_rate: lowest frequency as a fraction of sample rate
        :type sample_rate: float

        :param use_spec_sub: highest frequency as a fraction of sample rate
        :type use_spec_sub: The second number to add
       
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

    mel = scaled_mel_filterbank(16000, 400, 24, 0, sample_rate/2)
    
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
        signal_mfcc = np.zeros((n_windows, mel.shape[0]))
        signal_energy = np.zeros(n_windows)
        
        j = 0
        for i_start in range(0, s_length, window_step):
            frames = signal[i_start : i_start + window_length] * window_hamming
            fft_mag = np.abs(np.fft.rfft(frames))
            mfcc = 20 * np.log10(np.matmul(mel, fft_mag))
            
            signal_mfcc[j,:] = mfcc
            signal_energy[j] = sum(fft_mag)
            j += 1
        
        F.append(signal_mfcc)
        E.append(signal_energy)
        
        # include procbar
        i += 1
        
    return F, E
    