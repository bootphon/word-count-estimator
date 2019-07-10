from librosa import filters, core
import numpy as np
import soundfile as sf
import pickle


DEFAULT_PARAMS="../models/envelope_estimator/default_data_processing_params.pickle"


class DataProcessing():
    """
    Class for data processing tasks.
    
    Its purpose is to:
        - extract features from the input audio files and create batches from
          those features.
        - once the batches have been processed by the syllable envelope
          estimator, reconstruct the syllable envelope of each file.
    
    Attributes
    ----------
    fgen_window_length : float
        Length of the sliding window for the features generation as a fraction
        of the audio files sample rate.
        Defaults to 0.025.
    fgen_window_step : float
        Step of the sliding window for the features generation as a fraction
        of the audio files sample rate.
        Defaults to 0.01.
    cut_window_length : int
        Length of the sliding window for the cutting operation.
        Defaults to 300.
    cut_window_step : int
        Step of the sliding window for the cutting operation.
        Defaults to 100.
    meme : float array
        Array applied to features matrices.
    devi : float array
        Array applied to features matrices.
        
    Methods
    -------
    save_parameters(params_file)
        Save the data processing parameters to a given file.
    load_parameters(params_file)
        Load the data processing parameters from a given file.
    generate_features_batches(self, audio_files)
        Generate features batches from a list of audio files.
    reconstruct_envelopes(envelopes_batches, timestamps, ori_frames_length)
        Reconstruct the syllable envelope of each file from the result batches
        of the envelope estimator.
    """
    
    def __init__(self, fgen_window_length = 0.025, fgen_window_step = 0.01,
                 cut_window_length = 300, cut_window_step = 100):
        """
        Parameters
        ----------
        fgen_window_length : float
            Length of the sliding window for the features generation as a fraction
            of the audio files sample rate.
            Defaults to 0.025.
        fgen_window_step : float
            Step of the sliding window for the features generation as a fraction
            of the audio files sample rate.
            Defaults to 0.01.
        cut_window_length : int
            Length of the sliding window for the cutting operation.
            Defaults to 300.
        cut_window_step : int
            Step of the sliding window for the cutting operation.
            Defaults to 100.
        """
        
        self.fgen_window_length = fgen_window_length
        self.fgen_window_step = fgen_window_step
        self.cut_window_length = cut_window_length
        self.cut_window_step = cut_window_step

        params =  pickle.load(open(DEFAULT_PARAMS, 'rb'))
        self.meme = params["meme"]
        self.devi = params["devi"]
        
    def save_parameters(self, params_file):
        """
        Save the data processing parameters to a given file.
        
        Parameters
        ----------
        params_file : str
            Path to the parameters file.
        """
        
        params = self.__dict__
        pickle.dump(params, open(params_file, 'wb'))
        
    def load_parameters(self, params_file):
        """
        Load the data processing parameters from a given file.

        Parameters
        ----------
        params_file : str
            Path to the parameters file.
        """
        
        params = pickle.load(open(params_file, 'rb'))
        for attr in params:
            setattr(self, attr, params[attr])
        
    def generate_features_batches(self, audio_files):
        """
        Generate features batches from a list of audio files.
        
        Parameters
        ----------
        params_file : str
            Path to the parameters file.
            
        Returns
        -------
        features_batches : ndarray
            3D array of equally sized batches of features.
        batches_timestamps : ndarray
            3D array of the timestamps and frame numbers of each value in the
            batches.
        ori_frames_length: ndarray
            1D array containing the lengths of the original features frames.
        """
        
        features_frames = generate_features_frames(audio_files,
                                                   self.fgen_window_length,
                                                   self.fgen_window_step)[1]
        
        features_batches, batches_timestamps, ori_frames_length = \
            cut_features(features_frames, self.cut_window_length,
                         self.cut_window_step, self.meme, self.devi)
        
        return features_batches, batches_timestamps, ori_frames_length
        
    def reconstruct_envelopes(self, envelopes_batches, timestamps, ori_frames_length):
        """
        Reconstruct the syllable envelope of each file from the result batches
        of the envelope estimator.
        
        Parameters
        ----------
        features_batches : ndarray
            3D array of equally sized batches of features.
        timestamps : ndarray
            3D array of the timestamps and frame numbers of each value in the
            batches.
        ori_frames_length: ndarray
            1D array containing the lengths of the original features frames.
            
        Returns
        -------
        envelopes : list
            list of the envelope arrays of the audio files.
        """
        
        n_files = len(ori_frames_length)
        envelopes = []
        tot_sums = []
        
        for k in range(n_files):
           envelopes.append(np.zeros(ori_frames_length[k]))
           tot_sums.append(np.zeros(ori_frames_length[k]))
        
        # add (in envelopes) and count (in tot_sums) values in the envelopes
        # batches which come from the same feature and batch
        n_batches = envelopes_batches.shape[0]
        for i in range(n_batches):
            batch_ts = timestamps[i,:,:]
            for j in range(self.cut_window_length):
                frame_nb = batch_ts[j][0]
                ts = batch_ts[j][1]
                if ts > -1:
                    envelopes[frame_nb][ts] += envelopes_batches[i][j]
                    tot_sums[frame_nb][ts] += 1
        
        # compute the mean of values
        for k in range(n_files):
            envelopes[k] =  envelopes[k] / tot_sums[k]
        
        return envelopes


# Utility functions

def scaled_mel_filters(sample_rate, n_fft, n_mel_filters, fmin, fmax):
    """
    Determines the matrix for a mel_filters-filterbank and scales it for each
    filter values to sum up to 1.
    
    Parameters
    ----------
    sample_rate : int
        Sample rate of the signal.
    n_fft : int
        Number of FFT components.
    n_mel_filters : int
        Number of mel_filters bands to generate.
    fmin: float
        Lowest frequency as a fraction of sample rate.
    fmax: float
        highest frequency as a fraction of sample rate.
    
    Returns
    -------
    mel_filters_scaled : ndarray
        2D array representing the mel_filters transform matrix.
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
    
    return mel_filters_scaled


def generate_features_frames(audio_files, window_length, window_step,
                             sample_rate=16000, use_spec_sub=False):
    """
    Determines the energy and MFCCs of each signal.
    
    Remark: in the original code, the energy is computed but never used. Here 
    it is also computed as might be used as additional features.
    
    Parameters
    ----------
    audio_files: list
        List of the audio files (str) to process.
    window_length: int
        Length of the sliding window as a fraction of the sample rate.
    window_step: int
        Step of the sliding window as a fraction of the sample rate.
    sample_rate: int
        Sample rate the audio files should have.
        Defaults to 16000.
    use_spec_sub: boolean
        Use spectral subtraction or not.
        Defaults to False.
    
    Returns
    -------
    E : list
        List of energy array per file.
    F : list
        List of MFCCs matrices (2D array) per file.
    """
    
    window_length = round(window_length * sample_rate)
    window_step = round(window_step * sample_rate)
    window_hamming = np.hamming(window_length)
    n_files = len(audio_files)
    E = []
    F = []

    mel_filters = scaled_mel_filters(sample_rate, window_length, 24, 0, sample_rate/2)
    
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

        # TODO: add spectral subtraction if possible
        
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
        
        E.append(signal_energy)
        F.append(signal_mfcc)
        
        # TODO: add progress bar
        
        i += 1
        
    return E, F


def cut_features(features_frames, window_length, window_step, meme, devi):
    """
    Cut the file's features frames into batches of equal size that can be
    processed by the BLSTM network. This is done because the BLSTM envelope
    estimator needs the input batches to be equally sized.
    The timestamps and original frames length are kept to reconstruct the 
    results of the BLSTM.
    
    Parameters
    ----------
    features_frames : list
        List of MFCCs matrices (2D array) per file.
    window_length : int
        Length of the sliding window.
    window_step : int
        Step of the sliding window.
    meme : float array
        Array applied to features matrices.
    devi : float array
        Array applied to features matrices.
        
    Returns
    -------
    features_batches : ndarray
        3D array of equally sized batches of features.
    timestamps : ndarray
        3D array of the timestamps and frame numbers of each value in the
        batches.
    ori_frames_length: ndarray
        1D array containing the lengths of the original features frames.
    """
    
    # join every file's features matrix together in one matrix
    tot_features = np.concatenate(features_frames)
    tot_length = len(tot_features)
    
    # timestamps keeps track of the number of the origin feature frame and the
    # timestamps of the features
    timestamps = np.zeros((tot_length, 2), dtype=int)
    ori_frames_length = np.zeros(len(features_frames), dtype=int)
    frame_nb = 0
    l = 0
    for f in features_frames:
        frame_len = len(f)
        ori_frames_length[frame_nb] = frame_len
        for i in range(l, l+frame_len):
            timestamps[i] = (frame_nb, i-l)
        frame_nb += 1
        l += frame_len
    
    # apply meme and devi corrections to features (not sure what this is for)
    tot_features = tot_features - meme
    tot_features = tot_features / devi
    
    # add zeros if the length is not a multiple of the step's size
    excess = tot_length % window_step
    if excess != 0:
        tot_features = np.concatenate((tot_features,
                                       np.zeros((window_length - excess, 24))))
        timestamps = np.concatenate((timestamps,
                                       np.full((window_length - excess, 2),
                                               (frame_nb - 1, -1))))
        tot_length = len(tot_features)
    
    # slide window over total_features and append it to features_batches at each step
    n_batches = ((tot_length - window_length) // window_step) + 1
    features_batches = np.zeros((n_batches, window_length, 24))
    batches_timestamps = np.zeros((n_batches, window_length, 2), dtype=int)
    k = 0
    for i_start in range(0, tot_length - window_length + window_step, window_step):
        features_batches[k, :, :] = tot_features[i_start:i_start + window_length, :]
        batches_timestamps[k, :] = timestamps[i_start:i_start + window_length, :]
        k += 1
    
    features_batches[np.isneginf(features_batches)] = 0
    features_batches[features_batches == np.inf] = 0
    features_batches[features_batches == np.NaN] = 0

    # TODO: divide into multiple files if needed
    
    return features_batches, batches_timestamps, ori_frames_length