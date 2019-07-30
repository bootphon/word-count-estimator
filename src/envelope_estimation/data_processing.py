from librosa import filters, core
import numpy as np
import soundfile as sf
import pickle
from tqdm import tqdm
#from logmmse import logmmse


DEFAULT_PARAMS="../models/envelope_estimator/default_data_processing_params.pickle"


class DataProcessing():
    """
    Class for data processing tasks.
    
    Its purpose is to:
        - extract features from the input audio files and create a batch from
        those features.
        - once the batch has been processed by the syllable envelope estimator,
        reconstruct the syllable envelope of each file from it.
    
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
    generate_features_batch(self, audio_files)
        Generate a features batch from a list of audio files.
    reconstruct_envelopes(envelopes_batch, timestamps, ori_frames_length)
        Reconstruct the syllable envelope of each file from the result batch
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

        # meme and devi are constant and directly loaded from default file.
        params =  pickle.load(open(DEFAULT_PARAMS, 'rb'))
        self.meme = params["meme"]
        self.devi = params["devi"]
        
    def save_parameters(self, params_file):
        """
        Save the data processing parameters to a file.
        
        Parameters
        ----------
        params_file : str
            Path to the file to store the parameters.
        """

        try:
            params = self.__dict__
            pickle.dump(params, open(params_file, 'wb'))
        except IOError:
            print("Wrong input file for the data processing parameters.")
        
    def load_parameters(self, params_file):
        """
        Load the data processing parameters from a file.

        Parameters
        ----------
        params_file : str
            Path to the file where the parameters are stored.
        """

        try:
            params = pickle.load(open(params_file, 'rb'))
            for attr in params:
                setattr(self, attr, params[attr])
        except IOError:
            print("Wrong input file for the data processing parameters.")
        
    def generate_features_batch(self, audio_files):
        """
        Generate a features batch from a list of audio files.
        
        Parameters
        ----------
        audio_files : list
            List of the paths to the audio files to process.
            
        Returns
        -------
        feature_batch : ndarray
            3D, array of equally sized frames of features.
        batch_timestamps : ndarray
            3D, array of the original timestamps and file number of each features
            column in the frames of the batch.
        files_length: ndarray
            1D, array containing the lengths of the audio files.
        """
        
        print("Generating the feature batch.")
        
        files_features = generate_features(audio_files,
                                           self.fgen_window_length,
                                           self.fgen_window_step)[1]
        
        feature_batch, batch_timestamps, files_length = \
                    cut_features(files_features, self.cut_window_length,
                                 self.cut_window_step, self.meme, self.devi)
        
        return feature_batch, batch_timestamps, files_length
        
    def reconstruct_envelopes(self, envelope_batch, batch_timestamps, files_length):
        """
        Reconstruct the syllable envelope of each file from the envelope batch.
        The envelope batch is the result of the envelope estimation on the 
        feature batch.

        Parameters
        ----------
        envelope_batch : ndarray
            2D, array of the envelope arrays.
        batch_timestamps : ndarray
            2D, array of the original timestamps and file number of each features
            column in the frames of the batch.
        files_length: ndarray
            1D, array containing the lengths of the audio files.
        
        Returns
        -------
        envelopes : list
            List of the envelope arrays of the audio files.
        """
        
        n_files = len(files_length)
        envelopes = []
        tot_sums = []
         
        for k in range(n_files):
           envelopes.append(np.zeros(files_length[k]))
           tot_sums.append(np.zeros(files_length[k]))
        
        # add (in envelopes) and count (in tot_sums) values in the envelopes
        # frames which come from the same timestamp in the same file.
        n_frames = envelope_batch.shape[0]
        frame_length = self.cut_window_length
        for i in range(n_frames):
            frame_ts = batch_timestamps[i]
            for j in range(frame_length):
                file_nb, ori_ts = frame_ts[j]
                if ori_ts > -1:                     
                    # -1 marks the zeros that have been added to fit the window size
                    # during the feature generation, hence the values are ignored.
                    envelopes[file_nb][ori_ts] += envelope_batch[i][j]
                    tot_sums[file_nb][ori_ts] += 1
        
        # compute the mean of the values.
        for k in range(n_files):
            envelopes[k] =  envelopes[k] / tot_sums[k]
        
        return envelopes


# Utility functions

def scaled_mel_filters(sample_rate, n_fft, n_mel_filters, fmin, fmax):
    """
    Determine the matrix for a mel_filters-filterbank and scales it for each
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
        Highest frequency as a fraction of sample rate.
    
    Returns
    -------
    mel_filters_scaled : ndarray
        2D, array representing the mel_filters transform matrix.
    """

    mel_filters = filters.mel(sample_rate, n_fft, n_mel_filters, fmin, fmax, 1)
    
    M = []
    for row in mel_filters:
        m = min(x for x in row if x > 0)
        row_normalized = row / m
        row_scaled = row_normalized / row_normalized.sum()
        M.append(row_scaled)
    mel_filters_scaled = np.array(M)
    
    return mel_filters_scaled


def generate_features(audio_files, window_length, window_step,
                      sample_rate=16000, speech_enhance=False):
    """
    Determine the energy and MFCCs of each signal.
    
    Remark: in the original code, the energy is computed but never used. Here 
    it is also computed as it might be used as additional features.
    
    Parameters
    ----------
    audio_files : list
        List of the paths to the audio files to process.
    window_length: int
        Length of the sliding window as a fraction of the sample rate.
    window_step: int
        Step of the sliding window as a fraction of the sample rate.
    sample_rate: int
        Sample rate the audio files should have.
        Defaults to 16000.
    speech_enhance: boolean
        Use speech enhancing. Current function slows process.
        Defaults to False.
    
    Returns
    -------
    E : list
        List of the energy arrays of the audio files.
    F : list
        List of the MFCCs frames (2D array) of the audio files.
    """
    
    window_length = round(window_length * sample_rate)
    window_step = round(window_step * sample_rate)
    window_hamming = np.hamming(window_length)
    n_files = len(audio_files)
    E = []
    F = []

    mel_filters = scaled_mel_filters(sample_rate, window_length, 24, 0, sample_rate/2)
    
    pbar = tqdm(total=n_files)
    
    i = 0
    while i < n_files:
        f = audio_files[i]
        signal, f_sample_rate = sf.read(f)
        signal_len = len(signal)
        
        if f_sample_rate != sample_rate:
            signal = core.resample(signal, f_sample_rate, sample_rate)
            f_sample_rate = sample_rate
        
        #if speech_enhance:
        #    signal = logmmse(signal, sample_rate)[0]
        #    signal_len = len(signal)
        
        # add 0s at both sides of the signal
        signal = np.concatenate((signal, [0] * (window_length // 2)))
        signal = np.concatenate(([0] * (window_length // 2), signal))

        # TODO: add spectral subtraction if possible. For now it is the logmmse
        # algo as I did not find an equivalent for specsub.m in Python.
        
        # slide window over signal and compute energy and MFFCs at each step
        n_windows = signal_len // window_step + 1
        signal_mfccs = np.zeros((n_windows, mel_filters.shape[0]))
        signal_energy = np.zeros(n_windows)
        
        j = 0
        for i_start in range(0, signal_len, window_step):
            window = signal[i_start:i_start + window_length] * window_hamming
            fft_magnitude = np.abs(np.fft.rfft(window))
            mfccs = 20 * np.log10(np.matmul(mel_filters, fft_magnitude))
            
            signal_mfccs[j, :] = mfccs
            signal_energy[j] = sum(fft_magnitude)
            j += 1
        
        E.append(signal_energy)
        F.append(signal_mfccs)
        
        i += 1
        pbar.update(1)
        
    pbar.close()
    
    return E, F


def cut_features(files_features, window_length, window_step, meme, devi):
    """
    Cut the files' features into a batch of overlapping frames of equal size 
    that can be processed by the BLSTM network. This is done because the BLSTM 
    envelope estimator needs the input batch to be equally sized.
    The timestamp and filenumber of each feature column are kept to reconstruct
    the results of the BLSTM.
    
    Parameters
    ----------
    files_features : list
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
    feature_batch : ndarray
        3D array of equally sized batch of features.
    timestamps : ndarray
        3D array of the timestamps and frame numbers of each value in the
        batch.
    files_length: ndarray
        1D array containing the lengths of the original features frames.
    """
    
    # join every file's features matrix together in one matrix
    tot_features = np.concatenate(files_features)
    tot_length = len(tot_features)
    
    # timestamps keeps track of the number of the origin file and the
    # timestamp of the feature column
    timestamps = np.zeros((tot_length, 2), dtype=int)
    files_length = np.zeros(len(files_features), dtype=int)
    file_nb = 0
    l = 0
    for frame in files_features:
        frame_len = len(frame)
        files_length[file_nb] = frame_len
        for i in range(l, l+frame_len):
            timestamps[i] = (file_nb, i-l)
        file_nb += 1
        l += frame_len
    
    # apply meme and devi corrections to features (not sure what this is for)
    tot_features = tot_features - meme
    tot_features = tot_features / devi
    
    # add zeros if the length is not a multiple of the step's size
    excess = tot_length % window_step
    if excess != 0:
        f_excess = np.zeros((window_length - excess, 24))
        tot_features = np.concatenate((tot_features, f_excess))
        ts_excess = np.full((window_length - excess, 2), (file_nb-1, -1))
        timestamps = np.concatenate((timestamps, ts_excess))
        tot_length = len(tot_features)
    
    # slide window over total_features and append it to feature_batch at each step
    n_frames = (tot_length - window_length)//window_step + 1
    feature_batch = np.zeros((n_frames, window_length, 24))
    batch_timestamps = np.zeros((n_frames, window_length, 2), dtype=int)
    k = 0
    for i_start in range(0, tot_length - window_length + window_step, window_step):
        feature_batch[k] = tot_features[i_start:i_start + window_length, :]
        batch_timestamps[k] = timestamps[i_start:i_start + window_length, :]
        k += 1
    
    feature_batch[np.isneginf(feature_batch)] = 0
    feature_batch[feature_batch == np.inf] = 0
    feature_batch[feature_batch == np.NaN] = 0

    return feature_batch, batch_timestamps, files_length

