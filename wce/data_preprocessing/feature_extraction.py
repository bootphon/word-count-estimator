import sys
import yaml
import pkg_resources
import librosa
import numpy as np

try:
    path_meme = "matrices/meme.npy"
    path_devi = "matrices/devi.npy"
    module_name = 'wce.data_preprocessing.feature_extraction'
    meme_f = pkg_resources.resource_stream(module_name, path_meme)
    devi_f = pkg_resources.resource_stream(module_name, path_devi)
    meme = np.load(meme_f)
    devi = np.load(devi_f)
except:
    sys.exit("Problem occuring with matrices files meme and devi.")


class FeatureExtraction:
    """
    Class for feature extraction tasks.

    Attributes
    ----------
    sample_rate : int, optional
        Defaults to 16000.
    window_length : float, optional
        Length of the sliding window as a fraction of the audio files sample rate.
        Defaults to 0.025.
    window_step : float, optional
        Step of the sliding window as a fraction of the audio files sample rate.
        Defaults to 0.01.
    fmin : int, optional
        Lowest frequency (in Hz).
        Defaults to 0.
    fmax : int, optional
        Highest frequency (in Hz).
        Defaults to None. (will become sample_rate/2)
    n_mels : int, optional
        Number of Mel bands to generate.
        Defaults to 24.
    coefs : int, optional
        Number of coefficients.
        Defaults to 24.
    extractor : str, otpional
        Extracting method.
        Default to 'okko'.
    enhancement : bool, optional
        Use speech enhancement.
        Defaults to False.
    use_meme_devi : bool, optional
        Use meme and devi arrays.
        Defaults to True.
    mel_rescale : bool, optional
        Rescale the mel filters.
        Defaults to True.

    Methods
    -------
    save(params_file)
        Save the data processing parameters to a given file.
    load(params_file)
        Load the data processing parameters from a given file.
    rescale_mel_filter(mel_basis)
        Rescale mel filters so values in a band add up to 1. 
    mel_features_librosa(wav)
        Determine the log-mel energies of each signal using the librosa library.
    mel_features_okko(wav)
        Determine the energy and log-mel energies of each signal using Okko's
        method.
    generate_features(audio_file)
        Generate the log-mel energies sequence of an audio file.
    """

    def __init__(self, sample_rate=16000, window_length=0.025,
                 window_step=0.01, fmin=0.0, fmax=None, n_mels=24,
                 coefs=24, extractor='okko', enhancement=False,
                 use_meme_devi=True, mel_rescale=True):

        self.sample_rate = sample_rate
        self.window_length = window_length
        self.window_step = window_step
        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels
        self.coefs = coefs
        self.extractor = extractor
        self.enhancement = enhancement
        self.use_meme_devi = use_meme_devi
        self.mel_rescale = mel_rescale

    def save(self, params_file):
        """
        Save the feature extraction parameters to a file.

        Parameters
        ----------
        params_file : str
            Path to the file to store the parameters.
        """

        try:
            with open(params_file) as f:
                params = yaml.safe_load(f)
        except IOError:
            sys.exit("Wrong parameters file.")

        params['data_processing']['feature_extraction'] = self.__dict__

        with open(params_file, 'w') as f:
            yaml.dump(params, f)

    def load(self, params_file):
        """
        Load the feature extraction parameters from a file.

        Parameters
        ----------
        params_file : str
            Path to the file where the parameters are stored.
        """

        try:
            with open(params_file) as f:
                params = yaml.safe_load(f)
                params = params['data_processing']
        except IOError:
            sys.exit("Wrong parameters file.")

        for attr in params['feature_extraction']:
            setattr(self, attr, params['feature_extraction'][attr])

    @staticmethod
    def rescale_mel_filters(mel_basis):
        """
        Rescale mel filters so values in a band add up to 1. 

        Parameters
        ----------
        mel_basis : ndarray
            2D, mel transform matrix.
            
        Returns
        -------
        mel_basis : ndarray
            2D, scaled mel transform matrix.
        """

        M = []
        for row in mel_basis:
            m = min(x for x in row if x > 0)
            row_normalized = row / m
            row_scaled = row_normalized / row_normalized.sum()
            M.append(row_scaled)
        mel_basis = np.array(M)

        return mel_basis

    def mel_features_librosa(self, wav):
        """
        Determine the log-mel energies of each signal using the librosa library.

        Parameters
        ----------
        wav : ndarray
            1D, waveform.

        Returns
        -------
        features : ndarray
            2D, log-mel energies matrix.
        """

        n_fft = int(self.window_length * self.sample_rate)
        hop_length = int(self.window_step * self.sample_rate)

        S, n_fft = librosa.core.spectrum._spectrogram(y=wav, n_fft=n_fft, power=1,
                                                      hop_length=hop_length)

        mel_basis = librosa.filters.mel(self.sample_rate, n_fft, n_mels=self.n_mels,
                                        htk=True, fmin=self.fmin, fmax=self.fmax)

        if self.mel_rescale:
            mel_basis = self.rescale_mel_filters(mel_basis)

        mel_spectro = np.dot(mel_basis, S)

        features = librosa.core.spectrum.power_to_db(mel_spectro)[:self.coefs]
        features = features.transpose()

        return features

    def mel_features_okko(self, wav):
        """
        Determine the energy and log-mel energies of each signal using the
        method in the original repo.

        Parameters
        ----------
        wav : ndarray
            1D, waveform.

        Returns
        -------
        total_mel_energies : ndarray
            2D, log-mel energies matrix.
        """

        wav_len = len(wav)
        n_fft = round(self.window_length * self.sample_rate)
        hop_length = round(self.window_step * self.sample_rate)
        window_hamming = np.hamming(n_fft)

        mel_basis = librosa.filters.mel(self.sample_rate, n_fft, self.n_mels,
                                        self.fmin, self.fmax, 1)

        if self.mel_rescale:
            mel_basis = self.rescale_mel_filters(mel_basis)

        # add 0s at both sides of the wav
        wav = np.concatenate((wav, [0] * (n_fft // 2)))
        wav = np.concatenate(([0] * (n_fft // 2), wav))

        # slide window over wav and compute energy and MFFCs at each step
        n_windows = wav_len // hop_length + 1
        total_mel_energies = np.zeros((n_windows, mel_basis.shape[0]))
        total_energy = np.zeros(n_windows)

        j = 0
        for i_start in range(0, wav_len, hop_length):
            window = wav[i_start:i_start+n_fft] * window_hamming
            fft_magnitude = np.abs(np.fft.rfft(window))
            mel_energies = 20 * np.log10(np.matmul(mel_basis, fft_magnitude))

            total_mel_energies[j, :] = mel_energies
            total_energy[j] = sum(fft_magnitude)
            j += 1
        
        total_mel_energies = total_mel_energies[:, :self.coefs]

        return total_mel_energies

    def generate_features(self, audio_file):
        """
        Generate the a matrix of features for an audio file.

        Parameters
        ----------
        audio_file : str
            Path to the audio file to process.

        Returns
        -------
        features : ndarray
            2D, matrix of features: MFCC sequence.
        """

        wav, sr = librosa.load(audio_file, self.sample_rate)

        # TODO: add speech enhancement
        if self.enhancement:
            pass

        if self.extractor == 'librosa':
            features = self.mel_features_librosa(wav)
        elif self.extractor == 'okko':
            features = self.mel_features_okko(wav)
        else:
            sys.exit("Indicate a valid feature extractor name in config file.")

        if self.use_meme_devi:
            features = features - meme
            features = features / devi

        return features

