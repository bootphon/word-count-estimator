import yaml
import pkg_resources
import librosa
import numpy as np

try:
    path_meme = "matrices/meme.npy"
    path_devi = "matrices/meme.npy"
    module_name = 'wce.envelope_estimation.processing'
    meme_f = pkg_resources.resource_stream(module_name, path_meme)
    devi_f = pkg_resources.resource_stream(module_name, path_devi)
    meme = np.load(meme_f)
    devi = np.load(devi_f)
except:
    print("Problem occuring with matrices files meme and devi.")


class DataProcessing():
    """
    Class for data processing tasks.

    Attributes
    ----------

    Methods
    -------
    save_parameters(params_file)
        Save the data processing parameters to a given file.
    load_parameters(params_file)
        Load the data processing parameters from a given file.
    generate_features(audio_file)
        Generate the MFCC sequence of an audio file.
    """

    def __init__(self, sample_rate=16000, window_length=0.025,
                 window_step=0.01, fmin=0.0, fmax=None, n_mels=24,
                 coefs=24, enhancement=False, use_meme_devi=True,
                 mel_normalize=True):
        """
        Parameters
        ----------
        sample_rate :
        window_length : float
            Length of the sliding window for the features generation as a fraction
            of the audio files sample rate.
            Defaults to 0.025.
        window_step : float
            Step of the sliding window for the features generation as a fraction
            of the audio files sample rate.
            Defaults to 0.01.
        fmin :
        fmax : 
        n_mels :
        coefs :
        enhancement :
        use_meme_devi :
        """

        self.sample_rate = sample_rate
        self.window_length = window_length
        self.window_step = window_step
        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels
        self.coefs = coefs
        self.enhancement = enhancement
        self.use_meme_devi = use_meme_devi
        self.mel_normalize = mel_normalize

    def save_parameters(self, params_file):
        """
        Save the data processing parameters to a file.

        Parameters
        ----------
        params_file : str
            Path to the file to store the parameters.
        """

        try:
            with open(params_file) as f:
                params = yaml.load(f)
        except IOError:
            print("Wrong parameters file.")

        params['data_processing'] = self.__dict__

        with open(params_file, 'w') as f:
            yaml.dump(params, f)

    def load_parameters(self, params_file):
        """
        Load the data processing parameters from a file.

        Parameters
        ----------
        params_file : str
            Path to the file where the parameters are stored.
        """

        try:
            with open(params_file) as f:
                params = yaml.load(f)
        except IOError:
            print("Wrong parameters file.")

        for attr in params['data_processing']:
            setattr(self, attr, params[attr])

    def generate_features(self, audio_file):
        """
        Generate the MFCC sequence of an audio file.

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

        n_fft = int(self.window_length * sr)
        hop_length = int(self.window_step * sr)

        S, n_fft = librosa.core.spectrum._spectrogram(y=wav, n_fft=n_fft, power=1,
                                                      hop_length=hop_length)

        mel_basis = librosa.filters.mel(sr, n_fft, n_mels=self.n_mels, htk=True,
                                        fmin=self.fmin, fmax=self.fmax)

        if self.mel_normalize:
            M = []
            for row in mel_basis:
                m = min(x for x in row if x > 0)
                row_normalized = row / m
                row_scaled = row_normalized / row_normalized.sum()
                M.append(row_scaled)
            mel_basis = np.array(M)

        mel_spectro = np.dot(mel_basis, S)

        features = librosa.core.spectrum.power_to_db(mel_spectro)[:self.coefs]
        features = features.transpose()

        if self.use_meme_devi:
            features = features - meme
            features = features / devi

        return features

