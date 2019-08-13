import os
import numpy as np
import yaml


class Batch():
    """
    Class for batch management tasks.

    Attributes
    ----------
    window_length : int
        Length of the sliding window.
        Defaults to 300.
    window_step : int
        Step of the sliding window.
        Defaults to 100.

    Methods
    -------
    save(params_file)
        Save the data processing parameters to a given file.
    load(params_file)
        Load the data processing parameters from a given file.
    generate_batch(features)
        Generate a batch of overlapping feature chunks from a feature matrix.
    reconstruct_envelopes(envelope_batch, timestamps, ori_length)
        Reconstruct the syllable envelope of a waveform from the envelope batch.
    """

    def __init__(self, window_length=300, window_step=100):

        self.window_length = window_length
        self.window_step = window_step

    def save(self, params_file):
        """
        Save the batch generation parameters to a file.

        Parameters
        ----------
        params_file : str
            Path to the file to store the parameters.
        """

        try:
            with open(params_file) as f:
                params = yaml.load(f)[0]
        except IOError:
            print("Wrong parameters file.")

        params['data_prosessing']['batch'] = self.__dict__

        with open(params_file, 'w') as f:
            yaml.dump(params, f)

    def load(self, params_file):
        """
        Load the batch generation parameters from a file.

        Parameters
        ----------
        params_file : str
            Path to the file where the parameters are stored.
        """

        try:
            with open(params_file) as f:
                params = yaml.load(f)
                params = params['data_processing']
        except IOError:
            print("Wrong parameters file.")

        for attr in params['batch']:
            setattr(self, attr, params['batch'][attr])

    def generate_batch(self, features_list):
        """
        Generate a batch of overlapping feature chunks from a feature matrices
        list.
        Timestamps and original length are returned as well as the batch for
        future reconstruction.

        Parameters
        ----------
        features : ndarray
            2D, matrix of features of a waveform.

        Returns
        -------
        batch : ndarray
            3D, array of feature chunks of equal size.
        timestamps : ndarray
            1D, array of the original timestamps of each feature column in the
            batch.
        wav_lengths : int
            Length of the original audio waveform.
        """

        tot_features = np.concatenate(features_list)
        tot_length = len(tot_features)

        batch = []
        batch_timestamps = []

        timestamps = np.zeros((tot_length, 2), dtype=int)
        wav_lengths = np.zeros(len(features_list), dtype=int)
        wav_nb = 0
        l = 0
        for features_mat in features_list:

            wav_len = len(features_mat)
            wav_lengths[wav_nb] = wav_len
            
            for i in range(l, l+wav_len):
                timestamps[i] = (wav_nb, i-l)

            wav_nb += 1
            l += wav_len

        i = 0
        while i < tot_length:
            
            if i + self.window_length > tot_length:
                chunk = np.zeros((self.window_length, tot_features.shape[1]))
                chunk[:tot_length-i] = tot_features[i:tot_length]
                ts = np.full((self.window_length, 2), (-1, -1))
                ts[:tot_length-i] = timestamps[i:tot_length]
            else:
                chunk = tot_features[i:i+self.window_length]
                ts = timestamps[i:i+self.window_length]

            batch.append(chunk)
            batch_timestamps.append(ts)

            i += self.window_step

        batch = np.array(batch)
        batch_timestamps = np.array(batch_timestamps)

        return batch, batch_timestamps, wav_lengths

    def reconstruct_envelope(self, envelope_batch, timestamps, wav_lengths):
        """
        Reconstruct the syllable envelope of a waveform from an envelope batch.

        Parameters
        ----------
        envelope_batch : ndarray
            3D, array of envelope segments of equal size.
        timestamps : ndarray
            1D, array of the original timestamps of segment's value in the
            batch.
        wav_length : int
            Length of the original audio waveforms.

        Returns
        -------
        envelope : ndarray
            1D, reconstructed envelope.
        """

        n_wav = len(wav_lengths)
        envelopes = []
        tot_counts = []

        for k in range(n_wav):
            envelopes.append(np.zeros(wav_lengths[k]))
            tot_counts.append(np.zeros(wav_lengths[k]))

        n_segments = envelope_batch.shape[0]
        for i in range(n_segments):
            for j in range(self.window_length):
                # -1 marks the zeros that have been added to fit the window size
                # and are to be ignored.
                wav_nb, ori_ts = timestamps[i, j]
                if ori_ts > -1:
                    envelopes[wav_nb][ori_ts] += envelope_batch[i][j]
                    tot_counts[wav_nb][ori_ts] += 1

        # compute the mean of the values.
        for k in range(n_wav):
            envelopes[k] = envelopes[k] / tot_counts[k]

        return envelopes

