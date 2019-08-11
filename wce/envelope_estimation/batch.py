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
    save_parameters(params_file)
        Save the data processing parameters to a given file.
    load_parameters(params_file)
        Load the data processing parameters from a given file.
    generate_batch(features)
        Generate a batch of overlapping feature chunks from a feature matrix.
    reconstruct_envelopes(envelope_batch, timestamps, ori_length)
        Reconstruct the syllable envelope of a waveform from the envelope batch.
    """

    def __init__(self, window_length=300, window_step=100):
        """
        Parameters
        ----------
        window_length : int
            Length of the sliding window for the cutting operation.
            Defaults to 300.
        window_step : int
            Step of the sliding window for the cutting operation.
            Defaults to 100.
        """

        self.window_length = window_length
        self.window_step = window_step

    def save_parameters(self, params_file):
        """
        Save the batch generation parameters to a file.

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

        params['batch_generation'] = self.__dict__

        with open(params_file, 'w') as f:
            yaml.dump(params, f)

    def load_parameters(self, params_file):
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
        except IOError:
            print("Wrong parameters file.")

        for attr in params['batch_generation']:
            setattr(self, attr, params[attr])

    def generate_batch(self, features):
        """
        Generate a batch of overlapping feature chunks from a feature matrix.
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
        f_length : int
            Length of the original audio waveform.
        """

        f_length = len(features)
        batch = []
        timestamps = []

        i = 0
        while i < f_length:
            
            if i + self.window_length > f_length:
                chunk = np.zeros((self.window_length, features.shape[1]))
                chunk[:f_length-i] = features[i:f_length]
                ts = np.full(self.window_length, -1)
                ts[:f_length-i] = np.arange(i, f_length)
            else:
                chunk = features[i:i+self.window_length]
                ts = np.arange(i, i+self.window_length)

            batch.append(chunk)
            timestamps.append(ts)
            i += self.window_step

        batch = np.array(batch)
        timestamps = np.array(timestamps)

        return batch, timestamps, f_length

    def reconstruct_envelope(self, envelope_batch, timestamps, ori_length):
        """
        Reconstruct the syllable envelope of a waveform from an envelope batch.

        Parameters
        ----------
        envelope_batch : ndarray
            3D, array of envelope segments of equal size.
        timestamps : ndarray
            1D, array of the original timestamps of segment's value in the
            batch.
        ori_length : int
            Length of the original audio waveform.

        Returns
        -------
        envelope : ndarray
            1D, reconstructed envelope.
        """
        
        envelope = np.zeros(ori_length)
        tot_counts = np.zeros(ori_length)
        n_segments = envelope_batch.shape[0]

        for i in range(n_segments):
            for j in range(self.window_length):
                # -1 marks the zeros that have been added to fit the window size
                # and are to be ignored.
                ori_ts = timestamps[i, j]
                if ori_ts > -1:
                    envelope[ori_ts] += envelope_batch[i][j]
                    tot_counts[ori_ts] += 1

        # compute the mean of the values.
        envelope = envelope / tot_counts

        return envelope

