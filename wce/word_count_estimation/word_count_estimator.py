import os, sys
import pickle
from math import sqrt
from statsmodels.api import OLS, add_constant
import numpy as np

from .utility import peakdet

THR = np.concatenate((np.linspace(0.0001, 0.0009, 9),
                      np.array([0.001]),
                      np.linspace(0.002, 0.01, 9),
                      np.linspace(0.012, 0.492, 25),
                      np.linspace(0.6, 1, 5)))

class WordCountEstimator:
    """
    Model to estimate the word counts of audio files from their syllable
    envelopes.

    Given a syllable envelope, the number of syllables nuclei are determined
    using a peak picking algorithm. Then a linear mapping of the nuclei count
    to the word count is made.
    Both the peak picking and linear mapping can be trained/adapted if the
    number of words per files are provided.

    Attributes
    ----------
    threshold : float
        Minimum value separating a maximum and its left neighbour for this
        maximum to be considered a peak.
    lin_reg : <statsmodels.regression.linear_model>
        OLS model to map the nuclei count to the word count.
    alpha : float
        Recall of the SAD to readjust the word counts.
    additional_features : list
        List of features (str) to add to the estimated word count for the linear
        mapping training.

    Methods
    -------
    summary()
        Print a summary of the model.
    save_model(model_file)
        Save the model to a given file.
    load_model(model_file)
        Load the model from a given file.
    add_features(envelope)
        Compute the desired features from a syllable envelope.
    train(envelopes, target_word_counts, thresholds, model_file)
        Train the model given syllable envelopes and their respective target
        word counts. The resulting model is saved to model_file.
    predict(envelopes)
        Predicts the word counts for a given list of syllable envelopes.
    """

    def __init__(self, threshold=0.5, alpha=1, additional_features=[]):

        self.threshold = threshold
        self.lin_reg = OLS([1], [1]).fit()
        self.alpha = alpha
        self.additional_features = additional_features

    def summary(self):
        """
        Print a summary of the model.
        """

        print("Summary of WCE model:")
        for attr in self.__dict__:
            if attr != "lin_reg":
                print(attr, self.__dict__[attr])
        print("lin_reg coefficients", self.__dict__["lin_reg"].params)

    def save_model(self, model_file):
        """
        Save the model to a given file.

        Parameters
        ----------
        model_file : str
            Path to the model's file.
        """

        pickle.dump(self.__dict__, open(model_file, 'wb'))

    def load_model(self, model_file):
        """
        Load the model from a given file.

        Parameters
        ----------
        model_file : str
            Path to the model's file.
        """

        model = pickle.load(open(model_file, 'rb'))
        for attr in model:
            setattr(self, attr, model[attr])

    def add_features(self, envelope):
        """
        Compute the desired features from a syllable envelope.

        Parameters
        ----------
        envelope : ndarray
            1D array containing the values of the syllable envelope.

        Returns
        -------
        features : list
            List of the computed features.
        """

        features = []

        if 'duration' in self.additional_features:
            durs = len(envelope) / 100
            features.append(durs)

        if 'sonority_total_energy' in self.additional_features:
            en_sonor_total = np.sum(envelope)
            features.append(en_sonor_total)

        if 'sonority_mean_energy' in self.additional_features:
            en_sonor_mean = np.mean(envelope)
            features.append(en_sonor_mean)

        if 'sonority_SD_energy' in self.additional_features:
            en_sonor_sd = np.std(envelope)
            features.append(en_sonor_sd)

        # TODO: Possibility to add more.

        return features

    def train(self, envelopes, target_word_counts, model_file, thresholds=THR):
        """
        Train the model given syllable envelopes and their respective target
        word counts. The resulting model is then saved to model_file.

        Training works as follows:
            - estimate the number of syllable nuclei per envelope according to
            different thresholds and chose the threshold that produces the best
            correlation between the estimated number of nuclei and the target
            number of word counts.
            - using the estimated number of nuclei resulting from the optimal
            threshold, determine the coefficients of the linear mapping.

        Parameters
        ----------
        envelopes : ndarray
            2D, array of envelope per waveform.
        target_word_counts : list
            List of the word count per envelope.
        model_file: str
            Path of where to save the model file.
        thresholds : list
            List of the thresholds values to test for the model adaptation.
        """

        self.additional_features = ["duration", "sonority_mean_energy",
                                    "sonority_SD_energy"]

        n_envelopes = len(envelopes)
        n_thresholds = len(thresholds)

        # count syllable nuclei per files
        estimated_nuclei_counts = np.zeros((n_envelopes, n_thresholds))
        for i in range(n_envelopes):
            for j in range(n_thresholds):
                n_syl_nuclei = len(peakdet(envelopes[i], thresholds[j])[0])
                estimated_nuclei_counts[i, j] = n_syl_nuclei

        # determine best threshold
        corvals = np.zeros(n_thresholds)
        for k in range(n_thresholds):
            all_zeros = not np.any(estimated_nuclei_counts[:, k])
            if not all_zeros:
                corvals[k] = np.corrcoef(target_word_counts,
                                         estimated_nuclei_counts[:, k],
                                         rowvar=False)[0][1]

        try:
            opti_k = np.nanargmax(corvals)
        except:
            opti_k = 0
        opti_threshold = thresholds[opti_k]
        nuclei_counts = estimated_nuclei_counts[:, opti_k]

        # create an array X from nuclei_counts and additional features
        X = np.zeros((n_envelopes, 1 + len(self.additional_features)))
        for l in range(n_envelopes):
            X[l, 0] = nuclei_counts[l]
            X[l, 1:] = self.add_features(envelopes[l])
        X = add_constant(X, has_constant='add')

        # multiple linear regression on X and target_word_counts
        self.lin_reg = OLS(target_word_counts, X).fit()

        # readjust coefficients by dividing by alpha: the recall of the SAD
        self.lin_reg.params /= self.alpha

        self.threshold = opti_threshold
        self.save_model(model_file)

    def predict(self, envelopes):
        """
        Predicts the word counts for a given list of syllable envelopes.

        Parameters
        ----------
        envelopes : ndarray
            2D, array of envelope per file.

        Returns
        -------
        word_counts : ndarray
            2D array containing the estimated word count per audio file/envelope.
        """

        n_envelopes = len(envelopes)

        X = np.zeros((n_envelopes, 1 + len(self.additional_features)))
        for k in range(n_envelopes):
            n_syl_nuclei = len(peakdet(envelopes[k], self.threshold)[0])
            X[k, 0] = n_syl_nuclei
            X[k, 1:] = self.add_features(envelopes[k])

        if len(self.lin_reg.params) > 1:
            X = add_constant(X, has_constant='add')

        word_counts = self.lin_reg.predict(X)

        return word_counts

