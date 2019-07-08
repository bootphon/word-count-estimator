"""
TODO
"""

import numpy as np
import pickle
from sklearn import linear_model

from word_count_estimation.peakdetect import peakdet
from word_count_estimation.add_features import add_features


class WordCountEstimator:
    
    def __init__(self):
        self.threshold = 0.5
        self.M = np.array([1])
        self.alpha = 1
        self.additional_features = ["duration", "sonority_mean_energy", "sonority_SD_energy"]
        
    def summary(self):
        print("model characteristics")
    
    def load_model_from_file(self, model_file): 
        model = pickle.load(open(model_file, 'rb'))
        for attr in model:
            setattr(self, attr, model[attr])
    
    def train(self, envelopes, target_word_counts, thresholds,
              model_file="../models/word_count_estimator/curr_model.pickle"):
        
        n_files = len(envelopes)
        n_thresholds = len(thresholds)
        estimated_nuclei_counts = np.zeros((n_files, n_thresholds))
        
        # count syllable nuclei per files
        for i in range(n_files):
            for j in range(n_thresholds):
                n_syl_nuclei = len(peakdet(envelopes[i], thresholds[j])[0])
                estimated_nuclei_counts[i, j] = n_syl_nuclei
        
        # determine best threshold
        # PROBLEM: corrcoef not the same
        corvals = np.zeros(n_thresholds)
        for k in range(n_thresholds):
            corvals[k] = np.corrcoef(target_word_counts,
                                     estimated_nuclei_counts[:, k])[0][1]
        opti_k = np.nanargmax(corvals)
        opti_threshold = corvals[opti_k]
        nuclei_counts = estimated_nuclei_counts[:, opti_k]
        
        # create an array X from nuclei_counts and additional features
        X = np.zeros((n_files, 1 + len(self.additional_features)))
        for l in range(n_files):
            X[l, 0] = nuclei_counts[l]
            X[l, 1:] = add_features(envelopes[l], self.additional_features)
        
        # determine M coefficients by linear regression on X and
        # target_word_counts
        lr = linear_model.LinearRegression()
        lr.fit(X, target_word_counts)
        opti_M = lr.coef_
        
        # readjust M by dividing by alpha: the recall of the SAD
        opti_M = opti_M / self.alpha
        print('M', opti_M)
        
        # save results to a pickle file
        model = dict()
        model["alpha"] = self.alpha
        model["M"] = opti_M
        model["threshold"] = opti_threshold
        pickle.dump(model, open(model_file, 'wb'))
        
    def predict(self, envelopes):
        n_files = len(envelopes)
        
        X = np.zeros((n_files, 1 + len(self.additional_features)))
        for k in range(n_files):
            n_syl_nuclei = len(peakdet(envelopes[k], self.threshold)[0])
            X[k, 0] = n_syl_nuclei
            X[k, 1:] = add_features(envelopes[k], self.additional_features)
        
        word_counts = np.matmul(X, self.M)
        
        word_counts[np.isneginf(word_counts)] = 0
        word_counts[word_counts == np.inf] = 0
        word_counts[word_counts == np.NaN] = 0
        
        return word_counts