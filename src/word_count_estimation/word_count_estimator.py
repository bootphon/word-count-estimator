"""
TODO
"""

import numpy as np
import pickle
from sklearn import linear_model
import statsmodels.api as sm
from math import sqrt

from word_count_estimation.peakdetect import peakdet
from word_count_estimation.add_features import add_features


class WordCountEstimator:
    
    def __init__(self):
        self.threshold = 0.5
        self.M = np.array([1])
        self.alpha = 1
        self.additional_features = []
        
    def summary(self):
        print("model characteristics")
    
    def load_model_from_file(self, model_file): 
        model = pickle.load(open(model_file, 'rb'))
        for attr in model:
            setattr(self, attr, model[attr])
    
    def train(self, envelopes, target_word_counts, thresholds,
              model_file="../models/word_count_estimator/curr_model.pickle"):
        
        self.additional_features = ["duration",
                                    "sonority_mean_energy",
                                    "sonority_SD_energy"]
        
        n_files = len(envelopes)
        n_thresholds = len(thresholds)
        
        # count syllable nuclei per files
        estimated_nuclei_counts = np.zeros((n_files, n_thresholds))
        for i in range(n_files):
            for j in range(n_thresholds):
                n_syl_nuclei = len(peakdet(envelopes[i], thresholds[j])[0])
                estimated_nuclei_counts[i, j] = n_syl_nuclei
        
        for j in range(n_thresholds):
            print(thresholds[j], estimated_nuclei_counts[:,j])
        
        # determine best threshold
        # PROBLEM: corrcoef not the same
        corvals = np.zeros(n_thresholds)
        for k in range(n_thresholds):
            corvals[k] = np.corrcoef(target_word_counts,
                                     estimated_nuclei_counts[:, k],
                                     rowvar=False)[0][1]

        print("corvals", corvals)
        try:
            opti_k = np.nanargmax(corvals)
        except:
            opti_k = 0
        opti_threshold = thresholds[opti_k]
        nuclei_counts = estimated_nuclei_counts[:, opti_k]
        
        print("opti_counts", nuclei_counts)
        
        # create an array X from nuclei_counts and additional features
        X = np.zeros((n_files, 1 + len(self.additional_features)))
        for l in range(n_files):
            X[l, 0] = nuclei_counts[l]
            X[l, 1:] = add_features(envelopes[l], self.additional_features)
        
        # determine M coefficients by multpiple linear regression on X and
        # target_word_counts
        X = sm.add_constant(X)
        est = sm.OLS(target_word_counts, X).fit()
        opti_M = est.params

        # readjust M by dividing by alpha: the recall of the SAD
        opti_M = opti_M / self.alpha
        print('M', opti_M)
        
        print("M", opti_M)
        
        # compute RMSE
        estimated_word_counts = np.matmul(X, opti_M)
        a = estimated_word_counts[np.where(target_word_counts > 0)]
        b = target_word_counts[np.where(target_word_counts > 0)]
        RMSE_train = sqrt(np.square(np.mean(((a-b) / b))))*100

        print("Relative RMSE error on training set: {:.2f} per SAD segment".format(RMSE_train))
        
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