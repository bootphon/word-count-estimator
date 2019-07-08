import matplotlib.pyplot as plt

from envelope_estimation.models.envelope_estimator import EnvelopeEstimator
from envelope_estimation.models import reconstruct_envelope as re
from envelope_estimation.features import generate_features as gf
from envelope_estimation.features import cut_features as cf
from word_count_estimation.word_count_estimator import WordCountEstimator

import numpy as np

audio_files = ["../data/sample_1.wav", "../data/sample_2.wav"]
F, E = gf.generate_features(audio_files, window_step=0.01)
X, X_ts, fl = cf.cut_features(F, "C:/Users/Médéric Carriat/Desktop/wce/models/envelope_estimator/LSTM_params_BLSTM_fourlang_60_60_augmented_dropout_v2.h5.mat")
model = EnvelopeEstimator()
#syl.initialize_BLSTM_model(X.shape[1:])
model.load_model_from_file("../models/envelope_estimator/BLSTM_fourlang_60_60_augmented_dropout_v2.h5")
model.summary()
env_windows = model.predict(X)
env = re.reconstruct_envelope(env_windows, X_ts, fl, 300)

thr = np.concatenate((np.linspace(0.0001, 0.0009, 9),
                      np.array([0.001]),
                      np.linspace(0.002, 0.01, 9),
                      np.linspace(0.012, 0.492, 25),
                      np.linspace(0.6, 1, 5)))



wce = WordCountEstimator()
train = np.array([1, 4])
wce.train(env, train, thr)
wce.load_model_from_file("../models/word_count_estimator/curr_model.pickle")
n = wce.predict(env)
