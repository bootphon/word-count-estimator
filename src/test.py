"""
Test file.

"""

import numpy as np
import glob
import matplotlib.pyplot as plt

from envelope_estimation import DataProcessing, EnvelopeEstimator
from word_count_estimation import WordCountEstimator


audio_files_train = glob.glob("../data/3/*.wav")
audio_files_train.sort(key= lambda x: int(x.split("\\")[1][:-4]))
print(audio_files_train)
dp = DataProcessing()
X_train, timestamps, ori_frames_length = dp.generate_features_batches(audio_files_train)
model = EnvelopeEstimator()
#syl.initialize_BLSTM_model(X.shape[1:])
model.load_model_from_file("../models/envelope_estimator/BLSTM_fourlang_60_60_augmented_dropout_v2.h5")
model.summary()
env_windows = model.predict(X_train)
env = dp.reconstruct_envelopes(env_windows, timestamps, ori_frames_length)

audio_files_test = ["../data/1.wav"]
X_test, timestamps_t, ori_frames_length_t = dp.generate_features_batches(audio_files_test)
env_windows_t = model.predict(X_test)
env_t = dp.reconstruct_envelopes(env_windows_t, timestamps_t, ori_frames_length_t)


thr = np.concatenate((np.linspace(0.0001, 0.0009, 9),
                      np.array([0.001]),
                      np.linspace(0.002, 0.01, 9),
                      np.linspace(0.012, 0.492, 25),
                      np.linspace(0.6, 1, 5)))



wce = WordCountEstimator()
train = np.array([84, 138, 91, 23, 217, 116, 96, 148, 47, 105, 141, 549, 224, 352, 258])
wce.train(env, train, thr)
wce.load_model_from_file("../models/word_count_estimator/curr_model.pickle")
n = wce.predict(env_t)
n_tr = wce.predict(env)
