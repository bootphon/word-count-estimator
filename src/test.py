from syllabification.models.syllabifier import Syllabifier
from syllabification.features import generate_features, cut_features

audio_files = ["../data/sample_1.wav", "../data/sample_2.wav"]
F, E = generate_features.generate_features(audio_files, window_step=0.01)
X, X_ts = cut_features.cut_features(F, "C:/Users/Médéric Carriat/Desktop/wce/models/LSTM_params_BLSTM_fourlang_60_60_augmented_dropout_v2.h5.mat")
syl = Syllabifier("../models/BLSTM_fourlang_60_60_augmented_dropout_v2.h5")
env = syl.predict(X)
