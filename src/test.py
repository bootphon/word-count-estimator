from syllabification.models.syllabifier import Syllabifier
from syllabification.features import generate_features as gf
from syllabification.features import cut_features as cf

audio_files = ["../data/sample_1.wav", "../data/sample_2.wav"]
F, E = gf.generate_features(audio_files, window_step=0.01)
X, X_ts = cf.cut_features(F, "C:/Users/Médéric Carriat/Desktop/wce/models/LSTM_params_BLSTM_fourlang_60_60_augmented_dropout_v2.h5.mat")
syl = Syllabifier()
syl.initialize_BLSTM_model(X.shape[1:])
syl.summary()
env = syl.predict(X)
