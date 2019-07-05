import matplotlib.pyplot as plt

from syllabification.models.syllabifier import Syllabifier
from syllabification.models import get_syllables as gs
from syllabification.features import generate_features as gf
from syllabification.features import cut_features as cf

audio_files = ["../data/sample_1.wav", "../data/sample_2.wav"]
F, E = gf.generate_features(audio_files, window_step=0.01)
X, X_ts, fl = cf.cut_features(F, "C:/Users/Médéric Carriat/Desktop/wce/models/LSTM_params_BLSTM_fourlang_60_60_augmented_dropout_v2.h5.mat")
model = Syllabifier()
#syl.initialize_BLSTM_model(X.shape[1:])
model.load_model_from_file("../models/BLSTM_fourlang_60_60_augmented_dropout_v2.h5")
model.summary()
env = model.predict(X)
syl = gs.reconstruct_envelope(env, X_ts, fl, 300)

plt.plot(syl[0])