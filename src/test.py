import matplotlib.pyplot as plt

from envelope_estimation.models.envelope_estimator import EnvelopeEstimator
from envelope_estimation.models import reconstruct_envelope as re
from envelope_estimation.features import generate_features as gf
from envelope_estimation.features import cut_features as cf
from syllabification import get_syllables as gs


audio_files = ["../data/sample_1.wav", "../data/sample_2.wav"]
F, E = gf.generate_features(audio_files, window_step=0.01)
X, X_ts, fl = cf.cut_features(F, "C:/Users/Médéric Carriat/Desktop/wce/models/LSTM_params_BLSTM_fourlang_60_60_augmented_dropout_v2.h5.mat")
model = EnvelopeEstimator()
#syl.initialize_BLSTM_model(X.shape[1:])
model.load_model_from_file("../models/BLSTM_fourlang_60_60_augmented_dropout_v2.h5")
model.summary()
env_windows = model.predict(X)
env = re.reconstruct_envelope(env_windows, X_ts, fl, 300)

plt.plot(env[0])
plt.plot(env[1])

syl = gs.count_syllables(env, 0.5)