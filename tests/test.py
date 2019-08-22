import os
from math import sqrt
import pytest
import numpy as np
from dotenv import load_dotenv

from wce.data_preprocessing.feature_extraction import FeatureExtraction
from wce.data_preprocessing.batch import Batch
from wce.envelope_estimation.envelope_estimator import EnvelopeEstimator
from wce.word_count_estimation.word_count_estimator import WordCountEstimator


load_dotenv("./.env")
env_path = os.getenv("DEFAULT_ENV")
default_wce_path = os.getenv("DEFAULT_WCE")
test_wce_path = os.getenv("TEST_WCE")


@pytest.fixture
def get_envelopes():
    feature_extractor = FeatureExtraction(extractor="librosa")
    batchifier = Batch()
    env_model = EnvelopeEstimator()
    env_model.load_model(env_path)

    target_counts = np.array([8, 5, 7, 1, 4, 6, 6 ,6, 5, 6])
    with open("data/test_data.txt") as f:
        audio_files = f.read().splitlines()

    feature_list = []
    for audio_file in audio_files:
        feature_mat = feature_extractor.generate_features(audio_file)
        feature_list.append(feature_mat)

    feature_batch, timestamps, wav_lengths = batchifier.generate_batch(
                                                                feature_list)

    envelope_batch = env_model.predict(feature_batch)

    envelopes = batchifier.reconstruct_envelope(envelope_batch, timestamps,
                                                wav_lengths)

    return envelopes, target_counts


def test_untrained_predict(get_envelopes):

    wce = WordCountEstimator()
    wce.summary()
    wce.load_model(default_wce_path)
    X, target_counts = get_envelopes
    y = np.array(wce.predict(X))

    a = y[np.where(target_counts > 0)]
    b = target_counts[np.where(target_counts > 0)]
    RMSE = sqrt(np.square(np.mean(((a-b) / b))))*100
    print("RMSE :", RMSE)
    print("y", y)

    assert RMSE < 10


def test_train_and_predict(get_envelopes):

    envelopes, target_counts = get_envelopes
    X_train, X_test = envelopes[0:8], envelopes[8:10]
    y_train, y_test = target_counts[0:8], target_counts[8:10]
    
    wce = WordCountEstimator()
    wce.train(X_train, y_train, model_file=test_wce_path)
    wce.load_model(test_wce_path)
    y_pred = wce.predict(X_test)

    a = y_pred[np.where(y_test > 0)]
    b = y_test[np.where(y_test > 0)]
    RMSE = sqrt(np.square(np.mean(((a-b) / b))))*100
    print("RMSE :", RMSE)
    print("y_pred", y_pred)
    print("y_test", y_test)
    
    assert RMSE < 20

