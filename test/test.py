import os
from math import sqrt
import pytest
import numpy as np
from dotenv import load_dotenv


from wce.envelope_estimation.data_processing import DataProcessing
from wce.envelope_estimation.envelope_estimator import EnvelopeEstimator
from wce.word_count_estimation.annotations_processing import process_annotations
from wce.word_count_estimation.speech_extractor import extract_speech, retrieve_files_word_counts
from wce.word_count_estimation.word_count_estimator import WordCountEstimator


load_dotenv("../.env")
env_path = os.getenv("DEFAULT_ENV_EST")
default_wce_path = os.getenv("DEFAULT_WCE")
adapted_wce_path = os.getenv("ADAPTED_WCE")


@pytest.fixture
def get_envelopes():

    with open("data/test_data.txt") as f:
        wav_list = f.read().splitlines()
    target_counts = [8, 5, 7, 1, 4, 6, 6 ,6, 5, 6]

    dp = DataProcessing()
    feature_batch, batch_timestamps, files_length = \
                    dp.generate_features_batch(wav_list)

    env_estimator = EnvelopeEstimator()
    env_estimator.load_model(env_path)
    envelope_batch = env_estimator.predict(feature_batch)
    envelopes = dp.reconstruct_envelopes(envelope_batch,
                                         batch_timestamps,
                                         files_length)
    return envelopes, target_counts


def test_untrained_predict(get_envelopes):

    wce = WordCountEstimator()
    wce.load_model(default_wce_path)
    envelopes = get_envelopes[0]
    word_counts = np.array(wce.predict(envelopes))
    
    assert ([10., 7., 9., 1., 4., 8., 7., 6., 5., 9.] == word_counts).all()


def test_train_and_predict(get_envelopes):

    envelopes, target_counts = get_envelopes
    target_counts = np.array(target_counts)
    
    wce = WordCountEstimator()
    wce.train(envelopes, target_counts, model_file=adapted_wce_path)
    wce.load_model(adapted_wce_path)
    word_counts = wce.predict(envelopes)

    a = word_counts[np.where(target_counts > 0)]
    b = target_counts[np.where(target_counts > 0)]
    RMSE = sqrt(np.square(np.mean(((a-b) / b))))*100
    
    assert RMSE < 0.6
