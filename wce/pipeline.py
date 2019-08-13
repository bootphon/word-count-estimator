import os, sys, glob
import csv
import shutil
from tqdm import tqdm

from .envelope_estimation.feature_extraction import FeatureExtraction
from .envelope_estimation.batch import Batch
from .envelope_estimation.envelope_estimator import EnvelopeEstimator
from .word_count_estimation.annotations_processing import process_annotations
from .word_count_estimation.speech_extractor import extract_speech, \
                                                    retrieve_files_word_counts
from .word_count_estimation.word_count_estimator import WordCountEstimator


def train(audio_dir, annotations_dir, rttm_dir, sad_name, data_processing_config,
          env_model_path, wce_model_path, selcha_script):
    """
    TODO
    """

    if not os.path.exists(audio_dir):
        raise IOError("Audio directory does not exist.")
    if not os.path.exists(annotations_dir):
        raise IOError("Annotation directory does not exist.")
    if not os.path.exists(rttm_dir):
        raise IOError("SAD directory does not exist.")
    if not os.path.exists(data_processing_config):
        raise IOError("Data processing config file does not exist.")
    if not os.path.exists(env_model_path):
        raise IOError("Envelope estimator model file does not exist.")
    if not os.path.exists(selcha_script):
        raise IOError("Selcha script does not exist.")

    feature_extractor = FeatureExtraction()
    feature_extractor.load(data_processing_config)
    batchifier = Batch()
    batchifier.load(data_processing_config)
    env_model = EnvelopeEstimator()
    env_model.load_model(env_model_path)
    wce_model = WordCountEstimator()

    print("Processing annotations.")
    tot_files_words, tot_seg_words, wav_list, alpha = \
            process_annotations(audio_dir, annotations_dir, rttm_dir,
                                sad_name, selcha_script)

    wce_model.alpha = alpha
    audio_files = wav_list
    target_counts = tot_seg_words

    print("Extracting features from data.")
    feature_list = []

    pbar = tqdm(total=len(audio_files))
    for audio_file in audio_files:
        feature_mat = feature_extractor.generate_features(audio_file)
        feature_list.append(feature_mat)
        pbar.update(1)
    pbar.close()
    
    feature_batch, timestamps, wav_lengths = batchifier.generate_batch(
                                                                feature_list)

    print("Predicting syllable envelopes.")
    envelope_batch = env_model.predict(feature_batch)

    envelopes = batchifier.reconstruct_envelope(envelope_batch, timestamps,
                                                wav_lengths)

    print("Training WCE model.")
    wce_model.train(envelopes, target_counts, model_file=wce_model_path)

    chunks_dir = os.path.dirname(audio_files[0])
    shutil.rmtree(chunks_dir)

    print("Training ended successfully.")
    print("WCE model saved at: {}".format(wce_model_path))
    wce_model.summary()


def predict(audio_dir, results_path, data_processing_config,
            env_model_path, wce_model_path, rttm_dir=None, sad_name=None):
    """
    TODO
    """

    if not os.path.exists(audio_dir):
        raise IOError("Audio directory does not exist.")
    if not os.path.exists(data_processing_config):
        raise IOError("Data processing config file does not exist.")
    if not os.path.exists(env_model_path):
        raise IOError("Envelope estimator model file does not exist.")
    if not os.path.exists(wce_model_path):
        raise IOError("Envelope estimator model file does not exist.")

    feature_extractor = FeatureExtraction()
    feature_extractor.load(data_processing_config)
    batchifier = Batch()
    batchifier.load(data_processing_config)
    env_model = EnvelopeEstimator()
    env_model.load_model(env_model_path)
    wce_model = WordCountEstimator()
    wce_model.load_model(wce_model_path)

    if rttm_dir and sad_name:
        if not os.path.exists(rttm_dir):
            raise IOError("SAD directory does not exist.")
        print("Extracting speech segments from audio files.")
        audio_files = extract_speech(audio_dir, rttm_dir, sad_name)
    elif rttm_dir:
        sys.exit("SAD name missing.")
    elif sad_name:
        sys.exit("rttm directory missing.")
    else:
        audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))

    print("Extracting features from data.")
    feature_list = []

    pbar = tqdm(total=len(audio_files))
    for audio_file in audio_files:
        feature_mat = feature_extractor.generate_features(audio_file)
        feature_list.append(feature_mat)
        pbar.update(1)
    pbar.close()
    
    feature_batch, timestamps, wav_lengths = batchifier.generate_batch(
                                                                feature_list)

    print("Predicting syllable envelopes.")
    envelope_batch = env_model.predict(feature_batch)

    envelopes = batchifier.reconstruct_envelope(envelope_batch, timestamps,
                                                wav_lengths)

    print("Predicting word counts.")
    wce_model = WordCountEstimator()
    wce_model.load_model(wce_model_path)
    word_counts = wce_model.predict(envelopes)

    if rttm_dir:
        retrieve_files_word_counts(word_counts, audio_files, results_path)
        chunks_dir = os.path.dirname(audio_files[0])
        shutil.rmtree(chunks_dir)
    else:
        with open(results_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';')
            for i in range(len(audio_files)):
                csv_writer.writerow((audio_files[i], word_counts[i]))

    print("Prediction ended successfully.")
    print("Models used: ")
    print(" - Envelope estimator: {}".format(os.path.basename(env_model_path)))
    print(" - Wordcount estimator: {}".format(os.path.basename(wce_model_path)))
    print("Results saved at: {}".format(results_path))

