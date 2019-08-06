"""
CLI for the Word Count Estimation (WCE).

usage: cli.py [-h] {train,predict} ...

positional arguments:
  {train,predict}  desired mode
    train          train mode
    predict        predict mode

optional arguments:
  -h, --help       show this help message and exit
"""

import argparse
import os
import csv
import shutil
import sys
from dotenv import load_dotenv
from tensorflow.python.util import deprecation

# To disable tensorflow deprecation warnings.
import logging
logging.getLogger('tensorflow').disabled = True
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# To not use GPU for envelope estimation.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from wce.envelope_estimation.data_processing import DataProcessing
from wce.envelope_estimation.envelope_estimator import EnvelopeEstimator
from wce.word_count_estimation.annotations_processing import process_annotations
from wce.word_count_estimation.speech_extractor import extract_speech, retrieve_files_word_counts
from wce.word_count_estimation.word_count_estimator import WordCountEstimator


load_dotenv("./.env")

def train(args):
    """
    usage: cli.py train [-h] [-e ENV_MODEL_FILE] [-w WCE_MODEL_FILE] [-r REF_PATH]
                    audio_dir annotations_dir rttm_dir SAD_name

    positional arguments:
      audio_dir             directory where the audio files are stored
      annotations_dir       directory where the annotation files are stored
      rttm_dir              directory where the SAD .rttm files are stored
      SAD_name              name of the SAD used

    optional arguments:
      -h, --help            show this help message and exit
      -e ENV_MODEL_FILE, --env_model_path ENV_MODEL_FILE
                            path to the syllable envelope estimator model file
      -w WCE_MODEL_FILE, --wce_model_path WCE_MODEL_FILE
                            path to the word count estimator model file
      -r REF_PATH, --ref_path REF_PATH
                            path to the output reference file containing the word
                            counts of each audio file
    """

    if not os.path.exists(args.audio_dir):
        raise IOError("Audio directory does not exist.")
    if not os.path.exists(args.annotations_dir):
        raise IOError("Annotation directory does not exist.")
    if not os.path.exists(args.rttm_dir):
        raise IOError("SAD directory does not exist.")
    if not os.path.exists(args.env_model_path):
        raise IOError("Envelope estimator model file does not exist.")

    env_model_name = os.path.basename(args.env_model_path)
    wce_model_name = os.path.basename(args.wce_model_path)
    print("Envelope estimation model used: {}".format(env_model_name))
    print("The resulting wce model will be saved to {}".format(wce_model_name))

    selcha_script = os.getenv("SELCHA_SCRIPT_PATH")
    tot_files_words, tot_seg_words, wav_list, alpha = \
            process_annotations(args.audio_dir, args.annotations_dir,
                                args.rttm_dir, args.SAD_name, selcha_script)

    audio_files = wav_list
    target_counts = tot_seg_words
    dp = DataProcessing()
    feature_batch, batch_timestamps, files_length = dp.generate_features_batch(audio_files)

    env_estimator = EnvelopeEstimator()
    env_estimator.load_model(args.env_model_path)
    envelope_batch = env_estimator.predict(feature_batch)
    envelopes = dp.reconstruct_envelopes(envelope_batch,
                                         batch_timestamps,
                                         files_length)

    wce = WordCountEstimator()
    wce.alpha = alpha
    wce.train(envelopes, target_counts, model_file=args.wce_model_path)

    if args.ref_path:
        if not os.path.exists(os.path.dirname(args.ref_path)):
            raise IOError("Output directory does not exist.")
        with open(args.ref_path, 'w') as ref:
            csvwriter = csv.writer(ref, delimiter=';')
            for row in tot_files_words:
                csvwriter.writerow(row)

    chunks_dir = os.path.dirname(audio_files[0])
    shutil.rmtree(chunks_dir)

    wce.summary()


def predict(args):
    """
    usage: cli.py predict [-h] [-e ENV_MODEL_FILE] [-w WCE_MODEL_FILE]
                      audio_dir rttm_dir SAD_name output

    positional arguments:
      audio_dir             directory where the audio files are stored
      rttm_dir              directory where the SAD .rttm files are stored
      SAD_name              name of the SAD used
      output                path to the word count output .csv file

    optional arguments:
      -h, --help            show this help message and exit
      -e ENV_MODEL_FILE, --env_model_path ENV_MODEL_FILE
                            path to the syllable envelope estimator model file
      -w WCE_MODEL_FILE, --wce_model_path WCE_MODEL_FILE
                            path to the word count estimator model file
    """

    if not os.path.exists(args.audio_dir):
        raise IOError("Audio directory does not exist.")
    if not os.path.exists(args.rttm_dir):
        raise IOError("SAD directory does not exist.")
    if not os.path.exists(os.path.dirname(args.output)):
        raise IOError("Output directory does not exist.")
    if not os.path.exists(args.env_model_path):
        raise IOError("Envelope estimator model file does not exist.")
    if not os.path.exists(args.wce_model_path):
        raise IOError("Envelope estimator model file does not exist.")

    env_model_name = os.path.basename(args.env_model_path)
    wce_model_name = os.path.basename(args.wce_model_path)
    print("Envelope estimation model used: {}".format(env_model_name))
    print("WCE model used: {}".format(wce_model_name))

    audio_files = extract_speech(args.audio_dir, args.rttm_dir, args.SAD_name)
    dp = DataProcessing()
    feature_batch, timestamps, files_length = dp.generate_features_batch(audio_files)

    env_estimator = EnvelopeEstimator()
    env_estimator.load_model(args.env_model_path)
    envelopes_batch = env_estimator.predict(feature_batch)
    envelopes = dp.reconstruct_envelopes(envelopes_batch,
                                         timestamps,
                                         files_length)

    wce = WordCountEstimator()
    wce.load_model(args.wce_model_path)
    word_counts = wce.predict(envelopes)

    retrieve_files_word_counts(word_counts, audio_files, args.output)

    chunks_dir = os.path.dirname(audio_files[0])
    shutil.rmtree(chunks_dir)


def main():
    """
    Main function in charge of parsing the command.
    """

    env_path = os.getenv("DEFAULT_ENV_EST")
    default_wce_path = os.getenv("DEFAULT_WCE")
    adapted_wce_path = os.getenv("ADAPTED_WCE")

    parser = argparse.ArgumentParser(description="Word count estimation model.")

    subparsers = parser.add_subparsers(help='desired mode')

    parser_train = subparsers.add_parser('train', help='train mode')
    parser_train.add_argument('audio_dir',
                              help='directory where the audio files are stored')
    parser_train.add_argument('rttm_dir',
                              help='directory where the SAD .rttm files are stored')
    parser_train.add_argument('annotations_dir',
                              help='directory where the annotation files are stored')
    parser_train.add_argument('SAD_name', help='name of the SAD used')
    parser_train.add_argument('-e', '--env_model_path',
                              help='path to the syllable envelope estimator model file',
                              default=env_path)
    parser_train.add_argument('-w', '--wce_model_path',
                              help='path to the word count estimator model file',
                              default=adapted_wce_path)
    parser_train.add_argument('-r', '--ref_path',
                              help='path to the output reference file containing\
                              the word counts of each audio file')
    parser_train.set_defaults(func=train)

    parser_predict = subparsers.add_parser('predict', help='predict mode')
    parser_predict.add_argument('audio_dir',
                                help='directory where the audio files are stored')
    parser_predict.add_argument('rttm_dir',
                                help='directory where the SAD .rttm files are stored')
    parser_predict.add_argument('output', help='path to the word count output .csv file')
    parser_predict.add_argument('SAD_name', help='name of the SAD used')
    parser_predict.add_argument('-e', '--env_model_path',
                                help='path to the syllable envelope estimator model file',
                                default=env_path)
    parser_predict.add_argument('-w', '--wce_model_path',
                                help='path to the word count estimator model file',
                                default=default_wce_path)
    parser_predict.set_defaults(func=predict)

    args = parser.parse_args()
    try:
        func = args.func
    except:
        parser.print_help()
        sys.exit(0)
    func(args)

if __name__ == '__main__':
    main()

