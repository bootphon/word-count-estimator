"""
CLI for the Word Count Estimation (WCE).

"""

import argparse
import numpy as np
import glob
import os
from dotenv import load_dotenv

from envelope_estimation import DataProcessing, EnvelopeEstimator
from word_count_estimation.annotations_processing import process_annotations
from word_count_estimation import WordCountEstimator

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

load_dotenv("./.env")

def train(args):
    
    env_model_name = os.path.basename(args.env_model_file)
    wce_model_name = os.path.basename(args.wce_model_file)
    print("Envelope estimation model used: {}".format(env_model_name))
    print("The resulting wce model will be saved to {}".format(wce_model_name))
   
    selcha_script = os.getenv("SELCHA_SCRIPT_PATH")
    print(selcha_script)
    tot_words, wav_list, alpha = process_annotations(args.audio_dir, 
                                                     args.annotations_dir,
                                                     args.rttm_dir,
                                                     args.SAD_name,
                                                     selcha_script)
    
    audio_files = wav_list
    train = tot_words

    dp = DataProcessing()
    X_train, timestamps, ori_frames_length = dp.generate_features_batch(audio_files)
    
    env_estimator = EnvelopeEstimator()
    env_estimator.load_model(args.env_model_file)
    envelopes_batch = env_estimator.predict(X_train)
    envelopes = dp.reconstruct_envelopes(envelopes_batch,
                                         timestamps,
                                         ori_frames_length)
    
    wce = WordCountEstimator()
    wce.train(envelopes, train, model_file=args.wce_model_file)
    
    
def predict(args):
    
    env_model_name = os.path.basename(args.env_model_file)
    wce_model_name = os.path.basename(args.wce_model_file)
    print("Envelope estimation model used: {}".format(env_model_name))
    print("WCE model used: {}".format(wce_model_name))
    
    audio_files = glob.glob(os.path.join(args.audio_dir, "*.wav"))
    
    dp = DataProcessing()
    X, timestamps, ori_frames_length = dp.generate_features_batch(audio_files)
    
    env_estimator = EnvelopeEstimator()
    env_estimator.load_model(args.env_model_file)
    envelopes_batch = env_estimator.predict(X)
    envelopes = dp.reconstruct_envelopes(envelopes_batch,
                                         timestamps,
                                         ori_frames_length)

    wce = WordCountEstimator()
    wce.load_model(args.wce_model_file)
    word_counts = wce.predict(envelopes)
    
    print(word_counts)


def main():
    
    env_path="../models/envelope_estimator/BLSTM_fourlang_60_60_augmented_dropout_v2.h5"
    default_wce_path="../models/word_count_estimator/default_model.pickle"
    adapted_wce_path="../models/word_count_estimator/adapted_model.pickle"
    
    parser = argparse.ArgumentParser(description="Word count estimation model.")
    
    subparsers = parser.add_subparsers(help='desired mode')
    
    parser_train = subparsers.add_parser('train', help='train mode')
    parser_train.add_argument('audio_dir',
                              help='directory where the audio files are stored')
    parser_train.add_argument('annotations_dir',
                              help='directory where the annotation files are stored')
    parser_train.add_argument('rttm_dir',
                              help='directory where the SAD .rttm files are stored')
    parser_train.add_argument('SAD_name', help='name of the SAD used')
    parser_train.add_argument('-e', '--env_model_file',
                              help='path to the syllable envelope estimator model file',
                              default=env_path)
    parser_train.add_argument('-w', '--wce_model_file',
                              help='path to the word count estimator model file',
                              default=adapted_wce_path)
    parser_train.set_defaults(func=train)
    
    parser_predict = subparsers.add_parser('predict', help='predict mode')
    parser_predict.add_argument('audio_dir',
                                help='directory where the audio files are stored')
    parser_predict.add_argument('rttm_dir',
                                help='directory where the SAD .rttm files are stored')
    parser_predict.add_argument('SAD_name', help='name of the SAD used')
    parser_predict.add_argument('-e', '--env_model_file',
                                help='path to the syllable envelope estimator model file',
                                default=env_path)
    parser_predict.add_argument('-w', '--wce_model_file',
                                help='path to the word count estimator model file',
                                default=default_wce_path)
    parser_predict.set_defaults(func=predict)
    
    args = parser.parse_args()
    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments")
    func(args)



if __name__ == '__main__':
    main()
