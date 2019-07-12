"""
CLI for the Word Count Estimation (WCE).

"""

import argparse
import numpy as np
import glob
import os

from envelope_estimation import DataProcessing, EnvelopeEstimator
from word_count_estimation import WordCountEstimator

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def train(args):
    
    env_model_name = os.path.basename(args.env_model_file)
    wce_model_name = os.path.basename(args.wce_model_file)
    print("Envelope estimation model used: {}".format(env_model_name))
    print("The resulting wce model will be saved to {}".format(wce_model_name))
    
    audio_files = glob.glob(os.path.join(args.audio_dir, "*.wav"))
    annotation_files = glob.glob(os.path.join(args.annotations_dir, "*.eaf"))
    audio_files.sort(key= lambda x: int(os.path.basename(x)[:-4]))
    
    dp = DataProcessing()
    X_train, timestamps, ori_frames_length = dp.generate_features_batch(audio_files)
    
    env_estimator = EnvelopeEstimator()
    env_estimator.load_model(args.env_model_file)
    envelopes_batch = env_estimator.predict(X_train)
    envelopes = dp.reconstruct_envelopes(envelopes_batch,
                                         timestamps,
                                         ori_frames_length)
    
    wce = WordCountEstimator()
    train = np.array([84, 138, 91, 23, 217, 116, 96, 148, 47, 105, 141, 549, 224, 352, 258])
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
    parser_predict.add_argument('-e', '--env_model_file',
                                help='path to the syllable envelope estimator model file',
                                default=env_path)
    parser_predict.add_argument('-w', '--wce_model_file',
                                help='path to the word count estimator model file',
                                default=default_wce_path)
    parser_predict.set_defaults(func=predict)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
