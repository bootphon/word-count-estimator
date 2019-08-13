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

import os
import argparse
import logging
from dotenv import load_dotenv
from wce import pipeline


# To disable tensorflow deprecation warnings.
logging.getLogger('tensorflow').disabled = True
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# To not use GPU for envelope estimation.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


load_dotenv(".env")


def train(args):
    """
    TODO
    """

    selcha_script = os.getenv("SELCHA_SCRIPT")

    pipeline.train(audio_dir=args.audio_dir, annotations_dir=args.annotations_dir,
                   rttm_dir=args.rttm_dir, sad_name=args.sad_name,
                   data_processing_config=args.data_processing_config,
                   env_model_path=args.env_model_path, wce_model_path=args.wce_model_path,
                   selcha_script=selcha_script)


def predict(args):
    """
    TODO
    """

    pipeline.predict(audio_dir=args.audio_dir, results_path=args.results_path,
                     data_processing_config=args.data_processing_config,
                     env_model_path=args.env_model_path, wce_model_path=args.wce_model_path,
                     rttm_dir=args.rttm_dir, sad_name=args.sad_name)


def main():
    """
    Main function in charge of parsing the command.
    """

    env_path = os.getenv("DEFAULT_ENV")
    default_wce_path = os.getenv("DEFAULT_WCE")
    trained_wce_path = os.getenv("TRAINED_WCE")
    data_processing_path = os.getenv("DP_CONFIG")

    parser = argparse.ArgumentParser(description="Word count estimation model.")

    subparsers = parser.add_subparsers(help='desired mode')

    # Parser for the train mode
    parser_train = subparsers.add_parser('train', help='train mode')
    parser_train.add_argument('audio_dir',
                              help='directory where the audio files are stored')
    parser_train.add_argument('annotations_dir',
                              help='directory where the annotation files are stored')
    parser_train.add_argument('rttm_dir',
                              help='directory where the SAD .rttm files are stored')
    parser_train.add_argument('-s', '--sad_name', help='name of the SAD used')
    parser_train.add_argument('-d', '--data_processing_config',
                              help='path to the data processing config file',
                              default=data_processing_path)
    parser_train.add_argument('-e', '--env_model_path',
                              help='path to the syllable envelope estimator model file',
                              default=env_path)
    parser_train.add_argument('-w', '--wce_model_path',
                              help='path to the word count estimator model file',
                              default=trained_wce_path)
    parser_train.set_defaults(func=train)

    # Parser for the predict mode
    parser_predict = subparsers.add_parser('predict', help='predict mode')
    parser_predict.add_argument('audio_dir',
                                help='directory where the audio files are stored')
    parser_predict.add_argument('results_path',
                                help='path to the word count output .csv file')
    parser_predict.add_argument('-r', '--rttm_dir',
                                help='directory where the SAD .rttm files are stored')
    parser_predict.add_argument('-s', '--sad_name', help='name of the SAD used')
    parser_predict.add_argument('-d', '--data_processing_config',
                                help='path to the data processing config file',
                                default=data_processing_path)
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

