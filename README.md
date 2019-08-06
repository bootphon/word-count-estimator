# Word Count Estimator

CLI tool for word count estimation in audio files.

## Installation

- Download the `wce` from gitlab and install the required Python packages using pip:

        $ git clone https://gitlab.coml.lscp.ens.fr/babycloud/ml/wce.git
        $ cd wce
        $ pip install requirements.txt

- Install the external dependencies SoX and libsndfile1. On Debian/Ubuntu run:

        $ sudo apt-get update && install sox libsndfile1

## Usage

It is possible to use directly the `wce` through the CLI or to run it within a
docker container.

### Command-line

For a complete list of available options, run:

    python cli.py -h

The tool disposes of two commands: train and predict.

- **Train:**  
    Trains a WCE model on audio files given their respective SAD files and annotation
    files.

      python cli.py train wav_dir sad_dir annotation_dir sad_name -w output_model_file

    If no model file is indicated, it will be saved to a default file: 
    adapted_model.pickle.

- **Predict:**  
    Predicts the word counts of audio files given their respective SAD files.

      python cli.py predict wav_dir sad_dir output_file sad_name -w model_file

    If no model file is indicated, the default model will be used: 
    default_model.pickle.

### Docker

Using the provided `Dockerfile`:

- Build the docker image:

        $ sudo docker build -t wce .

- Then to predict the word counts for some audio files using the pre-trained 
model, run a docker container and mount your data and result directories to the
intended directories in the container:

        docker run \
          --name wce \
          -v my_data/:/app/data \
          -v my_results/:/app/results \
          wce
    When the process is done, `output.csv` will be available in `my_results/`.

- For any other command, the arguments will need to be specified and the volumes
adapted. For instance to use the train command, a `models` volume will need to be
specified.


## Tests

Two tests are available:

- On the default model, checks if the results are still the same.
- On a trained model, checks if the RMSE is still under a certain threshold.

To run them:

- Install pytest:

      pip install pytest

- And run:

      pytest test/test.py


## Acknowledgments

This tool is a python version of the original [MATLAB WCE](https://github.com/aclew/WCE_VM)
based on:

Rasanen, O., Seshadri, S., Karadayi, J., Riebling, E., Bunce, J., Cristia, A.,
Metze, F., Casillas, M., Rosemberg, C., Bergelson, E., & Soderstrom, M. (submitted):
Automatic word count estimation from daylong child-centered recordings in
various language environments using language-independent syllabification of speech
