"""Speech Extractor

Module to extract segments of speech from the original wavs and gather the
results of the WCE on the segments that come from the same file.

The module contains the following functions:

    * extract_speech - extract speech for one file.
    * extract_speech_from_dir - extract speech for files in a directory.
    * retrieve_word_counts - write the gathered results per file to a .csv file.
"""

import os, sys, glob
import csv
import subprocess
import shutil
import numpy as np

def extract_speech(audio_file, rttm_file, chunks_dir):
    """
    Extract speech segments from an audio file.

    Parameters
    ----------
    audio_file : str
        Path to the audio file.
    rttm_file : str
        Path to the corresponding rttm file.
    chunks_dir : str
        Path to the directory where to store the resulting wav chunks.

    Returns
    -------
    wav_list : list
        List of the .wav files corresponding to the speech segments.
    onsets : 
        List of the onsets of the speech segments.
    offsets : 
        List of the offsets of the speech segments.
    """
    
    wav_list = []
    onsets = []
    offsets = []

    try:
        with open(rttm_file, 'r') as rttm:
            i = 0
            for line in rttm:
                # Replace tabulations by spaces
                fields = line.replace('\t', ' ')
                # Remove several successive spaces
                fields = ' '.join(fields.split())
                fields = fields.split(' ')
                onset, duration, activity = float(fields[3]), float(fields[4]), fields[7]
                if activity == 'speech':
                    basename = os.path.basename(audio_file).split('.wav')[0]
                    output = os.path.join(chunks_dir, '_'.join([basename, str(i)])+'.wav')
                    cmd = ['sox', audio_file, output,
                           'trim', str(onset), str(duration)]
                    subprocess.call(cmd)
                    wav_list.append(output)
                    onsets.append(onset)
                    offsets.append(onset+duration)
                    i += 1
    except IOError:
        shutil.rmtree(chunks_dir)
        sys.exit("Issue when extracting speech segments from wav.")

    onsets = np.array(onsets)
    offsets = np.array(offsets)

    return wav_list, onsets, offsets

def extract_speech_from_dir(audio_dir, rttm_dir, sad_name):
    """
    Extract speech for files in a directory.

    Parameters
    ----------
    audio_dir : str
        Path to the directory containing the audio files (.wav).
    rttm_dir : str
        Path to the directory containing the SAD files (.rttm).
    sad_name : str
        Name of the SAD algorithm used.

    Returns
    -------
    wav_list : list
        List containing the path to the wav segments resulting from the trim.
    """

    wav_list = []
    audio_files = glob.glob(audio_dir + "/*.wav")
    if not wav_files:
        sys.exit(("speech_extractor.py : No audio files found in {}".format(audio_dir)))

    chunks_dir = os.path.join(audio_dir, "wav_chunks_predict")
    if not os.path.exists(chunks_dir):
        os.mkdir(chunks_dir)
    else:
        shutil.rmtree(chunks_dir)
        os.mkdir(chunks_dir)

    for audio_file in audio_files:
        rttm_filename = "{}_{}.rttm".format(sad_name, os.path.basename(wav)[:-4])
        rttm_file = os.path.join(rttm_dir, sad_filename)
        if not os.path.isfile(rttm_file):
            sys.exit("The SAD file %s has not been found." % rttm_file)

        wav_list.append(extract_speech(wav, rttm_file, sad_name, chunks_dir)[0])

    wav_list = np.concatenate(wav_list)

    return wav_list


def retrieve_files_word_counts(word_counts, wav_chunks_list, output_path):
    """
    Retrieve the word count for each file from the wav chunks' word counts.

    Parameters
    ----------
    word_counts : list
        List of the word counts per wav chunk.
    wav_chunks_list : list
        List of paths to the wav chunks.
    output_path : str
        Path to the output_path file where to store the results.
    """

    files = []
    files_word_counts = []

    for f in wav_chunks_list:
        filepath = '_'.join(f.split('_')[:-1])
        filename = os.path.basename(filepath)
        if filename not in files:
            files.append(filename)

    for f in files:
        indices = [x for x, y in enumerate(wav_chunks_list) if f in y]
        wc = 0
        for i in indices:
            wc += word_counts[i]
        files_word_counts.append((f, wc))

    with open(output_path, 'w') as out:
        csvwriter = csv.writer(out, delimiter=';')
        for row in files_word_counts:
            csvwriter.writerow(row)
    print("Output saved at: {}.".format(output_path))

