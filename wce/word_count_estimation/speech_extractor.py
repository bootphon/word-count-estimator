"""Speech Extractor

Module to extract segments of speech from the original wavs and gather the
results of the WCE on the segments that come from the same file.

The module contains the following functions:

    * extract_speech - returns the list of paths to the wavs segments.
    * retrieve_word_counts - writes the gathered results per file to a .csv file.
"""

import os
import sys
import subprocess
import glob
import shutil
import csv


def extract_speech(audio_dir, rttm_dir, sad_name):
    """
    Read audio files and their corresponding .rttm files and extracts segments
    of the wav files that have been annotated as being speech.

    Paremeters
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
    out_dir : str
        Directory for the extracted .wav segments.
    """

    wav_list = []
    wav_files = glob.glob(audio_dir + "/*.wav")
    if not wav_files:
        sys.exit(("speech_extractor.py : No audio files found in {}".format(audio_dir)))

    out_dir = os.path.join(audio_dir, "wav_chunks_predict")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)

    for wav in wav_files:

        sad_filename = "{}_{}.rttm".format(sad_name, os.path.basename(wav)[:-4])
        sad = os.path.join(rttm_dir, sad_filename)
        if not os.path.isfile(sad):
            sys.exit("The SAD file %s has not been found." % sad)

        try:
            with open(sad, 'r') as rttm:
                i = 0
                for line in rttm:
                    # Replace tabulations by spaces
                    fields = line.replace('\t', ' ')
                    # Remove several successive spaces
                    fields = ' '.join(fields.split())
                    fields = fields.split(' ')
                    onset, duration, activity = float(fields[3]), float(fields[4]), fields[7]
                    if activity == 'speech':
                        basename = os.path.basename(wav).split('.wav')[0]
                        output = os.path.join(out_dir, '_'.join([basename, str(i)])+'.wav')
                        cmd = ['sox', wav, output,
                               'trim', str(onset), str(duration)]
                        subprocess.call(cmd)
                        wav_list.append(output)
                        i += 1
        except:
            shutil.rmtree(chunks_dir)

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

