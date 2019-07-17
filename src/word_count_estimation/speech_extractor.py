import os
import sys
import subprocess
import glob
import shutil
import csv


def extract_speech(audio_dir, rttm_dir, sad_name):
    """
    Read a .rttm file and extracts segments of the wav files that
    have been annotated as being speech.

    :param wav: Path to the audio file (.wav).
    :param sad: Path to the rttm file (.rttm).
    :param out: Folder where the segments need to be stored.
    """

    wav_list = []
    wav_files = glob.glob(audio_dir + "/*.wav")

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
            print("The SAD file %s has not been found." % sad)
            sys.exit(1)

        with open(sad, 'r') as rttm:
            i = 0
            for line in rttm:
                # Replace tabulations by spaces
                fields = line.replace('\t', ' ')
                # Remove several successive spaces
                fields = ' '.join(fields.split())
                fields = fields.split(' ')
                onset, duration, activity = float(fields[3]), float(fields[4]), fields[7]
                offset = onset+duration
                if activity == 'speech':
                    basename = os.path.basename(wav).split('.wav')[0]
                    output = os.path.join(out_dir, '_'.join([basename, str(i)])+'.wav')
                    cmd = ['sox', wav, output,
                           'trim', str(onset), str(duration)]
                    subprocess.call(cmd)
                    wav_list.append(output)
                    i += 1

    return wav_list


def retrieve_files_word_counts(word_counts, wav_list, output):
    
    files = []
    files_word_counts = []
    
    for f in wav_list:
        filepath = '_'.join(f.split('_')[:-1])
        filename = os.path.basename(filepath)
        if filename not in files:
            files.append(filename)

    for f in files:
        indices = [x for x,y in enumerate(wav_list) if f in y]
        wc = 0
        for i in indices:
            wc += word_counts[i]
        files_word_counts.append((f, wc))

    with open(output, 'w') as out:
        csvwriter = csv.writer(out, delimiter=';')
        for row in files_word_counts:
            csvwriter.writerow(row)


