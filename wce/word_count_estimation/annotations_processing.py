"""Annotations Processing

Module to process the annotations files (.eaf).

It contains:

    * eaf2txt - returns a .txt containing timestamps and transcriptions from an
    annotation file.
    * enrich_txt - enriches the .txt file by adding information at each line.
    * count_annotations_words - counts the number of "seen" words, and real number
    of words in each audio file with regard to the SAD and the annotations.
    * process_annotations - main function using all above functions.
    * save_reference - additional function to save the results to a reference
    file for future comparison.
    (see more info in each function's docstring)
"""

import csv, sys, os
import shutil
import glob
import subprocess
import numpy as np
import pympi as pmp


def eaf2txt(eaf_file, output_folder):
    """
    Convert an eaf file to the txt format by extracting the onset, offset, ortho,
    and the speaker tier.
    Only the speaker tiers MOT, FAT and CHI are selected. Any other tier is
    ignored.

    Parameters
    ----------
    eaf_file : str
        Path to the eaf file.
    output_folder : str
        Path to the output directory.
    """

    basename = os.path.splitext(os.path.basename(eaf_file))[0]
    output_path = os.path.join(output_folder, basename + '.txt')

    with open(output_path, 'w') as output_file:

        EAF = pmp.Elan.Eaf(eaf_file)
        tiers = EAF.tiers
        for tier in tiers:

            annotations = []
            if ("CHI" in tier or "FAT" in tier or "MOT" in tier) and "@" not in tier:
                annotations = EAF.get_annotation_data_for_tier(tier)

            for annotation in annotations:
                parameters = EAF.get_parameters_for_tier(tier)
                onset, offset, transcript = annotation[0], annotation[1], annotation[2]

                speaker = tier
                if len(speaker) > 4:
                    speaker = speaker[4:7]
                else:
                    speaker = speaker[0:3]

                output_str = "%d\t%d\t%s\t%s\t%s\n" % (onset, offset, '',
                                                       transcript, speaker)
                output_file.write(output_str)

    return output_path


def enrich_txt(txt_file, script_path):
    """"
    Enriches the .txt file by syllabifying the transcriptions and counting the
    number of syllables and words.

    Parameters
    ----------
    txt_file : str
        Path to the .txt file to be enriched.
    script_path : str
        Path to script in charge of enriching the file.

    Returns
    -------
    enrich_file : str
        Path to the enriched file.
    """

    cmd = "{} {} {}_enriched.txt spanish".format(script_path, txt_file,
                                                 txt_file[:-4])
    subprocess.call(cmd, shell=True)
    os.remove(txt_file)
    enrich_file = "{}_enriched.txt".format(txt_file[:-4])

    return enrich_file


def count_annotations_words(enrich_file, rttm_file, audio_file, chunks_dir):
    """
    This function takes an enriched.txt file and .rttm file and measures the
    number of words and syllables in each of the SAD segments defined in the
    rttm file.
    It also extracts the .wav files corresponding to these SAD segments from the
    original audio file.

    This function is a slightly changed version of the one in the WCE_VM repo 
    at /aux_VM/combine_rttm_and_enrich.py. It has a second version which can 
    be found in the code.

    Parameters
    ----------
    enrich_file : str
        Path to the enriched file (.txt).
    rttm_file : str
        Path to the SAD file (.rttm).
    audio_file : str
        Path to the audio file (.wav).
    chunks_dir : str
        Path to the directory where to store the SAD wav chunks.

    Returns
    -------
    tot_words : float
        Real number of words in the audio file (derived from annotations).
    tot_syls : float
        Real number of syllable in the audio file (derived from annotations).
    SAD_segments_words : ndarray
        1D array containing the number of words in each SAD segment.
    SAD_segments_syllables : ndarray
        1D array containing the number of syllables in each SAD segment.
    wavList : list
        List of the .wav files corresponding to the SAD segments (same order
        as the 2 previous arrays).
    """

    curr_file = audio_file[:-4]
    filename = os.path.basename(curr_file)

    # Read enriched file first to get original annotated utterances and their
    # information
    onset = np.array([])
    offset = np.array([])
    sylcount = np.array([])
    wordcount = np.array([])

    speaker = np.array([])
    ortho = np.array([])
    syllables = np.array([])

    with open(enrich_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            onset = np.append(onset, float(row[0]))
            offset = np.append(offset, float(row[1]))
            speaker = np.append(speaker, row[3])
            ortho = np.append(ortho, row[4])
            wordcount = np.append(wordcount, float(row[5]))
            syllables = np.append(syllables, row[6])
            sylcount = np.append(sylcount, float(row[7]))

    order = np.argsort(onset)
    onset = onset[order]
    offset = offset[order]
    wordcount = wordcount[order]
    sylcount = sylcount[order]
    ortho = ortho[order]
    syllables = syllables[order]
    speaker = speaker[order]

    onset = onset/1000
    offset = offset/1000

    tot_words = sum(wordcount)
    tot_syls = sum(sylcount)

    # Read also the .rttm file to get SAD segment onsets and offsets
    SAD_onsets = np.array([])
    SAD_offsets = np.array([])

    wavList = []

    try:
        with open(rttm_file, 'rt') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                SAD_onset = float(row[3])
                SAD_offset = float(row[4])
                SAD_onsets = np.append(SAD_onsets, SAD_onset)
                SAD_offsets = np.append(SAD_offsets, SAD_onset+SAD_offset)
                wav_chunk_path = "{}/{}_{}.wav".format(chunks_dir, filename,
                                                       str(int(SAD_onset)*1000).zfill(8))
                cmd = ['sox', audio_file, wav_chunk_path,
                       'trim', str(SAD_onset), str(SAD_offset)]
                subprocess.call(cmd)
                wavList.append(wav_chunk_path)
    except:
        shutil.rmtree(chunks_dir)

    # Count only words that come from segments of the SAD that fully overlap
    # with segments of the reference.
    SAD_segments_words = np.array([])
    SAD_segments_syllables = np.array([])

    for sadseg in range(0, len(SAD_onsets)):
        SAD_segments_words = np.append(SAD_segments_words,0)
        SAD_segments_syllables = np.append(SAD_segments_syllables,0)
        SAD_onset = SAD_onsets[sadseg]
        SAD_offset = SAD_offsets[sadseg]

        # These segments must be fully within the SAD segment
        # Segments that end after SAD onset
        tmp1 = offset-SAD_onset > 0
        # Segments that begin before SAD offset
        tmp2 = onset-SAD_offset < 0
        # Valid segments are the ones with overlap
        tmp3 = (tmp1 & tmp2)

        # Which original segments overlap with the target segment
        if sum(tmp3) > 0:
            i = np.where(tmp3 > 0)
            segs_to_consider = range(max(0,i[0][0]-1), min(len(tmp1),i[0][-1]+2))

            # Calculate exact overlapping
            for j in range(0, len(segs_to_consider)):

                # For words
                total_words_in_real = wordcount[segs_to_consider[j]]
                total_syls_in_real = sylcount[segs_to_consider[j]]
                words_real = str.split(ortho[segs_to_consider[j]])
                y = 0
                wordlength = np.empty([len(words_real)])
                for ww in words_real:
                    wordlength[y] = len(ww)
                    y = y+1
                total_wordlength = sum(wordlength)

                # Where real segment starts and ends
                t1 = onset[segs_to_consider[j]]
                t2 = offset[segs_to_consider[j]]
                dur_real = t2-t1
                uniform_wordlengths = wordlength/total_wordlength*dur_real
                startpos = 0
                for jj in range(0, len(words_real)):
                    i1 = range(int((t1+startpos)*100),
                               int((t1+startpos+uniform_wordlengths[jj])*100))
                    i2 = range(int(SAD_onset*100), int(SAD_offset*100))
                    # Number of overlapping elements
                    olap = len(list(set(i1) & set(i2)))
                    # Proportion of utterance overlapping with SAD segment
                    coverage = (olap/100.0)/dur_real
                    if(coverage > 0):
                        SAD_segments_words[sadseg] += 1
                    startpos = startpos+uniform_wordlengths[jj]

                # Same for syllables
                total_syllables_in_real = sylcount[segs_to_consider[j]]
                syllables_real = np.empty([])
                syllables_tmp = str.split(syllables[segs_to_consider[j]])
                for ss in range (0,len(syllables_tmp)):
                    syllables_real = np.append(syllables_real,
                                        str.split(syllables_tmp[ss][0:-1],'-'))
                if(len(syllables_tmp) > 0):
                    syllables_real = syllables_real[1:]
                else:
                    syllables_real = []
                y = 0
                syllablelength = np.empty([len(syllables_real)])
                for ww in syllables_real:
                    syllablelength[y] = len(ww)
                    y = y+1
                total_syllablelength = sum(syllablelength)
                t1 = onset[segs_to_consider[j]]
                t2 = offset[segs_to_consider[j]]
                dur_real = t2-t1
                uniform_syllablelengths = syllablelength/total_syllablelength*dur_real
                startpos = 0
                for jj in range(0,len(syllables_real)):
                    i1 = range(int((t1+startpos)*100),
                               int((t1+startpos+uniform_syllablelengths[jj])*100))
                    i2 = range(int(SAD_onset*100),int(SAD_offset*100))
                    # Number of overlapping elements
                    olap = len(list(set(i1) & set(i2)))
                    # Proportion of utterance overlapping with SAD segment
                    coverage = (olap/100.0)/dur_real
                    if(coverage > 0):
                        SAD_segments_syllables[sadseg] += 1
                    startpos += uniform_syllablelengths[jj]

    return tot_words, tot_syls, SAD_segments_words, SAD_segments_syllables, wavList


def process_annotations(audio_dir, eaf_dir, rttm_dir, sad_name, selcha_script_path):
    """
    Process all annotations files in a given dir using the previous functions:
        1- retrieve information from the annotations
        2- cut the SAD segments
        3- count the number of words in those segments in annotations
        4- compute alpha

    Parameters
    ----------
    audio_dir : str
        Path to the directory of the audio files.
    eaf_dir : str
        Path to the directory of the annotations files.
    rttm_dir : str
        Path to the directory of the SAD files.
    sad_name : str
        Name of the SAD program used.
    selcha_script_path : str
        Path to script in charge of enriching the files.

    Returns
    -------
    tot_files_words : list
        List of tuples containing filenames and their respective word counts.
    tot_segments_words : list
        List of the word counts of all the segments coming from the files in
        audio_dir.
    wav_list : list
        List of the path to those segments' .wav (same order as previous list).
    alpha : float
        Alpha value, corresponding to the ratio of segments' number of words
        on the real number of words, to correct the error of the SAD when
        predicting later on.
    """

    eaf_files = glob.glob(os.path.join(eaf_dir, '*.eaf'))
    if not eaf_files:
        sys.exit("annotations_processing.py : No annotation file found in {}.".format(eaf_dir))

    tot_files_words = []
    tot_syls = []
    tot_segments_words = []
    tot_segments_syls = []
    wav_list = []

    chunks_dir = os.path.join(audio_dir, "wav_chunks")
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)
    else:
        shutil.rmtree(chunks_dir)
        os.makedirs(chunks_dir)

    for eaf_path in eaf_files:
        print("Processing %s" % eaf_path)

        txt_path = eaf2txt(eaf_path, eaf_dir)
        enrich_txt_path = enrich_txt(txt_path, selcha_script_path)

        rttm_name = "{}_{}.rttm".format(sad_name, os.path.basename(txt_path[:-4]))
        rttm_path = os.path.join(rttm_dir, rttm_name)
        audio_name = "{}.wav".format(os.path.basename(txt_path[:-4]))
        audio_path = os.path.join(audio_dir, audio_name)

        print("Searching for {} and {}".format(rttm_path, audio_path))

        if os.path.isfile(rttm_path):
            if os.path.isfile(audio_path):
                tw, ts, sw, ss, wl = count_annotations_words(enrich_txt_path, rttm_path,
                                                             audio_path, chunks_dir)
                tot_files_words.append((os.path.basename(audio_path)[:-4], tw))
                tot_syls.append(ts)
                tot_segments_words.append(sw)
                tot_segments_syls.append(ss)
                wav_list.append(wl)
                os.remove(enrich_txt_path)
            else:
                sys.exit("Missing .wav file for {}.".format(eaf_path))
        else:
            sys.exit("Missing .rttm for {}.".format(eaf_path))

    tot_segments_words = np.concatenate(tot_segments_words)
    wav_list = np.concatenate(wav_list)

    n = sum(x[1] for x in tot_files_words)
    alpha = np.sum(tot_segments_words) / n

    return tot_files_words, tot_segments_words, wav_list, alpha


def save_reference(tot_files_words, output_path):
    """
    Additional function to save the results to a reference
    file for future comparison.

    Parameters
    ----------
    tot_files_words : list
        List of tuples containing filenames and their respective word counts.
    output_path : str
        Path to the file where to store the reference .csv file.
    """

    if not os.path.exists(os.path.dirname(output_path)):
        raise IOError("Output directory does not exist.")

    with open(output_path, 'w') as ref:
        csvwriter = csv.writer(ref, delimiter=';')
        for row in tot_files_words:
            csvwriter.writerow(row)

