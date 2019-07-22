"""Annotations Processing

Module to process the annotations files (.eaf).

It contains:

    * eaf2txt - returns a .txt containing timestamps and transcription from an
    annotation file.
    * enrich_txt - enriches the .txt file by adding information at each line.
    * count_annotations_words - counts the number of "seen" words, and real number
    of words in each audio file with regard to the SAD and the annotations.
    * calc_alpha - compute the alpha coefficient from the number of "seen" words
    and the number of real words.
    * process_annotations - main function using all above functions.
    (see more info in each function's docstring)
"""

import csv, sys, os
import numpy as np
import soundfile as sf
import pympi as pmp
import argparse
import os
import glob
import subprocess
import sys


def eaf2txt(path_to_eaf, output_folder):
    """
    Convert an eaf file to the txt format by extracting the onset, offset, ortho,
    and the speaker tier.
    Only the speaker tiers MOT, FAT and CHI are selected. Any other tier is 
    ignored.

    Parameters
    ----------
    path_to_eaf : str
        Path to the eaf file.
    output_folder : str
        Path to the output directory. 

    Write a txt whose name is the same than the eaf's one in output_folder
    """

    basename = os.path.splitext(os.path.basename(path_to_eaf))[0]
    output_path = os.path.join(output_folder, basename + '.txt')
    
    with open(output_path, 'w') as output_file:

        EAF = pmp.Elan.Eaf(path_to_eaf)
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


def enrich_txt(path_to_txt, path_to_script):
    """"
    Enriches the .txt file by syllabifying the transcriptions, counting the
    number of syllables and words.
    
    Parameters
    ----------
    path_to_txt : str
        Path to the .txt file to be enriched.
    path_to_script : str
        Path to scripts in charge of enriching the file.

    Returns
    -------
    entxt_path : str
        Path to the enriched file.
    """

    cmd = "{} {} {}_enriched.txt spanish".format(path_to_script, path_to_txt,
                                                 path_to_txt[:-4])
    subprocess.call(cmd, shell=True)
    os.remove(path_to_txt)
    entxt_path = "{}_enriched.txt".format(path_to_txt[:-4])
    
    return entxt_path


def count_annotations_words(enrich_file, rttm_file, audio_file, save_dir):
    """
    This function takes an enriched.txt file and .rttm file and measures the
    number of words and syllables in each of the SAD segments defined in the
    rttm file.
    It also extracts .wav files corresponding to these SAD segments from the
    original audio file.

    This is function has a second version see the original function on the
    WCE_VM repo in /aux_VM/combine_rttm_and_enrich.py.

    Parameters
    ----------
    enrich_file : str
        Path to the enriched file (.txt).
    rttm_file : str
        Path to the SAD file (.rttm).
    audio_file : str
        Path to the audio file (.wav).
    save_dir : str
        Path to the output directory.

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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tot_words = sum(wordcount)
    tot_syls = sum(sylcount)

    # Read also the .rttm file to get SAD segment onsets and offsets
    SAD_onsets = np.array([])
    SAD_offsets = np.array([])

    mainWav, rate = sf.read(audio_file)
    wavList = []

    chunks_dir = os.path.join(save_dir, "wav_chunks")
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)

    with open(rttm_file, 'rt') as f:    
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            SAD_onset = float(row[3])
            SAD_offset = float(row[4])
            SAD_onsets = np.append(SAD_onsets, SAD_onset)
            SAD_offsets = np.append(SAD_offsets, SAD_onset+SAD_offset)
            wav_chunk_path = "{}/{}_{}.wav".format(chunks_dir, filename,
                                                   str(int(SAD_onset)*1000).zfill(8))
            chunk_start = int(np.floor(SAD_onset*rate))+1
            chunk_end = int(np.floor(float(SAD_onset*rate+SAD_offset*rate)))-1
            wav_chunk = mainWav[chunk_start:chunk_end]
            sf.write(wav_chunk_path, wav_chunk, rate)
            wavList.append(wav_chunk_path)

    # Count only words that come from segments of the SAD that fully overlap
    # with segments of the reference.
    SAD_segments_words = np.array([])
    SAD_segments_syllables = np.array([])

    for sadseg in range(0,len(SAD_onsets)):
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


def process_annotations(audio_dir, eaf_dir, rttm_dir, sad, selcha_script_path):
    """
    Process all annotations files in a given dir using the previous functions:
        - cut the SAD segments
        - count the number of words in those segments in annotations
        - compute alpha

    Parameters
    ----------
    audio_dir : str
        Path to the directory of the audio files.
    eaf_dir : str 
        Path to the directory of the annotations files.
    rttm_dir : str
        Path to the directory of the SAD files.
    sad : str
        Name of the SAD program used.
    selcha_script_path : str
        Path to scripts in charge of enriching the files.

    Returns
    -------
    tot_segments_words : list
        List of the word counts of all the segments coming from the files in 
        audio_dir.
    wav_list : list
        List of the path to those segments (same order as previous list).
    alpha : float
        Alpha value, corresponding to the ratio of segments' number of words
        on the real number of words, to correct the error of the SAD when
        predicting later on.
    """
    
    eaf_files = glob.iglob(os.path.join(eaf_dir, '*.eaf'))

    tot_words = []
    tot_syls = []
    tot_segments_words = []
    tot_segments_syls = [] 
    wav_list = []

    # TODO: CHANGE sort key to match naming convention + issue when path contains
    # '_'
    for eaf_path in sorted(eaf_files, key=lambda k : (int(k.split('_')[1]),
                                                      int(k.split('_')[-2]))):
        print("Processing %s" % eaf_path)
        txt_path = eaf2txt(eaf_path, eaf_dir)
        enrich_txt_path = enrich_txt(txt_path, selcha_script_path)
        rttm_name = "{}_{}.rttm".format(sad, os.path.basename(txt_path[:-4]))
        rttm_path = os.path.join(rttm_dir, rttm_name)
        audio_name = "{}.wav".format(os.path.basename(txt_path[:-4]))
        audio_path = os.path.join(audio_dir, audio_name)
        print(rttm_path, audio_path)

        if os.path.isfile(rttm_path):
            if os.path.isfile(audio_path):
                tw, ts, sw, ss, wl = count_annotations_words(enrich_txt_path, rttm_path,
                                                             audio_path, audio_dir)
                tot_words.append((os.path.basename(audio_path)[:-4], tw))
                tot_syls.append(ts)
                tot_segments_words.append(sw)
                print(sw)
                tot_segments_syls.append(ss)
                wav_list.append(wl)
            else:
                print("Missing .wav file for {}".format(eaf_path))
        else:
            print("Missing .rttm for {}".format(eaf_path))

    tot_segments_words = np.concatenate(tot_segments_words)
    wav_list = np.concatenate(wav_list)
    
    n = sum(x[1] for x in tot_words)
    alpha = n / np.sum(tot_segments_words)
    
    return tot_words, tot_segments_words, wav_list, alpha

