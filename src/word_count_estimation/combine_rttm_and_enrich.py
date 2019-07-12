# This function takes an enriched.txt file and .rttm file and measures the
# number of words and syllables in each of the SAD segments defined in the .rttm file.

import csv, sys, os
import numpy as np
import soundfile as sf


def count_annotations_words(enriched_file, rttm_file, audio_file, save_dir):

    path, filename = os.path.split(enrich_file)
    path2, filename2 = os.path.split(rttm_file)

    curr_file = audio_file[:-4] 

    # Read enriched file first to get original annotated utterances and their information

    index = 0
    onset = np.array([])
    offset = np.array([])
    sylcount = np.array([])
    wordcount = np.array([])

    speaker = np.array([])
    ortho = np.array([])
    syllables = np.array([])

    with open(enrich_file, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            onset = np.append(onset,float(row[0]))
            offset = np.append(offset,float(row[1]))
            speaker = np.append(speaker,row[3])
            ortho = np.append(ortho,row[4])
            wordcount = np.append(wordcount,float(row[5]))
            syllables = np.append(syllables,row[6])
            sylcount = np.append(sylcount,float(row[7]))


    order = np.argsort(onset)
    onset = onset[order]
    offset = offset[order]
    wordcount = wordcount[order]
    sylcount = sylcount[order]
    ortho = ortho[order]
    syllables = syllables[order]
    speaker = speaker[order]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    no_words_file = save_dir + '/' + curr_sp + '_totWords.txt'
    file = open(no_words_file,'w')
    file.write(str(sum(wordcount)))
    file.close()

    no_syls_file =  save_dir + '/' + curr_sp + '_totSyls.txt'
    file = open(no_syls_file,'w')
    file.write(str(sum(sylcount)))
    file.close()

    onset = onset/1000
    offset = offset/1000


    # Read also the .rttm file to get SAD segment onsets and offsets

    # Read 2/5min file onset and offset timestamps from the rttm_filename
    orig_onset = float(rttm_file[-18:-12])
    orig_offset = float(rttm_file[-11:-5])

    SAD_onsets = np.array([])
    SAD_offsets = np.array([])

    sp_main_file = audio_dir + curr_sp + '.wav'
    rate, mainWav = sf.read(sp_main_file)
    wavList = []

    with open(rttm_file, 'rt') as f:    
        reader = csv.reader(f, delimiter=' ')
        for row in reader:

            SAD_onsets = np.append(SAD_onsets,float(row[3]))
            SAD_onsets_tmp = float(row[3])
            SAD_offsets = np.append(SAD_offsets,float(row[4])+float(row[3]))
            
            short_wav_file = save_dir + curr_sp + '_' + str(int(float(row[3])*1000)).zfill(8) + '.wav'
            SAD_onsets_tmp = int(np.floor(float(row[3])*rate))+1
            SAD_offsets_tmp = int(np.floor(float(row[4])*rate+float(row[3])*rate))-1
            short_wav = mainWav[SAD_onsets_tmp:SAD_offsets_tmp]
            sf.write(short_wav_file, short_wav, rate)
            wavList.append(short_wav_file)

    with open(save_dir + '/' +curr_sp + '_wavList.txt', 'w') as f:
        for item in wavList:
            f.write("%s\n" % item)


    # Go through each SAD segment and find the overlapping annotated segments, assign the number of words
    # from each utterance multiplied by the proportion of overlap

    # NOTE: this should be done under the word unformity lenghth assumption
    """
    SAD_segment_words = np.array([])
    SAD_segment_syllables = np.array([])

    for sadseg in range(0,len(SAD_onset)):
        SAD_segment_words = np.append(SAD_segment_words,0)
        SAD_segment_syllables = np.append(SAD_segment_syllables,0)
        SAD_onset = SAD_onset[sadseg]
        SAD_offset = SAD_offset[sadseg]
        # These segments must be fully within the SAD segment
        # Segments that end after SAD onset
        tmp1 = offset-SAD_onset > 0
        # Segments that begin before SAD offset
        tmp2 = onset-SAD_offset < 0
        # Valid segments are the ones with overlap
        tmp3 = (tmp1 & tmp2)
        # Which original segments overlap with the target segment
        if(sum(tmp3)> 0):
            i = np.where(tmp3 > 0)
            segs_to_consider = range(max(0,i[0][0]-1),min(len(tmp1),i[0][-1]+2))
            # Calculate exact overlapping
            for j in range(0,len(segs_to_consider)):
                t1 = onset[segs_to_consider[j]]
                t2 = offset[segs_to_consider[j]]
                dur_real = t2-t1;
                i1 = range(int(t1*100),int(t2*100))
                i2 = range(int(SAD_onset*100),int(SAD_offset*100))
                # Number of overlapping elements
                olap = len(list(set(i1) & set(i2)))
                # Proportion of utterance overlapping with SAD segment
                coverage = (olap/100.0)/dur_real
                total_words_in_real = wordcount[segs_to_consider[j]]
                total_syls_in_real = sylcount[segs_to_consider[j]]
                words_to_assign = total_words_in_real*coverage
                syllables_to_assign = total_syls_in_real*coverage
                SAD_segment_words[sadseg] = SAD_segment_words[sadseg]+words_to_assign
                SAD_segment_syllables[sadseg] = SAD_segment_syllables[sadseg]+syllables_to_assign
    """

    # Version 2 with uniform spacing assumption

    SAD_segment_words = np.array([])
    SAD_segment_syllables = np.array([])

    for sadseg in range(0,len(SAD_onsets)):
        SAD_segment_words = np.append(SAD_segment_words,0)
        SAD_segment_syllables = np.append(SAD_segment_syllables,0)
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
        if(sum(tmp3)> 0):
            i = np.where(tmp3 > 0)
            segs_to_consider = range(max(0,i[0][0]-1),min(len(tmp1),i[0][-1]+2))
            # Calculate exact overlapping
            for j in range(0,len(segs_to_consider)):
                total_words_in_real = wordcount[segs_to_consider[j]]
                total_syls_in_real = sylcount[segs_to_consider[j]]            
                words_real = str.split(ortho[segs_to_consider[j]])
                y = 0
                wordlength = np.empty([len(words_real)])
                for ww in words_real:
                    wordlength[y] = len(ww)
                    y = y+1
                total_wordlength = sum(wordlength)

                # Where real segment starts adnd ends
                t1 = onset[segs_to_consider[j]]
                t2 = offset[segs_to_consider[j]]
                dur_real = t2-t1
                uniform_wordlengths = wordlength/total_wordlength*dur_real
                startpos = 0
                for jj in range(0,len(words_real)):
                    i1 = range(int((t1+startpos)*100),int((t1+startpos+uniform_wordlengths[jj])*100))
                    i2 = range(int(SAD_onset*100),int(SAD_offset*100))
                    # Number of overlapping elements
                    olap = len(list(set(i1) & set(i2)))
                    # Proportion of utterance overlapping with SAD segment
                    coverage = (olap/100.0)/dur_real
                    if(coverage > 0):
                        SAD_segment_words[sadseg] = SAD_segment_words[sadseg]+1
                    startpos = startpos+uniform_wordlengths[jj]
                # Same for syllables
                total_syllables_in_real = sylcount[segs_to_consider[j]]            
                syllables_real = np.empty([])
                syllables_tmp = str.split(syllables[segs_to_consider[j]])
                for ss in range (0,len(syllables_tmp)):
                    syllables_real = np.append(syllables_real,str.split(syllables_tmp[ss][0:-1],'-'))
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
                    i1 = range(int((t1+startpos)*100),int((t1+startpos+uniform_syllablelengths[jj])*100))
                    i2 = range(int(SAD_onset*100),int(SAD_offset*100))
                    # Number of overlapping elements
                    olap = len(list(set(i1) & set(i2)))
                    # Proportion of utterance overlapping with SAD segment
                    coverage = (olap/100.0)/dur_real
                    if(coverage > 0):
                        SAD_segment_syllables[sadseg] = SAD_segment_syllables[sadseg]+1
                    startpos = startpos+uniform_syllablelengths[jj]

    word_post_file = save_dir + '/' + curr_sp + '_SAD_segment_words_out.txt'
    syl_post_file = save_dir + '/' + curr_sp + '_SAD_segment_syllables_out.txt'
    if(save_dir):    
        np.savetxt(word_post_file ,SAD_segment_words,fmt='%i')
        np.savetxt(syl_post_file ,SAD_segment_syllables,fmt='%i')
    else:
        np.savetxt('words_out.txt',SAD_segment_words,fmt='%i')
        np.savetxt('syllables_out.txt',SAD_segment_syllables,fmt='%i')
