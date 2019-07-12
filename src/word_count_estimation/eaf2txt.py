#!/usr/bin/env python

"""
This script converts an eaf file into a txt file 
containing the following information :

    onset offset transcription receiver speaker_tier

It can be run either on a single eaf file,
or on a whole folder containing eaf files.

Example of use :
    python tools/eaf2txt.py -i data/0396.eaf    # One one file
    python tools/eaf2txt.py -i data/            # On a whole folder

About the naming convention of the output :
    For each file called input_file.eaf,
    the result will be stored in input_file.txt
"""

import pympi as pmp
import argparse
import os
import glob
import subprocess
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def eaf2txt(path_to_eaf, output_folder):
    """
    Convert an eaf file to the txt format by extracting the onset, offset, ortho,
    and the speaker tier. Note that the ortho field has been made by a human and needs
    to be cleaned up.
    Only the speaker tiers MOT, FAT and CHI are selected. Any other tier is 
    ignored.

    Parameters
    ----------
    path_to_eaf :   path to the eaf file.

    Write a txt whose name is the same than the eaf's one in output_folder
    """
    basename = os.path.splitext(os.path.basename(path_to_eaf))[0]
    output_path = os.path.join(output_folder, basename + '.txt')
    output_file = open(output_path, 'w')

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

            output_file.write("%d\t%d\t%s\t%s\t%s\n" % (onset, offset, '',
                                                        transcript, speaker))

    output_file.close()
    return output_path


def enrich_txt(path_to_txt, output_folder):

    cmd = "./selcha2clean.sh {} {}_enrich.txt spanish".format(path_to_txt,
                                                              path_to_txt[:-4])
    subprocess.call(cmd, shell=True)
    os.remove(path_to_txt)



def calc_alpha():
    pass

datadir = "../../data/3/"
eaf_files = glob.iglob(os.path.join(datadir, '*.eaf'))
for eaf_path in eaf_files:
    print("Processing %s" % eaf_path)
    outpath = eaf2txt(eaf_path, datadir)
    enrich_txt(outpath, datadir)

