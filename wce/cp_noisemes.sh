#!/bin/bash

for i in {1..6}; do
    rm ~/Documents/BabyCloud/wce/data/${i}/train/noisemes*
    rm ~/Documents/BabyCloud/wce/data/${i}/predict/noisemes*
    cp ~/Documents/BabyCloud/DiViMe/data/total_results/noisemesSad_BBC_*0${i}_* ~/Documents/BabyCloud/wce/data/${i}/predict
    cp ~/Documents/BabyCloud/DiViMe/data/total_results/noisemesSad_BBC_* ~/Documents/BabyCloud/wce/data/${i}/train
    rm ~/Documents/BabyCloud/wce/data/${i}/train/noisemesSad_BBC_*0${i}*
done
