#!/bin/bash


for i in {1..6}; do
    python cli.py train ../data/${i}/train ../data/${i}/train ../data/${i}/train  tocomboSad -w ../models/word_count_estimator/${i}_toco.pickle
    python cli.py predict ../data/${i}/predict ../data/${i}/predict/ tocomboSad ../results/total_results/tocombo_${i}.csv -w ../models/word_count_estimator/${i}_toco.pickle
done
