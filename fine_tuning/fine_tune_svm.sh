#!/bin/sh
# Run this to train/test full model
# All print statements are stored in the designated folder
category='Hkx'

log_folder="./fine_tuning_logs/$category"

mkdir $log_folder

for i in $(seq 0 1 5279)
  do
    timeout 60s python ft_svm.py --category=$category --index=$i
done 2>&1 | tee ./$log_folder/$category.txt