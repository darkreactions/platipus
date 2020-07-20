#!/bin/sh
# Run this to train/test full model
# All print statements are stored in the designated folder

log_folder="./fine_tuning_logs"

category='category-3'

mkdir $log_folder

for i in $(seq 0 500 4500)
  do
    start=$i
    end=$((i+500))
    python ft_svm.py --category=$category --start=$start --end=$end 2>&1 | tee ./$log_folder/$category-$start-$end.txt
done

python ft_svm.py --category=$category --start=5000 2>&1 | tee ./$log_folder/$category-$start-end.txt