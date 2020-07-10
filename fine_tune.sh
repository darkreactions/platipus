#!/bin/sh
# Run this to train/test full model
# All print statements are stored in the designated folder

log_folder="./fine_tuning_logs_$(date +%Y%m%d_%H%M%S)"

mkdir $log_folder

python run_al.py 2>&1 | tee ./$log_folder/$(date +%Y%m%d_%H%M%S).txt