#!/bin/sh
# Run this to train/test full model
# All print statements are stored in the designated folder

log_folder="./running_logs_$(date +%Y%m%d_%H%M%S)"

mkdir $log_folder

python main.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --Lt=1 --num_inner_updates=10 --Lv=10 --kl_reweight=.0001 --num_epochs=3000 --num_epochs_save=1000 --cross_validate --resume_epoch=0 --verbose --p_dropout_base=0.4 2> ./$log_folder/PLATIPUS_$(date +%Y%m%d_%H%M%S).txt

python maml.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --num_inner_updates=10 --num_epochs=3000 --num_epochs_save=1000 --resume_epoch=0 --cross_validate --verbose --p_dropout_base=0.4 2> ./$log_folder/MAML_$(date +%Y%m%d_%H%M%S).txt

python main.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --Lt=1 --Lv=100 --num_inner_updates=10 --kl_reweight=.0001 --num_epochs=0 --num_epochs_save=1000 --resume_epoch=3000 --cross_validate --verbose --p_dropout_base=0.4 --test 2> ./$log_folder/cv_$(date +%Y%m%d_%H%M%S).txt