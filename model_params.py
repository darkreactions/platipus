from pathlib import Path

common_params = {
    'datasource': 'drp_chem',
    'cross_validate': True,
    'verbose': False,
    'train_flag': True,
    'gpu_id': 1,
    'test_data': True,
    'meta': False,
    'full_dataset': False,
    'pretrain': True,
    'stats_path': Path('./results/cv_statistics.pkl'),
    'cv_stats_overwrite': True
}

knn_params = {
    'neighbors': 2,
    'model_name': 'Knn-2'
}

randomforest_params = {
    'train_size': 20,
    'config': None,
    'model_name': 'Random Forest'
}

meta_params = {
    'k_shot': 20,
    'n_way': 2,
    'inner_lr': 1e-3,
    'meta_lr': 1e-3,
    'pred_lr': 1e-1,
    'meta_batch_size': 10,
    'Lt': 1,
    'Lv': 100,
    'num_inner_updates': 10,
    'kl_reweight': 0.0001,
    'p_dropout_base': 0.4,
    'num_epochs_save': 2,
    'num_batches': 250,
    'model_name': 'Platipus'
}

meta_train = {
    'num_epochs': 4,
    'resume_epoch': 0,
    'train_flag': True
}

meta_test = {
    'num_epochs': 0,
    'resume_epoch': 4,
    'train_flag': False
}
