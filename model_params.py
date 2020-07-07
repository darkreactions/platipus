from pathlib import Path

common_params = {
    'datasource': 'drp_chem',
    'cross_validate': True,
    'verbose': False,
    'train_flag': True,
    'gpu_id': 1,
    'test_data': True,  # TODO: redundant with full_dataset?
    'meta': False,  # TODO: redundant with pretrain?
    'full_dataset': True,
    'fine_tuning': False,
    'pretrain': True,
    'train_size': 10,   # k after pretrain
    'active_learning_iter': 10,   # x before active learning
    'stats_path': Path('./results/cv_statistics.pkl'),
    'cv_stats_overwrite': True,
    'save_model': False
}

knn_params = {
    'neighbors': 2,
    'config': None,
    'model_name': 'KNN'
}

"""
The format for KNN's config should be a dictionary in the following form
{
    'n_neighbors': 2,
    'weights': 'uniform',
    'algorithm': 'auto',
    'leaf_size': 20,
    'p': 2,
    'metric': 'minkowski'
}
"""

svm_params = {
    'config': None,
    'model_name': 'SVM'
}

"""
The format for SVM config should be a dictionary in the following form
{
    'C':0.01,
    'kernel':'poly',
    'degree': 1,
    'gammas': 'scale',
    'shrinking': True,
    'tol': 0.01,
    'decision_function_shape': 'ovo',
    'break_ties': True,
    'class_weight': 'balanced'
}
"""

randomforest_params = {
    'config': None,
    'model_name': 'Random_Forest'
}

logisticregression_params = {
    'config': None,
    'model_name': 'Logistic_Regression'
}

decisiontree_params = {
    'config': None,
    'model_name': 'Decision_Tree'
}

meta_params = {
    'k_shot': 10,
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
