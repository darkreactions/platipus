from pathlib import Path

common_params = {
    'datasource': 'drp_chem',
    'cross_validate': True,
    'verbose': False,
    'train_flag': True,
    'gpu_id': 1,
    'test_data': True,  # TODO: redundant with full_dataset?
    'meta': False,
    'full_dataset': True,
    'fine_tuning': True,
    'with_historical_data': True,   # Train models with historical data of other amines
    'with_k': False,     # Train the model with k additional amine-specific experiments
    'train_size': 10,   # k after pretrain
    'active_learning': False,
    'active_learning_iter': 10,   # x before active learning
    'stats_path': Path('./results/cv_statistics.pkl'),
    'cv_stats_overwrite': True,     # TODO: change this to false when running all models
    'save_model': False
}

knn_configs = {
    'category_3': {
        'n_neighbors': 3,
        'leaf_size': 30,
        'p': 1
    },
    'category_4_i': {
        'n_neighbors': 1,
        'leaf_size': 1,
        'p': 1
    },
    'category_4_ii': {
        'n_neighbors': 3,
        'leaf_size': 30,
        'p': 1
    },
    'category_5_i': {
        'n_neighbors': 3,
        'leaf_size': 30,
        'p': 1
    },
    'category_5_ii': {
        'n_neighbors': 3,
        'leaf_size': 30,
        'p': 1
    },
}

knn_params = {
    'neighbors': 2,
    'configs': knn_configs,
    'model_name': 'KNN'
}

svm_configs = {
    'category_3': {

    },
    'category_4_i': {

    },
    'category_4_ii': {

    },
    'category_5_i': {

    },
    'category_5_ii': {

    },
}

svm_params = {
    'configs': svm_configs,
    'model_name': 'SVM'
}

rf_configs = {
    'category_3': {
            'n_estimators': 1000,
            'criterion': 'gini',
            'max_depth': 8,
            'max_features': None,
            'bootstrap': True,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'ccp_alpha': 0.0
    },
    'category_4_i': {
            'n_estimators': 200,
            'criterion': 'gini',
            'max_depth': 8,
            'max_features': None,
            'bootstrap': True,
            'min_samples_leaf': 2,
            'min_samples_split': 5,
            'ccp_alpha': 0.0
    },
    'category_4_ii': {
        'n_estimators': 200,
        'criterion': 'gini',
        'max_depth': 7,
        'max_features': None,
        'bootstrap': True,
        'min_samples_leaf': 2,
        'min_samples_split': 10,
        'ccp_alpha': 0.0
    },
    'category_5_i': {
        'n_estimators': 100,
        'criterion': 'gini',
        'max_depth': 8,
        'max_features': None,
        'bootstrap': True,
        'min_samples_leaf': 3,
        'min_samples_split': 5,
        'ccp_alpha': 0.0
    },
    'category_5_ii': {
        'n_estimators': 200,
        'criterion': 'gini',
        'max_depth': 7,
        'max_features': None,
        'bootstrap': True,
        'min_samples_leaf': 3,
        'min_samples_split': 5,
        'ccp_alpha': 0.0
    },
}

randomforest_params = {
    'config': rf_configs,
    'model_name': 'Random_Forest'
}

lr_configs = {
    'category_3': {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 0.1,
        'solver': 'lbfgs',
        'max_iters': 4000
    },
    'category_4_i': {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 0.1,
        'solver': 'lbfgs',
        'max_iters': 4000
    },
    'category_4_ii': {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 0.1,
        'solver': 'lbfgs',
        'max_iters': 4000
    },
    'category_5_i': {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 0.1,
        'solver': 'lbfgs',
        'max_iters': 4000
    },
    'category_5_ii': {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 0.1,
        'solver': 'lbfgs',
        'max_iters': 4000
    },
}

logisticregression_params = {
    'config': None,
    'model_name': 'Logistic_Regression'
}

dt_configs = {
    'category_3': {

    },
    'category_4_i': {

    },
    'category_4_ii': {

    },
    'category_5_i': {

    },
    'category_5_ii': {

    },
}

decisiontree_params = {
    'configs': dt_configs,
    'model_name': 'Decision_Tree'
}

gb_configs = {
    'category_3': {

    },
    'category_4_i': {

    },
    'category_4_ii': {

    },
    'category_5_i': {

    },
    'category_5_ii': {

    },
}

gradientboosting_params = {
    'config': None,
    'model_name': 'Gradient_Boosting'
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
