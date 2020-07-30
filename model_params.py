from pathlib import Path

common_params = {
    'active_learning': False,
    'active_learning_iter': 10,  # x before active learning
    'cross_validate': True,
    'cv_stats_overwrite': True,
    'datasource': 'drp_chem',
    'fine_tuning': False,
    'full_dataset': True,
    'gpu_id': 1,
    'meta': False,
    'save_model': False,
    'stats_path': Path('./results/cv_statistics.pkl'),
    'test_data': True,  # TODO: redundant with full_dataset?
    'train_flag': True,
    'train_size': 10,  # k after pretrain
    'verbose': False,
    'warning': True,
    'with_historical_data': True,  # Train models with historical data of other amines
    'with_k': False,  # Train the model with k additional amine-specific experiments
}

knn_configs = {
    'category_3': {
        'n_neighbors': 3,
        'leaf_size': 1,
        'p': 1
    },
    'category_4_i': {
        'n_neighbors': 1,
        'leaf_size': 1,
        'p': 1
    },
    'category_4_ii': {
        'n_neighbors': 3,
        'leaf_size': 1,
        'p': 1
    },
    'category_5_i': {
        'n_neighbors': 1,
        'leaf_size': 1,
        'p': 1
    },
    'category_5_ii': {
        'n_neighbors': 1,
        'leaf_size': 1,
        'p': 3
    },
}

knn_params = {
    'neighbors': 2,
    'configs': knn_configs,
    'model_name': 'KNN'
}

svm_configs = {
    'category_3': {
        'kernel': 'poly',
        'C': 0.001,
        'degree': 3,
        'gamma': 'auto',
        'tol': 0.001,
        'decision_function_shape': 'ovo',
        'break_ties': True,
        'class_weight': None
    },
    'category_4_i': {
        'kernel': 'poly',
        'C': 0.003,
        'degree': 3,
        'gamma': 'auto',
        'tol': 2,
        'decision_function_shape': 'ovo',
        'break_ties': True,
        'class_weight': {0: 0.3, 1: 0.7}
    },
    'category_5_i': {
        'kernel': 'poly',
        'C': 0.001,
        'degree': 2,
        'gamma': 'auto',
        'tol': 1,
        'decision_function_shape': 'ovo',
        'break_ties': True,
        'class_weight': None
    },
}

svm_params = {
    'configs': svm_configs,
    'model_name': 'SVM'
}

linearsvm_configs = {
    'category_3': {
        'penalty': 'l1',
        'loss': 'squared_hinge',
        'dual': False,
        'C': 0.01,
        'tol': 0.08875,
        'fit_intercept': True,
        'class_weight': {0: 0.05, 1: 0.95}
    },
    'category_4_i': {
        'penalty': 'l2',
        'loss': 'squared_hinge',
        'dual': False,
        'C': 0.001,
        'tol': 0.15,
        'fit_intercept': True,
        'class_weight': {0: 0.09, 1: 0.91}
    },
    'category_5_i': {
        'penalty': 'l1',
        'loss': 'squared_hinge',
        'dual': False,
        'C': 0.01,
        'tol': 1,
        'fit_intercept': False,
        'class_weight': {0: 0.05, 1: 0.95}
    },
}

linearsvm_params = {
    'configs': linearsvm_configs,
    'model_name': 'LinearSVM'
}

dt_configs = {
    'category_3': {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': 7,
        'min_samples_split': 2,
        'min_samples_leaf': 2
    },
    'category_4_i': {
        'criterion': 'gini',
        'splitter': 'random',
        'max_depth': 11,
        'min_samples_split': 8,
        'min_samples_leaf': 1,
        'class_weight': {0: 0.3, 1: 0.7}
    },
    'category_4_ii': {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': 3,
        'min_samples_split': 4,
        'min_samples_leaf': 1
    },
    'category_5_i': {
        'criterion': 'gini',
        'splitter': 'random',
        'max_depth': 11,
        'min_samples_split': 4,
        'min_samples_leaf': 3,
        'class_weight': {0: 0.1, 1: 0.9}
    },
    'category_5_ii': {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': 4,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
}

decisiontree_params = {
    'configs': dt_configs,
    'model_name': 'Decision_Tree',
    'visualize': False
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
        'max_iter': 4000
    },
    'category_4_i': {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 0.1,
        'solver': 'lbfgs',
        'max_iter': 4000
    },
    'category_4_ii': {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 0.1,
        'solver': 'lbfgs',
        'max_iter': 4000
    },
    'category_5_i': {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 0.1,
        'solver': 'lbfgs',
        'max_iter': 4000
    },
    'category_5_ii': {
        'penalty': 'l2',
        'dual': False,
        'tol': 1e-4,
        'C': 0.1,
        'solver': 'lbfgs',
        'max_iter': 4000
    },
}

logisticregression_params = {
    'config': lr_configs,
    'model_name': 'Logistic_Regression'
}

dt_configs = {
    'category_3': {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': 7,
        'min_samples_split': 2,
        'min_samples_leaf': 2
    },
    'category_4_i': {
        'criterion': 'gini',
        'splitter': 'random',
        'max_depth': 11,
        'min_samples_split': 8,
        'min_samples_leaf': 1,
        'class_weight': {0: 0.3, 1: 0.7}
    },
    'category_4_ii': {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': 3,
        'min_samples_split': 4,
        'min_samples_leaf': 1
    },
    'category_5_i': {
        'criterion': 'gini',
        'splitter': 'random',
        'max_depth': 11,
        'min_samples_split': 4,
        'min_samples_leaf': 3,
        'class_weight': {0: 0.1, 1: 0.9}
    },
    'category_5_ii': {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': 4,
        'min_samples_split': 2,
        'min_samples_leaf': 1
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
