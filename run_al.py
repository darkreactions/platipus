import models.meta.main as platipus
from pathlib import Path
import os

from models.non_meta import RandomForest, KNN, SVM, DecisionTree, LogisticRegression
from utils.plot import plot_metrics_graph
from utils import read_pickle, write_pickle, define_non_meta_model_name, find_avg_metrics
from model_params import common_params, knn_params, svm_params, randomforest_params, logisticregression_params, decisiontree_params,  meta_train, meta_test


if __name__ == '__main__':
    # Set up the results directory
    results_folder = './results'

    # TODO: maybe move this to a function in utils?
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print('No folder for results storage found')
        print('Make folder to store results at')
    else:
        print('Found existing folder. All results will be stored at')
    print(results_folder)

    # Append the data to cv_stats or overwrite the current results
    # TODO: It seems that this doesn't delete/overwrite the pkl file. Maybe wrong path?
    overwrite = common_params['cv_stats_overwrite']
    cv_stats_dst = common_params['stats_path']
    if os.path.exists(cv_stats_dst) and overwrite:
        print('Overwriting the current cv_stats.pkl')
        os.remove(cv_stats_dst)

    # Record used models for plotting later
    models_to_plot = []

    # Training different models
    # PLATIPUS
    # platipus_train_params = {**common_params, **meta_params, **meta_train}
    # params = platipus.initialize(["PLATIPUS"], platipus_train_params)
    # platipus.main(params)

    # platipus_test_params = {**common_params, **meta_params, **meta_test}
    # params = platipus.initialize(["PLATIPUS"], platipus_test_params)
    # platipus.main(params)

    # TODO: MAML

    # KNN w/ active learning
    # Trained under option 1
    KNN1_params = {**common_params, **knn_params}
    KNN1_params['model_name'] = define_non_meta_model_name(KNN1_params['model_name'], KNN1_params['pretrain'])
    models_to_plot.append(KNN1_params['model_name'])
    KNN.run_model(KNN1_params)

    # Trained under option 2
    KNN2_params = {**common_params, **knn_params}
    KNN2_params['pretrain'] = False
    KNN2_params['model_name'] = define_non_meta_model_name(KNN2_params['model_name'], KNN2_params['pretrain'])
    models_to_plot.append(KNN2_params['model_name'])
    KNN.run_model(KNN2_params)

    # SVM w/ active learning
    # Trained under option 1
    SVM1_params = {**common_params, **svm_params}
    SVM1_params['model_name'] = define_non_meta_model_name(SVM1_params['model_name'], SVM1_params['pretrain'])
    models_to_plot.append(SVM1_params['model_name'])
    SVM.run_model(SVM1_params)

    '''
    TODO: CAN'T RUN DUE TO INSUFFICIENT SUCCESSES
    # Trained under option 2
    SVM2_params = {**common_params, **svm_params}
    SVM2_params['pretrain'] = False
    SVM2_params['model_name'] = define_non_meta_model_name(SVM2_params['model_name'], SVM2_params['pretrain'])
    models_to_plot.append(SVM2_params['model_name'])
    SVM.run_model(SVM2_params)
    '''

    # Random Forest w/ active learning
    # Trained under option 1
    RF1_params = {**common_params, **randomforest_params}
    RF1_params['model_name'] = define_non_meta_model_name(RF1_params['model_name'], RF1_params['pretrain'])
    models_to_plot.append(RF1_params['model_name'])
    RandomForest.run_model(RF1_params)

    # Trained under option 2
    RF2_params = {**common_params, **randomforest_params}
    RF2_params['pretrain'] = False
    RF2_params['model_name'] = define_non_meta_model_name(RF2_params['model_name'], RF2_params['pretrain'])
    models_to_plot.append(RF2_params['model_name'])
    RandomForest.run_model(RF2_params)

    LR1_params = {**common_params, **logisticregression_params}
    LR1_params['model_name'] = define_non_meta_model_name(LR1_params['model_name'], LR1_params['pretrain'])
    models_to_plot.append(LR1_params['model_name'])
    LogisticRegression.run_model(LR1_params)

    # Trained under option 2
    '''
    TODO: CAN'T RUN DUE TO INSUFFICIENT SUCCESSES
    LR2_params = {**common_params, **logisticregression_params}
    LR2_params['pretrain'] = False
    LR2_params['model_name'] = define_non_meta_model_name(LR2_params['model_name'], LR2_params['pretrain'])
    models_to_plot.append(LR2_params['model_name'])
    LogisticRegression.run_model(LR2_params)
    '''

    # Random Forest w/ active learning
    # Trained under option 1
    DT1_params = {**common_params, **decisiontree_params}
    DT1_params['model_name'] = define_non_meta_model_name(DT1_params['model_name'], DT1_params['pretrain'])
    models_to_plot.append(DT1_params['model_name'])
    DecisionTree.run_model(DT1_params)

    # Trained under option 2
    DT2_params = {**common_params, **decisiontree_params}
    DT2_params['pretrain'] = False
    DT2_params['model_name'] = define_non_meta_model_name(DT2_params['model_name'], DT2_params['pretrain'])
    models_to_plot.append(DT2_params['model_name'])
    DecisionTree.run_model(DT2_params)

    cv_stats = read_pickle(common_params['stats_path'])
    amines = cv_stats[models_to_plot[0]]['amine']
    # print(cv_stats.keys())
    # print(amines)

    # Plotting individual graphs for each task-specific model
    for i, amine in enumerate(amines):
        plot_metrics_graph(96, cv_stats, './results', amine=amine, amine_index=i, models=models_to_plot)

    # Plotting avg graphs for all models
    avg_stats = find_avg_metrics(cv_stats)

    rand_model = list(avg_stats.keys())[0]
    num_examples = len(avg_stats[rand_model]['accuracies'])
    plot_metrics_graph(num_examples, avg_stats, './results')
