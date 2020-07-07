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

    '''    
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
    # Category 3
    KNN1_params = {**common_params, **knn_params}
    KNN1_params['model_name'] = define_non_meta_model_name(
        KNN1_params['model_name'],
        KNN1_params['active_learning'],
        KNN1_params['with_historical_data'],
        KNN1_params['with_k']
    )
    models_to_plot.append(KNN1_params['model_name'])
    KNN.run_model(KNN1_params)

    # Category 4.1
    KNN2_params = {**common_params, **knn_params}
    KNN2_params['with_k'] = True
    KNN2_params['model_name'] = define_non_meta_model_name(
        KNN2_params['model_name'],
        KNN2_params['active_learning'],
        KNN2_params['with_historical_data'],
        KNN2_params['with_k']
    )
    models_to_plot.append(KNN2_params['model_name'])
    KNN.run_model(KNN2_params)

    # Category 4.2
    KNN3_params = {**common_params, **knn_params}
    KNN3_params['with_historical_data'] = False
    KNN3_params['with_k'] = True
    KNN3_params['model_name'] = define_non_meta_model_name(
        KNN3_params['model_name'],
        KNN3_params['active_learning'],
        KNN3_params['with_historical_data'],
        KNN3_params['with_k']
    )
    models_to_plot.append(KNN3_params['model_name'])
    KNN.run_model(KNN3_params)

    # Category 5.1
    KNN4_params = {**common_params, **knn_params}
    KNN4_params['active_learning'] = True
    KNN4_params['with_k'] = True
    KNN4_params['model_name'] = define_non_meta_model_name(
        KNN4_params['model_name'],
        KNN4_params['active_learning'],
        KNN4_params['with_historical_data'],
        KNN4_params['with_k']
    )
    models_to_plot.append(KNN4_params['model_name'])
    KNN.run_model(KNN4_params)

    # Category 5.2
    KNN5_params = {**common_params, **knn_params}
    KNN5_params['active_learning'] = True
    KNN5_params['with_historical_data'] = False
    KNN5_params['with_k'] = True
    KNN5_params['model_name'] = define_non_meta_model_name(
        KNN5_params['model_name'],
        KNN5_params['active_learning'],
        KNN5_params['with_historical_data'],
        KNN5_params['with_k']
    )
    models_to_plot.append(KNN5_params['model_name'])
    KNN.run_model(KNN5_params)

    # SVM w/ active learning
    # Category 3
    SVM1_params = {**common_params, **svm_params}
    SVM1_params['model_name'] = define_non_meta_model_name(
        SVM1_params['model_name'],
        SVM1_params['active_learning'],
        SVM1_params['with_historical_data'],
        SVM1_params['with_k']
    )
    models_to_plot.append(SVM1_params['model_name'])
    SVM.run_model(SVM1_params)

    # Category 4.1
    SVM2_params = {**common_params, **svm_params}
    SVM2_params['with_k'] = True
    SVM2_params['model_name'] = define_non_meta_model_name(
        SVM2_params['model_name'],
        SVM2_params['active_learning'],
        SVM2_params['with_historical_data'],
        SVM2_params['with_k']
    )
    models_to_plot.append(SVM2_params['model_name'])
    SVM.run_model(SVM2_params)

    """# Category 4.2
    SVM3_params = {**common_params, **svm_params}
    SVM3_params['with_historical_data'] = False
    SVM3_params['with_k'] = True
    SVM3_params['model_name'] = define_non_meta_model_name(
        SVM3_params['model_name'],
        SVM3_params['active_learning'],
        SVM3_params['with_historical_data'],
        SVM3_params['with_k']
    )
    models_to_plot.append(SVM3_params['model_name'])
    SVM.run_model(SVM3_params)"""

    # Category 5.1
    SVM4_params = {**common_params, **svm_params}
    SVM4_params['active_learning'] = True
    SVM4_params['with_k'] = True
    SVM4_params['model_name'] = define_non_meta_model_name(
        SVM4_params['model_name'],
        SVM4_params['active_learning'],
        SVM4_params['with_historical_data'],
        SVM4_params['with_k']
    )
    models_to_plot.append(SVM4_params['model_name'])
    SVM.run_model(SVM4_params)

    """# Category 5.2
    SVM5_params = {**common_params, **svm_params}
    SVM5_params['active_learning'] = True
    SVM5_params['with_historical_data'] = False
    SVM5_params['with_k'] = True
    SVM5_params['model_name'] = define_non_meta_model_name(
        SVM5_params['model_name'],
        SVM5_params['active_learning'],
        SVM5_params['with_historical_data'],
        SVM5_params['with_k']
    )
    models_to_plot.append(SVM5_params['model_name'])
    SVM.run_model(SVM5_params)"""

    """    
    # Random Forest w/ active learning
    # Trained under option 1
    RF1_params = {**common_params, **randomforest_params}
    RF1_params['model_name'] = define_non_meta_model_name(RF1_params['model_name'], RF1_params['pretrain'])
    models_to_plot.append(RF1_params['model_name'])
    RandomForest.run_model(RF1_params)

    # Trained under option 2
    RF2_params = {**common_params, **randomforest_params}
    RF2_params['model_name'] = define_non_meta_model_name(RF2_params['model_name'], RF2_params['pretrain'])
    models_to_plot.append(RF2_params['model_name'])
    RandomForest.run_model(RF2_params)

    # Logistic Regression w/ active learning
    # Trained under option 1
    LR1_params = {**common_params, **logisticregression_params}
    LR1_params['model_name'] = define_non_meta_model_name(LR1_params['model_name'], LR1_params['pretrain'])
    models_to_plot.append(LR1_params['model_name'])
    LogisticRegression.run_model(LR1_params)

    # Trained under option 2
    # TODO: CAN'T RUN DUE TO INSUFFICIENT SUCCESSES
    LR2_params = {**common_params, **logisticregression_params}
    LR2_params['pretrain'] = False
    LR2_params['model_name'] = define_non_meta_model_name(LR2_params['model_name'], LR2_params['pretrain'])
    models_to_plot.append(LR2_params['model_name'])
    LogisticRegression.run_model(LR2_params)
    """

    # DecisionTree w/ active learning
    # Category 3
    DecisionTree1_params = {**common_params, **decisiontree_params}
    DecisionTree1_params['model_name'] = define_non_meta_model_name(
        DecisionTree1_params['model_name'],
        DecisionTree1_params['active_learning'],
        DecisionTree1_params['with_historical_data'],
        DecisionTree1_params['with_k']
    )
    models_to_plot.append(DecisionTree1_params['model_name'])
    DecisionTree.run_model(DecisionTree1_params)

    # Category 4.1
    DecisionTree2_params = {**common_params, **decisiontree_params}
    DecisionTree2_params['with_k'] = True
    DecisionTree2_params['model_name'] = define_non_meta_model_name(
        DecisionTree2_params['model_name'],
        DecisionTree2_params['active_learning'],
        DecisionTree2_params['with_historical_data'],
        DecisionTree2_params['with_k']
    )
    models_to_plot.append(DecisionTree2_params['model_name'])
    DecisionTree.run_model(DecisionTree2_params)

    # Category 4.2
    DecisionTree3_params = {**common_params, **decisiontree_params}
    DecisionTree3_params['with_historical_data'] = False
    DecisionTree3_params['with_k'] = True
    DecisionTree3_params['model_name'] = define_non_meta_model_name(
        DecisionTree3_params['model_name'],
        DecisionTree3_params['active_learning'],
        DecisionTree3_params['with_historical_data'],
        DecisionTree3_params['with_k']
    )
    models_to_plot.append(DecisionTree3_params['model_name'])
    DecisionTree.run_model(DecisionTree3_params)

    # Category 5.1
    DecisionTree4_params = {**common_params, **decisiontree_params}
    DecisionTree4_params['active_learning'] = True
    DecisionTree4_params['with_k'] = True
    DecisionTree4_params['model_name'] = define_non_meta_model_name(
        DecisionTree4_params['model_name'],
        DecisionTree4_params['active_learning'],
        DecisionTree4_params['with_historical_data'],
        DecisionTree4_params['with_k']
    )
    models_to_plot.append(DecisionTree4_params['model_name'])
    DecisionTree.run_model(DecisionTree4_params)

    # Category 5.2
    DecisionTree5_params = {**common_params, **decisiontree_params}
    DecisionTree5_params['active_learning'] = True
    DecisionTree5_params['with_historical_data'] = False
    DecisionTree5_params['with_k'] = True
    DecisionTree5_params['model_name'] = define_non_meta_model_name(
        DecisionTree5_params['model_name'],
        DecisionTree5_params['active_learning'],
        DecisionTree5_params['with_historical_data'],
        DecisionTree5_params['with_k']
    )
    models_to_plot.append(DecisionTree5_params['model_name'])
    DecisionTree.run_model(DecisionTree5_params)
    '''
    
    cv_stats = read_pickle(common_params['stats_path'])
    models_to_plot = list(cv_stats.keys())
    amines = cv_stats[models_to_plot[0]]['amine']
    # print(cv_stats.keys())
    # print(amines)

    # Plotting portion
    # Plot the models based on categories
    cat_3 = [model for model in models_to_plot if 'category_3' in model]
    cat_4 = [model for model in models_to_plot if 'category_4' in model]
    cat_5 = [model for model in models_to_plot if 'category_5' in model]

    all_cats = {
        'category_3': cat_3,
        'category_4': cat_4,
        'category_5': cat_5,
    }

    for cat in all_cats:
        # Identify category specific folder
        graph_folder = './results/{}'.format(cat)

        # Check (and create) designated folder
        if not os.path.exists(graph_folder):
            os.makedirs(graph_folder)
            print(f'No folder for graphs of {cat} models found')
            print('Make folder to store results at')
        else:
            print('Found existing folder. Graphs will be stored at')
        print(graph_folder)

        # Load all models to plot under the category
        models_to_plot = all_cats[cat]

        # Plotting individual graphs for each task-specific model
        for i, amine in enumerate(amines):
            graph_dst = '{0:s}/cv_metrics_{1:s}.png'.format(graph_folder, amine)
            plot_metrics_graph(96, cv_stats, graph_dst, amine=amine, amine_index=i, models=models_to_plot)

        # Plotting avg graphs for all models
        avg_stats = find_avg_metrics(cv_stats)
        rand_model = list(avg_stats.keys())[0]
        num_examples = len(avg_stats[rand_model]['accuracies'])
        graph_dst = '{0:s}/average_metrics_{1:s}.png'.format(graph_folder, cat)
        plot_metrics_graph(num_examples, avg_stats, graph_dst, models=models_to_plot)