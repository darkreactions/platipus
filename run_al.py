import os

from models.non_meta import (RandomForest, KNN, SVM, DecisionTree,
                             LogisticRegression, GradientBoosting)
from utils.plot import plot_all_graphs
from utils import read_pickle, run_non_meta_model
from model_params import (common_params, knn_params, svm_params,
                          randomforest_params, logisticregression_params,
                          decisiontree_params, gradientboosting_params)

if True:
    # Set up the results directory
    results_folder = './results'

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print('No folder for results storage found')
        print('Make folder to store results at')
    else:
        print('Found existing folder. All results will be stored at')
    print(results_folder)

    # Append the data to cv_stats or overwrite the current results
    overwrite = common_params['cv_stats_overwrite']
    cv_stats_dst = common_params['stats_path']
    if os.path.exists(cv_stats_dst) and overwrite:
        print('Overwriting the current cv_stats.pkl')
        os.remove(cv_stats_dst)

    # Listing the categories of experiments we are running
    categories = ['H', 'Hkx',
                  'kx', 'ALHk', 'ALk']

    # Meta-models
    # PLATIPUS
    # platipus_train_params = {**common_params, **meta_params, **meta_train}
    # params = platipus.initialize(["PLATIPUS"], platipus_train_params)
    # platipus.main(params)

    # platipus_test_params = {**common_params, **meta_params, **meta_test}
    # params = platipus.initialize(["PLATIPUS"], platipus_test_params)
    # platipus.main(params)

    # Non-meta models
    # KNN
    base_model = KNN
    model_params = knn_params
    for category in categories:
        run_non_meta_model(
            base_model,
            common_params,
            model_params,
            category
        )

    # SVM
    """
    base_model = SVM
    model_params = svm_params
    for category in categories:
        if '4_ii' not in category and '5_ii' not in category:
            # Use regular random drawn datasets for categories
            # that have sufficient successful experiments for training
            run_non_meta_model(
                base_model,
                common_params,
                model_params,
                category
            )
    """
    # else:
    # Use random drawn datasets with at least one success for
    # categories that few sufficient successful experiments for training
    #    run_non_meta_model(
    #        base_model,
    #        common_params,
    #        model_params,
    #        category,
    #        success=True
    #    )

    # DecisionTree
    base_model = DecisionTree
    model_params = decisiontree_params
    for category in categories:
        run_non_meta_model(
            base_model,
            common_params,
            model_params,
            category
        )

    # Random Forest
    base_model = RandomForest
    model_params = randomforest_params
    for category in categories:
        run_non_meta_model(
            base_model,
            common_params,
            model_params,
            category
        )

    # logistic Regression
    base_model = LogisticRegression
    model_params = logisticregression_params
    for category in categories:
        if '4_ii' not in category and '5_ii' not in category:
            # Use regular random drawn datasets for categories
            # that have sufficient successful experiments for training
            run_non_meta_model(
                base_model,
                common_params,
                model_params,
                category
            )
        else:
            # Use random drawn datasets with at least one success for
            # categories that few sufficient successful experiments for training
            run_non_meta_model(
                base_model,
                common_params,
                model_params,
                category,
                success=True
            )

    # Gradient Boosting
    base_model = GradientBoosting
    model_params = gradientboosting_params
    for category in categories:
        if '4_ii' not in category and '5_ii' not in category:
            # Use regular random drawn datasets for categories
            # that have sufficient successful experiments for training
            run_non_meta_model(
                base_model,
                common_params,
                model_params,
                category
            )
        else:
            # Use random drawn datasets with at least one success for
            # categories that few sufficient successful experiments for training
            run_non_meta_model(
                base_model,
                common_params,
                model_params,
                category,
                success=True
            )

if __name__ == '__main__':
    # Use cv_stats.pkl to plot all graphs
    cv_stats = read_pickle(common_params['stats_path'])
    plot_all_graphs(cv_stats)
