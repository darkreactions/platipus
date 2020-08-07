from itertools import product
from multiprocessing import Pool, Manager
import os
import pickle

from models.non_meta import RandomForest, KNN, SVM, DecisionTree, LogisticRegression, GradientBoosting
from utils import run_non_meta_model
from model_params import common_params, knn_params, svm_params, randomforest_params, logisticregression_params, \
    decisiontree_params, gradientboosting_params, meta_train, meta_test


# A function that will take the config and apply it to the model
def run_models(config):
    """A simple run_model function for multi-processing

    Args:
        config:         A tuple in the form of (result_data, base_model_name, model_params_name, category)
    """

    # Unpack the configuration names
    result_data, base_model_name, model_params_name, category = config

    # Find the corresponding model and model-related params
    base_model = globals()[base_model_name]
    model_params = globals()[model_params_name]

    # Run model with the given configuration
    run_non_meta_model(
        base_model,
        common_params,
        model_params,
        category,
        result_dict=result_data
    )


def main():
    """Main driver function to run multi-thread process"""

    # Set up the results directory
    results_folder = './results'

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print('No folder for results storage found')
        print('Make folder to store results at')
    else:
        print('Found existing folder. All results will be stored at')
    print(results_folder)

    # Listing the categories of experiments we are running
    categories = ['category_3', 'category_4_i', 'category_4_ii', 'category_5_i', 'category_5_ii']
    models_and_params = [
        ('KNN', 'knn_params'),
        ('SVM', 'svm_params'),
        ('DecisionTree', 'decisiontree_params'),
        ('RandomForest', 'randomforest_params'),
        ('LogisticRegression', 'logisticregression_params'),
        ('GradientBoosting', 'gradientboosting_params')
    ]

    # The following dictionary is shared among all processes,
    # so each model will generate its own key based on its config
    manager = Manager()
    result_data = manager.dict()

    # List of arguments to run desired models
    configs = [
        (result_data, model_and_params[0], model_and_params[1], category)
        for model_and_params, category in product(models_and_params, categories)
    ]

    # Ideally should be at least c-1 or c-2 where c is the number
    # of logical cores on the machine
    num_processes = 3

    # Submit all jobs by mapping function with provided configs
    with Pool(num_processes) as pool:
        pool.map(run_models, configs)

    if not common_params['fine_tuning']:
        # Append the data to cv_stats or overwrite the current results
        overwrite = common_params['cv_stats_overwrite']
        cv_stats_dst = common_params['stats_path']
        if os.path.exists(cv_stats_dst) and overwrite:
            print('Overwriting the current cv_stats.pkl')
            os.remove(cv_stats_dst)

        with open(cv_stats_dst, "wb") as f:
            pickle.dump(result_data, f)
    else:
        with open('./results/ft_logs.pkl', "wb") as f:
            pickle.dump(result_data, f)


if __name__ == '__main__':
    main()
