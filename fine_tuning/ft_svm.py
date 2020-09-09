import argparse
from collections import defaultdict
import itertools
import os
import pickle

import numpy as np

from models.non_meta.SVM import ActiveSVM
from utils.dataset import process_dataset
from utils.utils import PUK_kernel


def parse_args():
    """Set up the variables for fine tuning SVM.

    Retrieves argument values from the terminal command line to create an argparse.Namespace object for
        initialization.

    Args:
        N/A

    Returns:
        args: Namespace object with the following attributes:
            category:       A string representing the category the model is fine tuning under.
            index:          An integer representing the combination index to try for fine tuning.

    """

    parser = argparse.ArgumentParser(
        description='Setup variables for fine tuning SVM.')
    parser.add_argument('--category', type=str, default='H',
                        help="model's category to fine tune")

    parser.add_argument('--index', type=int, default=0,
                        help="model's combination index to try")

    args = parser.parse_args()

    return args


def grid_search(clf, combinations, path, num_draws, train_size, active_learning_iter, active_learning=True, w_hx=True,
                w_k=True, draw_success=False, model_name=''):
    """Fine tune the model based on average bcr performance to find the best model hyper-parameters.

    Similar to GridSearchCV in scikit-learn package, we try out all the combinations and evaluate performance
        across all amine-specific models under different categories.

    Args:
        clf:                        A class object representing the classifier being fine tuned.
        combinations:               A list of dictionaries representing the possible hyper-parameter values to try out.
        path:                       A string representing the directory path to store the statistics of all combinations
                                        tried during one stage of fine tuning.
        num_draws:                  An integer representing the number of random drawn to create the dataset.
        train_size:                 An integer representing the number of amine-specific experiments used for training.
                                        Corresponds to the k in the category description.
        active_learning_iter:       An integer representing the number of iterations in an active learning loop.
                                        Corresponds to the x in the category description.
        active_learning:            A boolean representing if active learning will be involved in testing or not.
        w_hx:                       A boolean representing if the models are trained with historical data or not.
        w_k:                        A boolean representing if the modes are trained with amine-specific experiments.
        draw_success:               A boolean representing if the models are trained on regular randomly-drawn datasets
                                        or random datasets with at least one success for each amine.
        model_name:                 A string representing the name of the model being fine tuned.

    Returns:
        best_option:                A dictionary representing the hyper-parameters that yields the best performance on
                                        average. The keys may vary for models.
    """

    # Load or initialize dictionary to keep all configurations' performances
    if os.path.exists(path):
        with open(path, 'rb') as f:
            ft_log = pickle.load(f)
    else:
        ft_log = defaultdict(dict)

    if model_name not in ft_log:
        ft_log[model_name] = defaultdict(dict)

    # Load the full dataset under specific categorical option
    dataset = process_dataset(
        num_draw=num_draws,
        train_size=train_size,
        active_learning_iter=active_learning_iter,
        verbose=False,
        cross_validation=True,
        full=True,
        active_learning=active_learning,
        w_hx=w_hx,
        w_k=w_k,
        success=draw_success
    )

    draws = list(dataset.keys())
    amine_list = list(dataset[0]['x_t'].keys())

    if 'Default' not in ft_log[model_name]:
        # Set baseline performance
        base_accuracies = []
        base_precisions = []
        base_recalls = []
        base_bcrs = []

        for amine in amine_list:
            if amine == 'XZUCBFLUEBDNSJ-UHFFFAOYSA-N' and draw_success:
                # Skipping the amine with only 1 successful experiment overall
                # Can't run 4-ii and 5-ii models on this amine
                continue
            else:
                ACLF = clf(amine=amine, verbose=False)

                for set_id in draws:
                    # Unload the randomly drawn dataset values
                    x_t, y_t, x_v, y_v, all_data, all_labels = dataset[set_id]['x_t'], \
                        dataset[set_id]['y_t'], \
                        dataset[set_id]['x_v'], \
                        dataset[set_id]['y_v'], \
                        dataset[set_id]['all_data'], \
                        dataset[set_id]['all_labels']

                    # Load the training and validation set into the model
                    ACLF.load_dataset(
                        set_id,
                        x_t[amine],
                        y_t[amine],
                        x_v[amine],
                        y_v[amine],
                        all_data[amine],
                        all_labels[amine]
                    )

                    # Train the data on the training set
                    ACLF.train(warning=False)

                ACLF.find_inner_avg()

                base_accuracies.append(
                    ACLF.metrics['average']['accuracies'][-1])
                base_precisions.append(
                    ACLF.metrics['average']['precisions'][-1])
                base_recalls.append(ACLF.metrics['average']['recalls'][-1])
                base_bcrs.append(ACLF.metrics['average']['bcrs'][-1])

        # Calculated the average baseline performances
        ft_log[model_name]['Default']['accuracies'] = sum(
            base_accuracies) / len(base_accuracies)
        ft_log[model_name]['Default']['precisions'] = sum(
            base_precisions) / len(base_precisions)
        ft_log[model_name]['Default']['recalls'] = sum(
            base_recalls) / len(base_recalls)
        ft_log[model_name]['Default']['bcrs'] = sum(base_bcrs) / len(base_bcrs)

    # Try out each possible combinations of hyper-parameters
    for option in combinations:
        accuracies = []
        precisions = []
        recalls = []
        bcrs = []

        for amine in amine_list:
            if amine == 'XZUCBFLUEBDNSJ-UHFFFAOYSA-N' and draw_success:
                # Skipping the amine with only 1 successful experiment overall
                # Can't run 4-ii and 5-ii models on this amine
                continue
            else:
                # print("Training and cross validation on {} amine.".format(amine))
                ACLF = clf(amine=amine, config=option, verbose=False)

                for set_id in draws:
                    # Unload the randomly drawn dataset values
                    x_t, y_t, x_v, y_v, all_data, all_labels = dataset[set_id]['x_t'], \
                        dataset[set_id]['y_t'], \
                        dataset[set_id]['x_v'], \
                        dataset[set_id]['y_v'], \
                        dataset[set_id]['all_data'], \
                        dataset[set_id]['all_labels']

                    # Load the training and validation set into the model
                    ACLF.load_dataset(
                        set_id,
                        x_t[amine],
                        y_t[amine],
                        x_v[amine],
                        y_v[amine],
                        all_data[amine],
                        all_labels[amine]
                    )

                    # Train the data on the training set
                    ACLF.train(warning=False)

                ACLF.find_inner_avg()

                accuracies.append(ACLF.metrics['average']['accuracies'][-1])
                precisions.append(ACLF.metrics['average']['precisions'][-1])
                recalls.append(ACLF.metrics['average']['recalls'][-1])
                bcrs.append(ACLF.metrics['average']['bcrs'][-1])

        ft_log[model_name][str(option)]['accuracies'] = sum(
            accuracies) / len(accuracies)
        ft_log[model_name][str(option)]['precisions'] = sum(
            precisions) / len(precisions)
        ft_log[model_name][str(option)]['recalls'] = sum(
            recalls) / len(recalls)
        ft_log[model_name][str(option)]['bcrs'] = sum(bcrs) / len(bcrs)

    # Save the fine tuning performances to pkl if not multi-processing
    with open(path, 'wb') as f:
        pickle.dump(ft_log, f)


def fine_tune(params):
    """Main fine tuning function

    Args:
        params:     A dictionary representing the parameters the specify the model we are fine tuning
    """

    # Create the list of all the combinations of the given hyper-parameters
    combinations = []

    class_weights = [{0: i, 1: 1.0 - i} for i in np.linspace(.1, .9, num=9)]
    class_weights.append('balanced')
    class_weights.append(None)

    ft_params = {
        'kernel': ['poly', 'sigmoid', 'rbf', PUK_kernel],
        'C': [.001, .01, .1, 1, 10, 100],
        'degree': [i for i in range(1, 6)],
        'gamma': ['auto', 'scale'],
        'tol': [.001, .01, .1, 1, 10],
        'decision_function_shape': ['ovo'],
        'break_ties': [True],
        'class_weight': class_weights
    }

    keys, values = zip(*ft_params.items())
    for bundle in itertools.product(*values):
        combinations.append(dict(zip(keys, bundle)))

    # Get rid of the redundant configurations
    unique_combo = []
    for config in combinations:
        if not (config['kernel'] != 'poly' and config['degree'] != 1):
            unique_combo.append(config)

    # Query the configuration given command line input
    # TODO: ugly patch. Not the best way to do this.
    idx = params['index']
    combo = unique_combo[idx: idx+1]

    # Preset the active_learning, w_hx, w_k, and draw_success settings
    # for each category
    cat_settings = {
        'H': [False, True, False, False],
        'Hkx': [False, True, True, False],
        'kx': [False, False, True, True],
        'ALHk': [True, True, True, False],
        'ALk': [True, False, True, True],
    }

    # Set the model and dataset specific variables
    category = params['category']
    settings = cat_settings[category]
    model_name = 'SVM_' + category
    # TODO these should change accordingly if the experimental plans change
    num_draws = 5
    train_size = 10
    active_learning_iter = 10

    # Indicate the path to store the fine tuning performances
    ft_log_path = './results/ft_{}.pkl'.format(model_name)

    # Conduct fine tuning
    grid_search(
        ActiveSVM,
        combo,
        ft_log_path,
        num_draws,
        train_size,
        active_learning_iter,
        active_learning=settings[0],
        w_hx=settings[1],
        w_k=settings[2],
        draw_success=settings[3],
        model_name=model_name
    )


def main():
    """Main driver function"""

    # This converts the args into a dictionary
    SVM_params = vars(parse_args())

    fine_tune(SVM_params)


if __name__ == '__main__':
    main()
