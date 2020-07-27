import argparse
import itertools
import pickle
import os

import numpy as np
from sklearn.metrics import roc_auc_score
import timeout_decorator

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

    parser = argparse.ArgumentParser(description='Setup variables for fine tuningh SVM.')
    parser.add_argument('--category', type=str, default='category_3', help="model's category to fine tune")

    parser.add_argument('--index', type=int, default=0, help="model's combination index to try")

    args = parser.parse_args()

    return args


@timeout_decorator.timeout(90)
def grid_search(clf, combinations, active_learning=True, w_hx=True, w_k=True, info=False):
    """Fine tune the model based on average bcr performance to find the best model hyper-parameters.

    Similar to GridSearchCV in scikit-learn package, we try out all the combinations and evaluate performance
        across all amine-specific models under different categories.

    Args:
        clf:                        A class object representing the classifier being fine tuned.
        combinations:               A list representing the possible hyper-parameter combinations to try out.
        active_learning:            A boolean representing if active learning will be involved in testing or not.
        w_hx:                       A boolean representing if the models are trained with historical data or not.
        w_k:                        A boolean representing if the modes are trained with amine-specific experiments.
        info:                       A boolean. Setting it to True will make the function print out additional
                                        information during the fine-tuning stage.
                                        Default to False.
    Returns:
        best_option:                A dictionary representing the hyper-parameters that yields the best performance on
                                        average. The keys may vary for models.
    """

    # Load the full dataset under specific categorical option
    amine_list, train_data, train_labels, val_data, val_labels, all_data, all_labels = process_dataset(
        train_size=10,
        active_learning_iter=10,
        verbose=False,
        cross_validation=True,
        full=True,
        active_learning=active_learning,
        w_hx=w_hx,
        w_k=w_k
    )

    # Find a benchmark metric for comparison
    ft_pkl_path = './fine_tuning_logs/ft_history.pkl'
    ft_history = {
        'base_accuracy': 0,
        'base_precision': 0,
        'base_recall': 0,
        'base_bcr': 0,
        'base_auc': 0,
        'auc': 0,
        'config': {}
    }
    if not os.path.exists(ft_pkl_path):
        # Set baseline performance for the first run
        base_accuracies = []
        base_precisions = []
        base_recalls = []
        base_bcrs = []
        base_aucs = []
    
        for amine in amine_list:
            ACLF = clf(amine=amine, verbose=False)
    
            # Exact and load the training and validation set into the model
            x_t, y_t = train_data[amine], train_labels[amine]
            x_v, y_v = val_data[amine], val_labels[amine]
            all_task_data, all_task_labels = all_data[amine], all_labels[amine]
            ACLF.load_dataset(x_t, y_t, x_v, y_v, all_task_data, all_task_labels)
    
            ACLF.train(warning=False)
    
            # Calculate AUC
            auc = roc_auc_score(ACLF.all_labels, ACLF.y_preds)
    
            base_accuracies.append(ACLF.metrics['accuracies'][-1])
            base_precisions.append(ACLF.metrics['precisions'][-1])
            base_recalls.append(ACLF.metrics['recalls'][-1])
            base_bcrs.append(ACLF.metrics['bcrs'][-1])
            base_aucs.append(auc)
    
        # Calculated the average baseline performances
        base_avg_accuracy = sum(base_accuracies) / len(base_accuracies)
        base_avg_precision = sum(base_precisions) / len(base_precisions)
        base_avg_recall = sum(base_recalls) / len(base_recalls)
        base_avg_bcr = sum(base_bcrs) / len(base_bcrs)
        base_avg_auc = sum(base_aucs) / len(base_aucs)

        ft_history['base_accuracy'] = base_avg_accuracy
        ft_history['base_precision'] = base_avg_precision
        ft_history['base_recall'] = base_avg_recall
        ft_history['base_bcr'] = base_avg_bcr
        ft_history['base_auc'] = base_avg_auc

        best_metric = base_avg_auc
        best_recall = base_avg_recall
        best_option = {}
    
        if info:
            print(f'Baseline average accuracy is {base_avg_accuracy}')
            print(f'Baseline average precision is {base_avg_precision}')
            print(f'Baseline average recall is {base_avg_recall}')
            print(f'Baseline average bcr is {base_avg_bcr}')
            print(f'Baseline average auc is {base_avg_auc}')
    else:
        # Load the pkl file
        with open(ft_pkl_path, 'rb') as f:
            ft_history = pickle.load(f)
        # Unload all the metrics and config
        base_avg_accuracy = ft_history['base_accuracy']
        base_avg_precision = ft_history['base_precision']
        base_avg_recall = ft_history['base_recall']
        base_avg_bcr = ft_history['base_bcr']
        base_avg_auc = ft_history['base_auc']
        best_metric = ft_history['auc']
        best_recall = ft_history['recall']
        best_option = ft_history['config']

    # Try out each possible combinations of hyper-parameters
    for option in combinations:
        accuracies = []
        precisions = []
        recalls = []
        bcrs = []
        aucs = []

        for amine in amine_list:
            # print("Training and cross validation on {} amine.".format(amine))
            ACLF = clf(amine=amine, config=option, verbose=False)

            # Exact and load the training and validation set into the model
            x_t, y_t = train_data[amine], train_labels[amine]
            x_v, y_v = val_data[amine], val_labels[amine]
            all_task_data, all_task_labels = all_data[amine], all_labels[amine]

            ACLF.load_dataset(x_t, y_t, x_v, y_v, all_task_data, all_task_labels)
            ACLF.train(warning=False)

            # Calculate AUC
            auc = roc_auc_score(ACLF.all_labels, ACLF.y_preds)

            accuracies.append(ACLF.metrics['accuracies'][-1])
            precisions.append(ACLF.metrics['precisions'][-1])
            recalls.append(ACLF.metrics['recalls'][-1])
            bcrs.append(ACLF.metrics['bcrs'][-1])
            aucs.append(auc)

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_bcr = sum(bcrs) / len(bcrs)
        avg_auc = sum(aucs) / len(aucs)

        if best_metric - avg_auc < .01 and avg_recall > best_recall:
            if info:
                print(f'The previous best option is {best_option}')
                print(f'The current best setting is {option}')
                print(f'The fine-tuned average accuracy is {avg_accuracy} vs. the base accuracy {base_avg_accuracy}')
                print(
                    f'The fine-tuned average precision is {avg_precision} vs. the base precision {base_avg_precision}')
                print(f'The fine-tuned average recall rate is {avg_recall} vs. the base recall rate {base_avg_recall}')
                print(f'The fine-tuned average bcr is {avg_bcr} vs. the base bcr {base_avg_bcr}')
                print(f'The fine-tuned average auc is {avg_auc} vs. the base auc {base_avg_auc}')
                print()

            best_metric = avg_auc
            best_recall = avg_recall
            best_option = option

    # Log last best performance and save it to path
    ft_history['auc'] = best_metric
    ft_history['recall'] = best_recall
    ft_history['config'] = best_option
    with open(ft_pkl_path, 'wb') as f:
        pickle.dump(ft_history, f)


def fine_tune(params):
    """TODO DOCUMENTATION"""
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

    unique_combo = []
    for config in combinations:
        if not (config['kernel'] != 'poly' and config['degree'] != 1):
            unique_combo.append(config)

    idx = params['index']
    combo = unique_combo[idx: idx+1]    # TODO: ugly patch. Not the best way doing this.

    cat_settings = {
        'category-3': [False, True, False],
        'category-4-i': [False, True, True],
        'category-4-ii': [False, False, True],
        'category-5-i': [True, True, True],
        'category-5-ii': [True, False, True],
    }

    category = params['category']
    settings = cat_settings[category]

    try:
        _ = grid_search(
            ActiveSVM,
            combo,
            active_learning=settings[0],
            w_hx=settings[1],
            w_k=settings[2],
            info=True
        )
    except TimeoutError:
        print()


def main():
    """Main driver function"""

    # This converts the args into a dictionary
    SVM_params = vars(parse_args())

    fine_tune(SVM_params)


if __name__ == '__main__':
    main()
