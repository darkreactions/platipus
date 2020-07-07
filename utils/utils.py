import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import torch

from matplotlib import pyplot as plt

from utils.dataset import (import_chemdata, cross_validation_data, hold_out_data,
                           import_test_dataset, import_full_dataset)


def write_pickle(path, data):
    """Write pickle file

    Save for reproducibility

    Args:
        path: The path we want to write the pickle file at
        data: The data we want to save in the pickle file
    """
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def read_pickle(path):
    """Read pickle file

    Make sure we don't overwrite our batches if we are validating and testing

    Args:
        path: The path we want to check the batches

    return: Data that we already stored in the pickle file
    """
    path = Path(path)
    data = None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_chem_dataset(k_shot, cross_validation=True, meta_batch_size=32,
                      num_batches=100, verbose=False, test=False):
    """Load in the chemistry data for training

    "I'm limited by the technology of my time."
    Ideally the model would choose from a Uniformly distributed pool of unlabeled reactions
    Then we would run that reaction in the lab and give it back to the model
    The issue is the labs are closed, so we have to restrict the model to the reactions drawn
    from uniform distributions that we have labels for
    The list below is a list of inchi keys for amines that have a reaction drawn from a uniform
    distribution with a successful outcome (no point in amines that have no successful reaction)

    Args:
        k_shot:             An integer. The number of unseen classes in the dataset
        params:             A dictionary. The dictionary that is initialized with parameters.
                            Use the key "cross_validate" in the dictionary to separate
                            loading training data and testing data
        meta_batch_size:    An integer. The batch size for meta learning, default is 32
        num_batches:        An integer. The batch size for training, default is 100
        verbose:            A boolean that gives information about
                            the number of features to train on is

    return:
        amine_left_out_batches:         A dictionary of batches with structure:
                                        key is amine left out,
                                        value has following hierarchy
                                        batches -> x_t, y_t, x_v, y_v -> meta_batch_size number of amines ->
                                        k_shot number of reactions -> each reaction has some number of features
        amine_cross_validate_samples:   A dictionary of batches with structure:
                                        key is amine which the data is for,
                                        value has the following hierarchy
                                        x_s, y_s, x_q, y_q -> k_shot number of reactions ->
                                        each reaction has some number of features
        amine_test_samples:             A dictionary that has the same structure as
                                        amine_cross_validate_samples
        counts:                         A dictionary to record the number of
                                        successful and failed reactions in the format of
                                        {"total": [# of failed reactions, # of successful reactions]}
    """
    if test:
        print('Getting Test dataset')
        return import_test_dataset(k_shot, meta_batch_size, num_batches, verbose=verbose, cross_validation=cross_validation)
    else:
        print('Getting FULL dataset')
        return import_full_dataset(k_shot, meta_batch_size, num_batches, verbose=verbose, cross_validation=cross_validation)


def find_avg_metrics(stats_dict, min_length=None):
    """Calculate the average metrics of several models' performances

    Args:
        stats_dict:         A dictionary representing the performance metrics of the machine learning models.
                                It has the format of {model_name: {metric_name: [[metric_values for each amine]]}}
        min_length:         An integer representing the fewest number of points to start metrics calculations.

    Returns:
        avg_stat:           A dictionary representing the average performance metrics of each model.
                                It has the format of {model_name: {metric_name: [avg_metric_values]}}.
    """

    # Set up default dictionary to store average metrics for each model
    avg_stat = {}
    
    all_models = list(stats_dict.keys())
    """random_model = all_models[0]

    if not min_length:
        min_length = len(stats_dict[random_model]['accuracies'][0])
        for model in all_models:
            length = len(min(stats_dict[model]['accuracies'], key=len))
            if min_length > length:
                min_length = length"""

    # Calculate average metrics by model
    for model in all_models:
        # Pre-fill each model's value with standard metrics dictionary
        avg_stat[model] = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'bcrs': []
        }
        max_length = len(max(stats_dict[model]['accuracies'], key=len))

        for i in range(max_length):

            total = 0
            num_points = 0
            for amine_metric in stats_dict[model]['accuracies']:
                if i < len(amine_metric):
                    total += amine_metric[i]
                    num_points += 1
            # avg_acc = total / len(stats_dict[model]['accuracies'])
            avg_acc = total / num_points
            avg_stat[model]['accuracies'].append(avg_acc)

            total = 0
            num_points = 0
            for amine_metric in stats_dict[model]['precisions']:
                if i < len(amine_metric):
                    total += amine_metric[i]
                    num_points += 1
            # avg_prec = total / len(stats_dict[model]['precisions'])
            avg_prec = total / num_points
            avg_stat[model]['precisions'].append(avg_prec)

            total = 0
            num_points = 0
            for amine_metric in stats_dict[model]['recalls']:
                if i < len(amine_metric):
                    total += amine_metric[i]
                    num_points += 1
            # avg_rec = total / len(stats_dict[model]['recalls'])
            avg_rec = total / num_points
            avg_stat[model]['recalls'].append(avg_rec)

            total = 0
            num_points = 0
            for amine_metric in stats_dict[model]['bcrs']:
                if i < len(amine_metric):
                    total += amine_metric[i]
                    num_points += 1
            # avg_bcr = total / len(stats_dict[model]['bcrs'])
            avg_bcr = total / num_points
            avg_stat[model]['bcrs'].append(avg_bcr)

    return avg_stat


def save_model(model, params, amine=None):
    """This is to save models

    Create specific folders and store the model in the folder

    Args:
        model:  A string that indicates which model we are using
        params: A dictionary of the initialized parameters
        amine:  The specific amine that we want to store models for.
                Default is None

    return: The path for dst_folder
    """
    # Make sure we are creating directory for all models
    dst_folder_root = '.'
    dst_folder = ""
    if amine is not None and amine in params["training_batches"]:
        dst_folder = '{0:s}/{1:s}_few_shot/{2:s}_{3:s}_{4:d}way_{5:d}shot_{6:s}'.format(
            dst_folder_root,
            model,
            model,
            params['datasource'],
            params['num_classes_per_task'],
            params['num_training_samples_per_class'],
            amine
        )
    elif amine is not None and amine in params["validation_batches"]:
        dst_folder = '{0:s}/{1:s}_few_shot/{2:s}_{3:s}_{4:d}way_{5:d}shot_{6:s}'.format(
            dst_folder_root,
            model,
            model,
            params['datasource'],
            params['num_classes_per_task'],
            params['num_training_samples_per_class'],
            amine
        )
        return dst_folder
    else:
        dst_folder = '{0:s}/{1:s}_few_shot/{2:s}_{3:s}_{4:d}way_{5:d}shot'.format(
            dst_folder_root,
            model,
            model,
            params['datasource'],
            params['num_classes_per_task'],
            params['num_training_samples_per_class']
        )
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print('No folder for storage found')
        print(f'Make folder to store meta-parameters at')
    else:
        print(
            'Found existing folder. Meta-parameters will be stored at')
    print(dst_folder)
    return dst_folder


def initialise_dict_of_dict(key_list):
    """Initialize a dictionary within a dictionary

    Helps us create a data structure to store model weights during gradient updates

    Args:
        key_list: A list of keys that will be initialized
        in each of the keys in the outer-most dictionary

    return:
        q: A dictionary that has a dictionary as the index of each key.
        In the format of {"key":{"key":0}}
    """

    q = dict.fromkeys(['mean', 'logSigma'])
    for para in q.keys():
        q[para] = {}
        for key in key_list:
            q[para][key] = 0
    return q


def create_stats_dict(models):
    """Creating the stats dictionary

    Args:
        model: A list of the models that we are creating metrics for

    return: A dictionary with format: {"model":{"metric1":[],"metric2":[], etc}, "model":{"metric1":[], etc}}
    """
    stats_dict = {}
    metrics = ['accuracies', 'confusion_matrices',
               'precisions', 'recalls', 'bcrs']
    for model in models:
        stats_dict[model] = {}
        for key in metrics:
            stats_dict[model][key] = []
    return stats_dict


def create_cv_stats_dict(models):
    """Creates a stats dictionary that stores the performance metrics during the cross-validation stage of all models on
            a specific amine.

    Args:
        models:         A list representing all the models to be evaluated.

    Returns:
        cv_stats_dict:  A dictionary that stores the performance metrics during the cross-validation stage of a specific
                            amine. It has the format of {model_name:{metric_name: [metric_value]}}.
    """
    metrics = {
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'bcrs': [],
        'corrects': [],
        'confusion_matrices': [],
        'prob_pred': []
    }

    cv_stats_dict = {}

    for model in models:
        # Pre-fill each model's value with standard metrics dictionary
        cv_stats_dict.setdefault(model, metrics)

    return cv_stats_dict


def update_cv_stats_dict(cv_stats_dict, model, correct, cm, accuracy, precision, recall, bcr, prob_pred=None, verbose=True):
    """Update the stats dictionary that stores the performance metrics during the cross-validation stage of a specific
            amine
    Args:
        cv_stats_dict:  A dictionary that stores the performance metrics during the cross-validation stage of a specific
                            amine. It has the format of {model_name:{metric_name: [metric_value]}}.
        model:          A string representing the ML model the metrics are evaluating. Should be either 'PLATIPUS' or
                            'MAML'.
        correct:        An torch.Tensor object representing an array-like element-wise comparison between the actual
                            labels and predicted labels.
        cm:             A numpy array representing the confusion matrix given our predicted labels and the actual
                            corresponding labels. It's a 2x2 matrix for the drp_chem model.
        accuracy:       A float representing the accuracy rate of the model: the rate of correctly predicted reactions
                            out of all reactions.
        precision:      A float representing the precision rate of the model: the rate of the number of actually
                            successful reactions out of all the reactions predicted to be successful.
        recall:         A float representing the recall rate of the model: the rate of the number of reactions predicted
                            to be successful out of all the acutal successful reactions.
        bcr:            A float representing the balanced classification rate of the model. It's the average value of
                            recall rate and true negative rate.
        prob_pred:      A torch.Tensor object representing all the probabilities of the current predictions of all data
                            points w/o active learning. Default to None since the model can be MAML.
        verbose:        A string representing whether we want to print the metrics out or not.

    Returns:
        cv_stats_dict:  A dictionary that stores the performance metrics during the cross-validation stage of a specific
                            amine. It has the same format as above, with values updated given the input metrics.
    """
    # Display and update individual performance metric
    cv_stats_dict[model]['corrects'].extend(correct.detach().cpu().numpy())
    cv_stats_dict[model]['accuracies'].append(accuracy)
    cv_stats_dict[model]['confusion_matrices'].append(cm)
    cv_stats_dict[model]['precisions'].append(precision)
    cv_stats_dict[model]['recalls'].append(recall)
    cv_stats_dict[model]['bcrs'].append(bcr)

    if isinstance(prob_pred, torch.Tensor):
        cv_stats_dict[model]['prob_pred'].extend(
            prob_pred.detach().cpu().numpy())

    if verbose:
        print('accuracy for model is', accuracy)
        print(cm)
        print('precision for model is', precision)
        print('recall for model is', recall)
        print('balanced classification rate for model is', bcr)

    return cv_stats_dict


def define_non_meta_model_name(model_name, active_learning, w_hx, w_k):
    """ Function to define the suffix of non-meta model
    Args:
        model_name:         A string representing the base model name of the non-meta model.
        active_learning:    A boolean representing if the model will conduct active learning or not.
        w_hx:               A boolean representing if the model will be trained with historical data or not.
        w_k:                A boolean representing if the model will be trained with k additional experiments of the
                                task-specific experiments or not.
    returns:
        A string representing the model name with proper suffix
    """

    suffix = ''

    if not active_learning:
        if w_hx:
            if not w_k:
                suffix = '_category_3'
            else:
                suffix = '_category_4_i'
        else:
            if w_k:
                suffix = '_category_4_ii'
            else:
                print('Invalid combination of parameters.')
                print("Can't find appropriate category for the model.")
                print("Using default name instead.")
    else:
        if w_hx:
            if w_k:
                suffix = '_category_5_i'
            else:
                print('Invalid combination of parameters.')
                print("Can't find appropriate category for the model.")
                print("Using default name instead.")
        else:
            if w_k:
                suffix = '_category_5_ii'
            else:
                print('Invalid combination of parameters.')
                print("Can't find appropriate category for the model.")
                print("Using default name instead.")

    return model_name + suffix


if __name__ == "__main__":
    params = {}
    params["cross_validate"] = True
    load_chem_dataset(5, params, meta_batch_size=32,
                      num_batches=100, verbose=True)

