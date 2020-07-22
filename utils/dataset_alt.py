import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from utils.dataset import import_full_dataset, import_test_dataset, write_sel_to_csv
from utils.dataset import find_index as find_index_org


def find_index(selected_experiements, all_experiments):
    """
    Checks if exp in selected_experiments exists in all_experiments and returns the indices
    If an experiment is not found index is set to None
    """
    qry_index = []
    for exp in selected_experiements:
        matches = np.argwhere(np.all(all_experiments == exp, axis=1))
        if matches.size:
            qry_index.append(matches.item(0))
        #qry_index.append(matches.item(0) if matches.size else None)
    return qry_index


def load_data_dict(data_dict, setting, amine, data):
    # x_t, y_t, x_v, y_v, all_data, all_labels = data
    data_types = ['x_t', 'y_t', 'x_v', 'y_v', 'all_data', 'all_labels']
    for i, dtype in enumerate(data_types):
        if dtype not in data_dict[setting]:
            data_dict[setting][dtype] = {}
        data_dict[setting][dtype][amine] = data[i]

    return data_dict


def get_data_dict(data_dict, setting):
    data_types = ['x_t', 'y_t', 'x_v', 'y_v', 'all_data', 'all_labels']

    return (data_dict[setting][dtype] for dtype in data_types)


def get_categories(import_function, dataset_type, data_dict,
                   train_size, active_learning_iter,
                   cross_validation, verbose):
    training_op1, validation_op1, testing_op1, counts_op1 = import_function(
        train_size,
        meta_batch_size=25,
        num_batches=250,
        verbose=verbose,
        cross_validation=cross_validation,
        meta=False
    )
    training_op2, validation_op2, testing_op2, counts_op2 = import_function(
        train_size,
        meta_batch_size=25,
        num_batches=250,
        verbose=verbose,
        cross_validation=cross_validation,
        meta=True
    )
    amines = list(training_op1.keys())

    for amine in amines:
        # Category 3:
        # Full dataset under option 1 W/O ACTIVE LEARNING
        x_t, y_t = training_op1[amine][0], training_op1[amine][1]
        x_v, y_v = validation_op1[amine][0], validation_op1[amine][1]
        all_data, all_labels = validation_op1[amine][0], validation_op1[amine][1]

        # Load into dictionary
        data = (x_t, y_t, x_v, y_v, all_data, all_labels)
        setting = (dataset_type, 'w/o_AL', 'w/_hx', 'w/o_k', 'w/o_x')
        data_dict = load_data_dict(data_dict, setting, amine, data)

        # Category 4(i):
        # For non-AL models only
        # Full dataset under option 1 W/O ACTIVE LEARNING

        # Select x many more for training
        # qry_k = np.random.choice(x_v.shape[0], size=train_size, replace=False)
        # qry_x = np.random.choice(x_v.shape[0], size=active_learning_iter, replace=False)
        qry_k = np.random.choice(
            x_v.shape[0], size=train_size, replace=False)

        # TODO: Rename this variable
        # This is the selected k
        k_x = x_v[qry_k]
        k_y = y_v[qry_k]

        # Load the selected k to data_dict for meta-models
        if 'k_x' not in data_dict[dataset_type]:
            data_dict[dataset_type]['k_x'] = {}
        if 'k_y' not in data_dict[dataset_type]:
            data_dict[dataset_type]['k_y'] = {}
        data_dict[dataset_type]['k_x'][amine] = k_x
        data_dict[dataset_type]['k_y'][amine] = k_y

        # Update training and validation set with the selection
        x_t = np.append(x_t, k_x).reshape(-1, x_t.shape[1])
        y_t = np.append(y_t, k_y)
        x_v = np.delete(x_v, qry_k, axis=0)
        y_v = np.delete(y_v, qry_k)

        qry_x = np.random.choice(
            x_v.shape[0], size=active_learning_iter, replace=False)

        # TODO: Rename this variable
        # This is the selected x
        x_qry = x_v[qry_x]
        y_qry = y_v[qry_x]

        # Update training and validation set with the selection
        x_t = np.append(x_t, x_qry).reshape(-1, x_t.shape[1])
        y_t = np.append(y_t, y_qry)
        x_v = np.delete(x_v, qry_x, axis=0)
        y_v = np.delete(y_v, qry_x)

        # Load into dictionary
        data = (x_t, y_t, x_v, y_v, all_data, all_labels)
        setting = (dataset_type, 'w/o_AL', 'w/_hx', 'w/_k', 'w/_x')
        data_dict = load_data_dict(data_dict, setting, amine, data)

        # Category 4(ii):
        # For non-MAML models only
        # Full dataset under option 2 W/O ACTIVE LEARNING
        x_t, y_t = validation_op2[amine][0], validation_op2[amine][1]
        x_v, y_v = validation_op2[amine][2], validation_op2[amine][3]
        all_data, all_labels = np.concatenate(
            (x_t, x_v)), np.concatenate((y_t, y_v))

        x_t = np.concatenate((k_x, x_qry))
        y_t = np.concatenate((k_y, y_qry))
        qry = find_index(x_t, all_data)
        print(qry)
        qry2 = find_index_org(x_t, all_data)
        print(qry2)
        x_v = np.delete(all_data, qry, axis=0)
        y_v = np.delete(all_labels, qry)

        # Update training and validation set with the selection
        # x_t = np.append(x_t, x_v[qry]).reshape(-1, x_t.shape[1])
        # y_t = np.append(y_t, y_v[qry])
        # x_v = np.delete(x_v, qry, axis=0)
        # y_v = np.delete(y_v, qry)

        # Load into dictionary
        data = (x_t, y_t, x_v, y_v, all_data, all_labels)
        setting = (dataset_type, 'w/o_AL', 'w/o_hx', 'w/_k', 'w/_x')
        data_dict = load_data_dict(data_dict, setting, amine, data)

        # Category 5(a):
        # For non-PLATIPUS models only
        # Full dataset under option 1 W/ ACTIVE LEARNING
        x_t, y_t = training_op1[amine][0], training_op1[amine][1]
        x_v, y_v = validation_op1[amine][0], validation_op1[amine][1]
        all_data, all_labels = validation_op1[amine][0], validation_op1[amine][1]

        # Select k many more for training before active learning loop
        # qry = np.random.choice(x_v.shape[0], size=train_size, replace=False)
        qry = find_index(k_x, x_v)

        # Update training and validation set with the selection
        x_t = np.append(x_t, x_v[qry]).reshape(-1, x_t.shape[1])
        y_t = np.append(y_t, y_v[qry])
        x_v = np.delete(x_v, qry, axis=0)
        y_v = np.delete(y_v, qry)

        # Load into dictionary
        data = (x_t, y_t, x_v, y_v, all_data, all_labels)
        setting = (dataset_type, 'w/_AL', 'w/_hx', 'w/_k', 'w/o_x')
        data_dict = load_data_dict(data_dict, setting, amine, data)

        # Category 5(b):
        # Full dataset under option 2 W/ ACTIVE LEARNING
        x_t, y_t = validation_op2[amine][0], validation_op2[amine][1]
        x_v, y_v = validation_op2[amine][2], validation_op2[amine][3]
        all_data, all_labels = np.concatenate(
            (x_t, x_v)), np.concatenate((y_t, y_v))

        # Select k many more for training before active learning loop
        qry = find_index(k_x, all_data)

        # Update training and validation set w/ the selection
        x_t = k_x
        y_t = k_y
        x_v = np.delete(all_data, qry, axis=0)
        y_v = np.delete(all_labels, qry)

        # Load into dictionary
        data = (x_t, y_t, x_v, y_v, all_data, all_labels)
        setting = (dataset_type, 'w/_AL', 'w/o_hx', 'w/_k', 'w/o_x')
        data_dict = load_data_dict(data_dict, setting, amine, data)
    return data_dict


def process_dataset(train_size=10, active_learning_iter=10,
                    verbose=True, cross_validation=True, full=True,
                    active_learning=True, w_hx=True, w_k=True):

    # data_dict = defaultdict(dict)
    data_dict = defaultdict(dict)

    data_dict_path = Path('./data/non_meta_data.pkl')
    if not data_dict_path.exists():
        # Full dataset
        data_dict = get_categories(import_full_dataset, 'full', data_dict,
                                   train_size, active_learning_iter,
                                   cross_validation, verbose)

        # Test dataset
        data_dict = get_categories(import_test_dataset, 'test', data_dict,
                                   train_size, active_learning_iter,
                                   cross_validation, verbose)
        with open(data_dict_path, "wb") as f:
            pickle.dump(data_dict, f)

        # Store the k experiments selected for each amine to two separate csv files
        write_sel_to_csv(data_dict)

    else:
        # Warning this is a simple test to see if the pickle exists. If parameters
        # such as train_size, active_learning_iter change and a pickle file exists,
        # we would be returning wrong results
        print('Found existing data file. Loading data to model...')
        with open(data_dict_path, "rb") as f:
            data_dict = pickle.load(f)

    dataset_size_setting = 'full' if full else 'test'
    active_learning_setting = 'w/_AL' if active_learning else 'w/o_AL'
    w_hx_setting = 'w/_hx' if w_hx else 'w/o_hx'
    w_k_setting = 'w/_k' if w_k else 'w/o_k'

    # For w_x setting
    if not active_learning:
        if w_hx:
            if not w_k:
                print('Training category 3 models.')
                w_x_setting = 'w/o_x'
            else:
                print('Training category 4(i) models.')
                w_x_setting = 'w/_x'
        else:
            print('Training category 4(ii) models.')
            w_x_setting = 'w/_x'
    else:
        if w_hx:
            print('Training category 5(i) models.')
            w_x_setting = 'w/o_x'
        else:
            print('Training category 5(ii) models.')
            w_x_setting = 'w/o_x'

    setting = (dataset_size_setting, active_learning_setting,
               w_hx_setting, w_k_setting, w_x_setting)

    x_t, y_t, x_v, y_v, all_data, all_labels = get_data_dict(
        data_dict, setting)
    amine_list = list(x_t.keys())

    return amine_list, x_t, y_t, x_v, y_v, all_data, all_labels
