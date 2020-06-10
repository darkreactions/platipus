import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import Path
import pickle
import os

from matplotlib import pyplot as plt


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


def load_chem_dataset(k_shot, cross_validation=True, meta_batch_size=32, num_batches=100, verbose=False):
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
    viable_amines = ['ZEVRFFCPALTVDN-UHFFFAOYSA-N',
                     'KFQARYBEAKAXIC-UHFFFAOYSA-N',
                     'NLJDBTZLVTWXRG-UHFFFAOYSA-N',
                     'LCTUISCIGMWMAT-UHFFFAOYSA-N',
                     'JERSPYRKVMAEJY-UHFFFAOYSA-N',
                     'JMXLWMIFDJCGBV-UHFFFAOYSA-N',
                     'VAWHFUNJDMQUSB-UHFFFAOYSA-N',
                     'WGYRINYTHSORGH-UHFFFAOYSA-N',
                     'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                     'VNAAUNTYIONOHR-UHFFFAOYSA-N',
                     'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                     'FJFIJIDZQADKEE-UHFFFAOYSA-N',
                     'XFYICZOIWSBQSK-UHFFFAOYSA-N',
                     'UMDDLGMCNFAZDX-UHFFFAOYSA-O',
                     'HBPSMMXRESDUSG-UHFFFAOYSA-N',
                     'NXRUEVJQMBGVAT-UHFFFAOYSA-N',
                     'CALQKRVFTWDYDG-UHFFFAOYSA-N',
                     'LLWRXQXPJMPHLR-UHFFFAOYSA-N',
                     'BAMDIFIROXTEEM-UHFFFAOYSA-N',
                     'XZUCBFLUEBDNSJ-UHFFFAOYSA-N']

    # Set up various strings corresponding to headers
    distribution_header = '_raw_modelname'
    amine_header = '_rxn_organic-inchikey'
    score_header = '_out_crystalscore'
    name_header = 'name'
    to_exclude = [score_header, amine_header, name_header, distribution_header]
    path = './data/0050.perovskitedata_DRP.csv'

    # Successful reaction is defined as having a crystal score of...
    SUCCESS = 4

    # Get amine and distribution counts for the data
    df = pd.read_csv(path)

    # Set up the 0/1 labels and drop non-uniformly distributed reactions

    df = df[df[distribution_header].str.contains('Uniform')]
    df = df[df[amine_header].isin(viable_amines)]

    if verbose:
        print('There should be 1661 reactions here, shape is', df.shape)
    df[score_header] = [1 if val == SUCCESS else 0 for val in df[score_header].values]
    amines = df[amine_header].unique().tolist()

    # Hold out 4 amines for testing, the other 16 are fair game for the cross validation
    # I basically picked these randomly since I have no idea which inchi key corresponds to what
    hold_out_amines = ['CALQKRVFTWDYDG-UHFFFAOYSA-N',
                       'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                       'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                       'JMXLWMIFDJCGBV-UHFFFAOYSA-N']

    if cross_validation:

        amines = [a for a in amines if a not in hold_out_amines]

        # Used to set up our weighted loss function
        counts = {}
        all_train = df[df[amine_header].isin(amines)]
        print('Number of reactions in training set', all_train.shape[0])
        all_train_success = all_train[all_train[score_header] == 1]
        print('Number of successful reactions in the training set',
              all_train_success.shape[0])

        # [Number of failed reactions, number of successful reactions]
        counts['total'] = [all_train.shape[0] -
                           all_train_success.shape[0], all_train_success.shape[0]]

        amine_left_out_batches = {}
        amine_cross_validate_samples = {}

        for amine in amines:
            # Since we are doing cross validation, create a training set without each amine
            print("Generating batches for amine", amine)
            available_amines = [a for a in amines if a != amine]

            all_train = df[df[amine_header].isin(available_amines)]
            print(
                f'Number of reactions in training set holding out {amine}', all_train.shape[0])
            all_train_success = all_train[all_train[score_header] == 1]
            print(
                f'Number of successful reactions in training set holding out {amine}', all_train_success.shape[0])

            counts[amine] = [all_train.shape[0] -
                             all_train_success.shape[0], all_train_success.shape[0]]
            batches = []
            for _ in range(num_batches):
                # t for train, v for validate (but validate is outer loop, trying to be consistent with the PLATIPUS code)
                batch = generate_batch(df, meta_batch_size,
                                       available_amines, to_exclude, k_shot)
                batches.append(batch)

            amine_left_out_batches[amine] = batches
            # print("hey this is {}".format(batches))

            # Now set up the cross validation data
            X = df[df[amine_header] == amine]
            y = X[score_header].values
            X = X.drop(to_exclude, axis=1).values
            cross_valid = generate_valid_test_batch(X, y, k_shot)

            amine_cross_validate_samples[amine] = cross_valid

        print('Generating testing batches for training')
        amine_test_samples = load_test_samples(hold_out_amines, df, to_exclude, k_shot, amine_header, score_header)

        if verbose:
            print('Number of features to train on is',
                  len(df.columns) - len(to_exclude))

        return amine_left_out_batches, amine_cross_validate_samples, amine_test_samples, counts

    else:
        print('Holding out', hold_out_amines)

        available_amines = [a for a in amines if a not in hold_out_amines]
        # Used to set up our weighted loss function
        counts = {}
        all_train = df[df[amine_header].isin(available_amines)]
        print('Number of reactions in training set', all_train.shape[0])
        all_train_success = all_train[all_train[score_header] == 1]
        print('Number of successful reactions in the training set',
              all_train_success.shape[0])

        counts['total'] = [all_train.shape[0] -
                           all_train_success.shape[0], all_train_success.shape[0]]

        batches = []
        print('Generating training batches')
        for _ in range(num_batches):
            # t for train, v for validate (but validate is outer loop, trying to be consistent with the PLATIPUS code)
            batch = generate_batch(df, meta_batch_size,
                                   available_amines, to_exclude, k_shot)
            batches.append(batch)

        print('Generating testing batches for testing! DO NOT RUN IF YOU SEE THIS LINE!')
        amine_test_samples = load_test_samples(hold_out_amines, df, to_exclude, k_shot, amine_header, score_header)

        if verbose:
            print('Number of features to train on is',
                  len(df.columns) - len(to_exclude))

        return batches, amine_test_samples, counts


def generate_batch(df, meta_batch_size, available_amines, to_exclude, k_shot, amine_header='_rxn_organic-inchikey',
                   score_header='_out_crystalscore'):
    """Generate the batch for training amines

    Args:
        df:                 The data frame of the amines data
        meta_batch_size:    An integer. Batch size for meta learning
        available_amines:   A list. The list of amines that we are generating batches on
        to_exclude:         A list. The columns in the dataset that we need to drop
        k_shot:             An integer. The number of unseen classes in the dataset
        amine_header:       The header of the amine list in the data frame,
                            default = '_rxn_organic-inchikey'
        score_header:       The header of the score header in the data frame,
                            default = '_out_crystalscore'

    return: A list of the batch with
    training and validation features and labels in numpy arrays.
    The format is [[training_feature],[training_label],[validation_feature],[validation_label]]
    """
    x_t, y_t, x_v, y_v = [], [], [], []

    for _ in range(meta_batch_size):
        # Grab the tasks
        X = df[df[amine_header] == np.random.choice(available_amines)]

        y = X[score_header].values

        # Drop these columns from the dataset
        X = X.drop(to_exclude, axis=1).values

        # Standardize features since they are not yet standardized in the dataset
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        spt = np.random.choice(X.shape[0], size=k_shot, replace=False)
        qry = np.random.choice(X.shape[0], size=k_shot, replace=False)

        x_t.append(X[spt])
        y_t.append(y[spt])
        x_v.append(X[qry])
        y_v.append(y[qry])

    return [np.array(x_t), np.array(y_t), np.array(x_v), np.array(y_v)]


def generate_valid_test_batch(X, y, k_shot):
    """Generate the batches for the amine used for cross validation or testing

    Args:
        X:      Dataframe. The features of the chosen amine in the dataset
        y:      Dataframe. The labels of the chosen amine in the dataset
        k_shot: An integer. The number of unseen classes in the dataset

    return: A list of the features and labels for the amine
    """
    spt = np.random.choice(X.shape[0], size=k_shot, replace=False)
    qry = [i for i in range(len(X)) if i not in spt]
    if len(qry) <= 5:
        print("Warning: minimal testing data for meta-learn assessment")

    x_s = X[spt]
    y_s = y[spt]
    x_q = X[qry]
    y_q = y[qry]

    scaler = StandardScaler()
    scaler.fit(x_s)

    x_s = scaler.transform(x_s)
    x_q = scaler.transform(x_q)

    return [x_s, y_s, x_q, y_q]


def find_avg_metrics(stats_dict, min_length):
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
    metrics = {
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'bcrs': []
    }

    avg_stat = {}

    # Calculate average metrics by model
    for model in stats_dict:
        # Pre-fill each model's value with standard metrics dictionary
        avg_stat.setdefault(model, metrics)
        for i in range(min_length):
            # Go by amine, this code is bulky but it's good for debugging
            total = 0
            for acc_list in stats_dict[model]['accuracies']:
                total += acc_list[i]
            avg_acc = total / len(stats_dict[model]['accuracies'])
            avg_stat[model]['accuracies'].append(avg_acc)

            total = 0
            for prec_list in stats_dict[model]['precisions']:
                total += prec_list[i]
            avg_prec = total / len(stats_dict[model]['precisions'])
            avg_stat[model]['precisions'].append(avg_prec)

            total = 0
            for rec_list in stats_dict[model]['recalls']:
                total += rec_list[i]
            avg_recall = total / len(stats_dict[model]['recalls'])
            avg_stat[model]['recalls'].append(avg_recall)

            total = 0
            for bcr_list in stats_dict[model]['balanced_classification_rates']:
                total += bcr_list[i]
            avg_brc = total / len(stats_dict[model]['balanced_classification_rates'])
            avg_stat[model]['bcrs'].append(avg_brc)

    return avg_stat


def plot_metrics_graph(num_examples, stats_dict, dst, amine=None, show=False):
    """Plot metrics graphs for all models in comparison

    The graph will have 4 subplots, which are for: accuracy, precision, recall, and bcr, from left to right,
        top to bottom

    Args:
        num_examples:       A list representing the number of examples we are working with at each point.
        stats_dict:         A dictionary with each model as key and a dictionary of model specific metrics as value.
                                Each metric dictionary has the same keys: 'accuracies', 'precisions', 'recalls', 'bcrs',
                                and their corresponding list of values for each model as dictionary values.
        dst:                A string representing the folder that the graph will be saved in.
        amine:              A string representing the amine that our model metrics are for. Default to be None.
        show:               A boolean representing whether we want to show the graph or not. Default to False to
                                seamlessly run the whole model,

    Returns:
        N/A
    """

    # Set up initial figure for plotting
    fig = plt.figure(figsize=(16, 12))

    # Setting up each sub-graph as axes
    # From left to right, top to bottom: Accuracy, Precision, Recall, BCR
    acc = plt.subplot(2, 2, 1)
    acc.set_ylabel('Accuracy')
    acc.set_title(f'Learning curve for {amine}') if amine else acc.set_title(f'Averaged learning curve')

    prec = plt.subplot(2, 2, 2)
    prec.set_ylabel('Precision')
    prec.set_title(f'Precision curve for {amine}') if amine else prec.set_title(f'Averaged precision curve')

    rec = plt.subplot(2, 2, 3)
    rec.set_ylabel('Recall')
    rec.set_title(f'Recall curve for {amine}') if amine else rec.set_title(f'Averaged recall curve')

    bcr = plt.subplot(2, 2, 4)
    bcr.set_ylabel('Balanced classification rate')
    bcr.set_title(f'BCR curve for {amine}') if amine else bcr.set_title(f'Averaged BCR curve')

    # Plot each model's metrics
    for model in stats_dict:
        acc.plot(num_examples, stats_dict[model]['accuracies'], 'o-', label=model)
        prec.plot(num_examples, stats_dict[model]['precisions'], 'o-', label=model)
        rec.plot(num_examples, stats_dict[model]['recalls'], 'o-', label=model)
        bcr.plot(num_examples, stats_dict[model]['bcrs'], 'o-', label=model)

    # Display subplot legends
    acc.legend()
    prec.legend()
    rec.legend()
    bcr.legend()

    fig.text(0.5, 0.04, "Number of samples given", ha="center", va="center")

    # Set the metrics graph's name and designated folder
    graph_name = 'cv_metrics_{0:s}.png'.format(amine) if amine else 'average_metrics.png'
    graph_dst = '{0:s}/{1:s}'.format(dst, graph_name)

    # Remove duplicate graphs in case we can't directly overwrite the files
    if os.path.isfile(graph_dst):
        os.remove(graph_dst)

    # Save graph in folder
    plt.savefig(graph_dst)
    print(f"Graph {graph_name} saved in folder {dst}")

    if show:
        plt.show()

        
def load_test_samples(hold_out_amines, df, to_exclude, k_shot, amine_header, score_header):
    """This is a function used for loading testing samples specifically

    Args:
        hold_out_amines:    The list of all holdout amines that are used for testing.
                            DO NOT TOUCH!
        df:                 The data frame of the amines data
        to_exclude:         A list. The columns in the dataset that we need to drop
        k_shot:             An integer. The number of unseen classes in the dataset
        amine_header:       The header of the amine list in the data frame.
        score_header:       The header of the score header in the data frame.

    return: A dictionary that contains the test sample amines' batches
    """
    amine_test_samples = {}
    for a in hold_out_amines:
        # grab task
        X = df[df[amine_header] == a]

        y = X[score_header].values
        X = X.drop(to_exclude, axis=1).values
        test_sample = generate_valid_test_batch(X, y, k_shot)

        amine_test_samples[a] = test_sample
    return amine_test_samples

  
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


if __name__ == "__main__":
    params = {}
    params["cross_validate"] = True
    load_chem_dataset(5, params, meta_batch_size=32, num_batches=100, verbose=True)
