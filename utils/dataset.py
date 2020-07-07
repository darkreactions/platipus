import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# Set up various strings corresponding to headers
distribution_header = '_raw_modelname'
amine_header = '_rxn_organic-inchikey'
score_header = '_out_crystalscore'
name_header = 'name'
to_exclude = [score_header, amine_header, name_header, distribution_header]
path = './data/0057.perovskitedata_DRPFeatures_2020-07-02.csv'

# Successful reaction is defined as having a crystal score of...
SUCCESS = 4


def import_full_dataset(k_shot, meta_batch_size, num_batches, verbose=False,
                        cross_validation=True, meta=True):
    wrong_amine = 'UMDDLGMCNFAZDX-UHFFFAOYSA-O'
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
                     'KFXBDBPOGBBVMC-UHFFFAOYSA-N',
                     'HBPSMMXRESDUSG-UHFFFAOYSA-N',
                     'NXRUEVJQMBGVAT-UHFFFAOYSA-N',
                     'CALQKRVFTWDYDG-UHFFFAOYSA-N',
                     'LLWRXQXPJMPHLR-UHFFFAOYSA-N',
                     'BAMDIFIROXTEEM-UHFFFAOYSA-N',
                     'XZUCBFLUEBDNSJ-UHFFFAOYSA-N']

    hold_out_amines = ['CALQKRVFTWDYDG-UHFFFAOYSA-N',
                       'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                       'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                       'JMXLWMIFDJCGBV-UHFFFAOYSA-N']
    df, amines = import_chemdata(verbose, viable_amines)

    if cross_validation:
        return cross_validation_data(df, amines, hold_out_amines, k_shot,
                                     meta_batch_size, num_batches,
                                     verbose, meta)
    else:
        return hold_out_data(df, amines, hold_out_amines, k_shot,
                             meta_batch_size, num_batches, verbose, meta)


def import_test_dataset(k_shot, meta_batch_size, num_batches, verbose=False,
                        cross_validation=True, meta=True):
    """
    meta_batch_size = 10
    k_shot = 20
    num_batches = 10
    """
    viable_amines = ['ZEVRFFCPALTVDN-UHFFFAOYSA-N',
                     'KFQARYBEAKAXIC-UHFFFAOYSA-N',
                     'NLJDBTZLVTWXRG-UHFFFAOYSA-N']
    hold_out_amines = []
    df, amines = import_chemdata(verbose, viable_amines)

    if cross_validation:
        return cross_validation_data(df, amines, hold_out_amines,
                                     k_shot, meta_batch_size,
                                     num_batches, verbose, meta)
    else:
        return hold_out_data(df, amines, hold_out_amines, k_shot,
                             meta_batch_size, num_batches, verbose, meta)


def import_chemdata(verbose, viable_amines):
    """ Import csv and set up dataframes

    Args:
        verbose:    Boolean value to print statements
        viable_amines: List of viable amines

    return:
        df:                 dataframe that includes only uniformly
                            sampled and viable amines
        amines:             List of all unique amines included in df


    """
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

    return df, amines


def cross_validation_data(df, amines, hold_out_amines, k_shot,
                          meta_batch_size, num_batches, verbose, meta):
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
        if meta:
            batches = []
            for _ in range(num_batches):
                # t for train, v for validate (but validate is outer loop,
                # trying to be consistent with the PLATIPUS code)
                batch = generate_batch(df, meta_batch_size,
                                       available_amines, to_exclude, k_shot)
                batches.append(batch)

            amine_left_out_batches[amine] = batches
        else:
            X = df[df[amine_header] != amine]
            y = X[score_header].values

            # Drop these columns from the dataset
            X = X.drop(to_exclude, axis=1).values

            # Standardize features since they are not yet standardized in the dataset
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            amine_left_out_batches[amine] = (X, y)

        # print("hey this is {}".format(batches))

        # Now set up the cross validation data
        X = df[df[amine_header] == amine]
        y = X[score_header].values
        X = X.drop(to_exclude, axis=1).values

        if meta:
            cross_valid = generate_valid_test_batch(X, y, k_shot)
        else:
            cross_valid = (X, y)

        amine_cross_validate_samples[amine] = cross_valid

    print('Generating testing batches for training')
    amine_test_samples = load_test_samples(
        hold_out_amines, df, to_exclude, k_shot, amine_header, score_header)

    if verbose:
        print('Number of features to train on is',
              len(df.columns) - len(to_exclude))

    return amine_left_out_batches, amine_cross_validate_samples, amine_test_samples, counts


def hold_out_data(df, amines, hold_out_amines, k_shot, meta_batch_size, num_batches, verbose, meta):
    """
    TODO: Implement non-meta branch for hold_out_amine!
    """

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
    amine_test_samples = load_test_samples(
        hold_out_amines, df, to_exclude, k_shot, amine_header, score_header)

    if verbose:
        print('Number of features to train on is',
              len(df.columns) - len(to_exclude))

    return batches, amine_test_samples, counts


def generate_batch(df, meta_batch_size, available_amines, to_exclude,
                   k_shot, amine_header='_rxn_organic-inchikey',
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


def find_index(selected_experiements, all_experiments):
    qry_index = []
    for experiment in selected_experiements:
        i = 0
        found = False
        while i in range(len(all_experiments)) and (found == False):
            if list(experiment) == list(all_experiments[i]):
                found = True
                qry_index.append(i)
            else:
                i += 1
    return qry_index


def process_dataset(train_size=10, active_learning_iter=10, verbose=True, cross_validation=True, full=True,
                    active_learning=True, w_hx=True, w_k=True):
    """TODO: DOCUMENTATION and COMMENTS"""

    # Initialize data dict dictionary
    data_dict = defaultdict(dict)
    dataset_sizes = ['full', 'test']
    al_options = ['w/_AL', 'w/o_AL']
    historical_options = ['w/_hx', 'w/o_hx']
    k_options = ['w/_k', 'w/o_k']
    x_options = ['w/_x', 'w/o_x']
    data_types = ['x_t', 'y_t', 'x_v', 'y_v', 'all_data', 'all_labels']

    for size in dataset_sizes:
        data_dict[size] = defaultdict(dict)
        for al in al_options:
            data_dict[size][al] = defaultdict(dict)
            for hx_op in historical_options:
                data_dict[size][al][hx_op] = defaultdict(dict)
                for k_op in k_options:
                    data_dict[size][al][hx_op][k_op] = defaultdict(dict)
                    for x_op in x_options:
                        data_dict[size][al][hx_op][k_op][x_op] = defaultdict(dict)
                        for dt in data_types:
                            data_dict[size][al][hx_op][k_op][x_op][dt] = defaultdict(dict)

    # Indicate path of the pkl file
    # If running the file inside this folder
    # change ./ to ../ for this path and the path of the csv above
    data_dict_path = './data/non_meta_data.pkl'

    if not os.path.exists(data_dict_path):
        print('No data used for non-meta models found')
        print('Generating data dictionary across models...')

        full_training_op1, full_validation_op1, full_testing_op1, full_counts_op1 = import_full_dataset(
            train_size,
            meta_batch_size=25,
            num_batches=250,
            verbose=verbose,
            cross_validation=cross_validation,
            meta=False
        )
        full_training_op2, full_validation_op2, full_testing_op2, full_counts_op2 = import_full_dataset(
            train_size,
            meta_batch_size=25,
            num_batches=250,
            verbose=verbose,
            cross_validation=cross_validation,
            meta=True
        )
        test_training_op1, test_validation_op1, test_testing_op1, test_counts_op1 = import_test_dataset(
            train_size,
            meta_batch_size=25,
            num_batches=250,
            verbose=verbose,
            cross_validation=cross_validation,
            meta=False
        )
        test_training_op2, test_validation_op2, test_testing_op2, test_counts_op2 = import_test_dataset(
            train_size,
            meta_batch_size=25,
            num_batches=250,
            verbose=verbose,
            cross_validation=cross_validation,
            meta=True
        )

        full_amines = list(full_training_op1.keys())
        test_amines = list(test_training_op1.keys())

        for amine in full_amines:
            # Category 3:
            # Full dataset under option 1 W/O ACTIVE LEARNING
            x_t, y_t = full_training_op1[amine][0], full_training_op1[amine][1]
            x_v, y_v = full_validation_op1[amine][0], full_validation_op1[amine][1]
            all_data, all_labels = full_validation_op1[amine][0], full_validation_op1[amine][1]

            # Load into dictionary
            data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['x_t'][amine] = x_t
            data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['y_t'][amine] = y_t
            data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['x_v'][amine] = x_v
            data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['y_v'][amine] = y_v
            data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['all_data'][amine] = all_data
            data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['all_labels'][amine] = all_labels

            # Category 4(i):
            # For non-AL models only
            # Full dataset under option 1 W/O ACTIVE LEARNING
            x_t, y_t = full_training_op1[amine][0], full_training_op1[amine][1]
            x_v, y_v = full_validation_op1[amine][0], full_validation_op1[amine][1]
            all_data, all_labels = full_validation_op1[amine][0], full_validation_op1[amine][1]

            # Select x many more for training
            # qry_k = np.random.choice(x_v.shape[0], size=train_size, replace=False)
            # qry_x = np.random.choice(x_v.shape[0], size=active_learning_iter, replace=False)
            qry_k = np.random.choice(x_v.shape[0], size=train_size, replace=False)

            # TODO: Rename this variable
            # This is the selected k
            k_x = x_v[qry_k]
            k_y = y_v[qry_k]

            # Update training and validation set with the selection
            x_t = np.append(x_t, k_x).reshape(-1, x_t.shape[1])
            y_t = np.append(y_t, k_y)
            x_v = np.delete(x_v, qry_k, axis=0)
            y_v = np.delete(y_v, qry_k)

            qry_x = np.random.choice(x_v.shape[0], size=active_learning_iter, replace=False)

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
            data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x']['x_t'][amine] = x_t
            data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x']['y_t'][amine] = y_t
            data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x']['x_v'][amine] = x_v
            data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x']['y_v'][amine] = y_v
            data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x']['all_data'][amine] = all_data
            data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x']['all_labels'][amine] = all_labels

            # Category 4(ii):
            # For non-MAML models only
            # Full dataset under option 2 W/O ACTIVE LEARNING
            x_t, y_t = full_validation_op2[amine][0], full_validation_op2[amine][1]
            x_v, y_v = full_validation_op2[amine][2], full_validation_op2[amine][3]
            all_data, all_labels = np.concatenate(
                (x_t, x_v)), np.concatenate((y_t, y_v))

            x_t = np.concatenate((k_x, x_qry))
            y_t = np.concatenate((k_y, y_qry))
            qry = find_index(x_t, all_data)
            x_v = np.delete(all_data, qry, axis=0)
            y_v = np.delete(all_labels, qry)

            # Update training and validation set with the selection
            # x_t = np.append(x_t, x_v[qry]).reshape(-1, x_t.shape[1])
            # y_t = np.append(y_t, y_v[qry])
            # x_v = np.delete(x_v, qry, axis=0)
            # y_v = np.delete(y_v, qry)

            # Load into dictionary
            data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['x_t'][amine] = x_t
            data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['y_t'][amine] = y_t
            data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['x_v'][amine] = x_v
            data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['y_v'][amine] = y_v
            data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['all_data'][amine] = all_data
            data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['all_labels'][amine] = all_labels

            # Category 5(a):
            # For non-PLATIPUS models only
            # Full dataset under option 1 W/ ACTIVE LEARNING
            x_t, y_t = full_training_op1[amine][0], full_training_op1[amine][1]
            x_v, y_v = full_validation_op1[amine][0], full_validation_op1[amine][1]
            all_data, all_labels = full_validation_op1[amine][0], full_validation_op1[amine][1]

            # Select k many more for training before active learning loop
            # qry = np.random.choice(x_v.shape[0], size=train_size, replace=False)
            qry = find_index(k_x, x_v)

            # Update training and validation set with the selection
            x_t = np.append(x_t, x_v[qry]).reshape(-1, x_t.shape[1])
            y_t = np.append(y_t, y_v[qry])
            x_v = np.delete(x_v, qry, axis=0)
            y_v = np.delete(y_v, qry)

            # Load into dictionary
            data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x']['x_t'][amine] = x_t
            data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x']['y_t'][amine] = y_t
            data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x']['x_v'][amine] = x_v
            data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x']['y_v'][amine] = y_v
            data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x']['all_data'][amine] = all_data
            data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x']['all_labels'][amine] = all_labels

            # Category 5(b):
            # Full dataset under option 2 W/ ACTIVE LEARNING
            x_t, y_t = full_validation_op2[amine][0], full_validation_op2[amine][1]
            x_v, y_v = full_validation_op2[amine][2], full_validation_op2[amine][3]
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
            data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['x_t'][amine] = x_t
            data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['y_t'][amine] = y_t
            data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['x_v'][amine] = x_v
            data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['y_v'][amine] = y_v
            data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['all_data'][amine] = all_data
            data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['all_labels'][amine] = all_labels

        for amine in test_amines:
            # Category 3:
            # Test dataset under option 1 W/O ACTIVE LEARNING
            x_t, y_t = test_training_op1[amine][0], test_training_op1[amine][1]
            x_v, y_v = test_validation_op1[amine][0], test_validation_op1[amine][1]
            all_data, all_labels = test_validation_op1[amine][0], test_validation_op1[amine][1]

            # Load into dictionary
            data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['x_t'][amine] = x_t
            data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['y_t'][amine] = y_t
            data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['x_v'][amine] = x_v
            data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['y_v'][amine] = y_v
            data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['all_data'][amine] = all_data
            data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x']['all_labels'][amine] = all_labels

            # Category 4(i):
            # For non-AL models only
            # test dataset under option 1 W/O ACTIVE LEARNING
            x_t, y_t = test_training_op1[amine][0], test_training_op1[amine][1]
            x_v, y_v = test_validation_op1[amine][0], test_validation_op1[amine][1]
            all_data, all_labels = test_validation_op1[amine][0], test_validation_op1[amine][1]

            # Select x many more for training
            # qry_k = np.random.choice(x_v.shape[0], size=train_size, replace=False)
            # qry_x = np.random.choice(x_v.shape[0], size=active_learning_iter, replace=False)
            qry_k = np.random.choice(x_v.shape[0], size=train_size, replace=False)

            # TODO: Rename this variable
            # This is the selected k
            k_x = x_v[qry_k]
            k_y = y_v[qry_k]

            # Update training and validation set w/ the selection
            x_t = np.append(x_t, k_x).reshape(-1, x_t.shape[1])
            y_t = np.append(y_t, k_y)
            x_v = np.delete(x_v, qry_k, axis=0)
            y_v = np.delete(y_v, qry_k)

            qry_x = np.random.choice(x_v.shape[0], size=active_learning_iter, replace=False)

            # TODO: Rename this variable
            # This is the selected x
            x_qry = x_v[qry_x]
            y_qry = y_v[qry_x]

            # Update training and validation set w/ the selection
            x_t = np.append(x_t, x_qry).reshape(-1, x_t.shape[1])
            y_t = np.append(y_t, y_qry)
            x_v = np.delete(x_v, qry_x, axis=0)
            y_v = np.delete(y_v, qry_x)

            # Load into dictionary
            data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x']['x_t'][amine] = x_t
            data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x']['y_t'][amine] = y_t
            data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x']['x_v'][amine] = x_v
            data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x']['y_v'][amine] = y_v
            data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x']['all_data'][amine] = all_data
            data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x']['all_labels'][amine] = all_labels

            # Category 4(ii):
            # For non-MAML models only
            # test dataset under option 2 W/O ACTIVE LEARNING
            x_t, y_t = test_validation_op2[amine][0], test_validation_op2[amine][1]
            x_v, y_v = test_validation_op2[amine][2], test_validation_op2[amine][3]
            all_data, all_labels = np.concatenate(
                (x_t, x_v)), np.concatenate((y_t, y_v))

            x_t = np.concatenate((k_x, x_qry))
            y_t = np.concatenate((k_y, y_qry))
            qry = find_index(x_t, all_data)
            x_v = np.delete(all_data, qry, axis=0)
            y_v = np.delete(all_labels, qry)

            # Update training and validation set with the selection
            # x_t = np.append(x_t, x_v[qry]).reshape(-1, x_t.shape[1])
            # y_t = np.append(y_t, y_v[qry])
            # x_v = np.delete(x_v, qry, axis=0)
            # y_v = np.delete(y_v, qry)

            # Load into dictionary
            data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['x_t'][amine] = x_t
            data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['y_t'][amine] = y_t
            data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['x_v'][amine] = x_v
            data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['y_v'][amine] = y_v
            data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['all_data'][amine] = all_data
            data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x']['all_labels'][amine] = all_labels

            # Category 5(a):
            # For non-PLATIPUS models only
            # test dataset under option 1 W/ ACTIVE LEARNING
            x_t, y_t = test_training_op1[amine][0], test_training_op1[amine][1]
            x_v, y_v = test_validation_op1[amine][0], test_validation_op1[amine][1]
            all_data, all_labels = test_validation_op1[amine][0], test_validation_op1[amine][1]

            # Select k many more for training before active learning loop
            # qry = np.random.choice(x_v.shape[0], size=train_size, replace=False)
            qry = find_index(k_x, x_v)

            # Update training and validation set with the selection
            x_t = np.append(x_t, x_v[qry]).reshape(-1, x_t.shape[1])
            y_t = np.append(y_t, y_v[qry])
            x_v = np.delete(x_v, qry, axis=0)
            y_v = np.delete(y_v, qry)

            # Load into dictionary
            data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x']['x_t'][amine] = x_t
            data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x']['y_t'][amine] = y_t
            data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x']['x_v'][amine] = x_v
            data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x']['y_v'][amine] = y_v
            data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x']['all_data'][amine] = all_data
            data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x']['all_labels'][amine] = all_labels

            # Category 5(b):
            # test dataset under option 2 W/ ACTIVE LEARNING
            x_t, y_t = test_validation_op2[amine][0], test_validation_op2[amine][1]
            x_v, y_v = test_validation_op2[amine][2], test_validation_op2[amine][3]
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
            data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['x_t'][amine] = x_t
            data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['y_t'][amine] = y_t
            data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['x_v'][amine] = x_v
            data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['y_v'][amine] = y_v
            data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['all_data'][amine] = all_data
            data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x']['all_labels'][amine] = all_labels

        with open(data_dict_path, "wb") as f:
            pickle.dump(data_dict, f)
    else:
        print('Found existing data file. Loading data to model...')
        with open(data_dict_path, "rb") as f:
            data_dict = pickle.load(f)

    if full:
        if not active_learning:
            if w_hx:
                if not w_k:
                    print('Training category 3 models.')
                    x_t, y_t, x_v, y_v, all_data, all_labels = data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'x_t'], \
                                                               data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'y_t'], \
                                                               data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'x_v'], \
                                                               data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'y_v'], \
                                                               data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'all_data'], \
                                                               data_dict['full']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'all_labels']
                else:
                    print('Training category 4(i) models.')
                    x_t, y_t, x_v, y_v, all_data, all_labels = data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'x_t'], \
                                                               data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'y_t'], \
                                                               data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'x_v'], \
                                                               data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'y_v'], \
                                                               data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'all_data'], \
                                                               data_dict['full']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'all_labels']
            else:
                print('Training category 4(ii) models.')
                x_t, y_t, x_v, y_v, all_data, all_labels = data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'x_t'], \
                                                           data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'y_t'], \
                                                           data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'x_v'], \
                                                           data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'y_v'], \
                                                           data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'all_data'], \
                                                           data_dict['full']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'all_labels']
        else:
            if w_hx:
                print('Training category 5(i) models.')
                x_t, y_t, x_v, y_v, all_data, all_labels = data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'x_t'], \
                                                           data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'y_t'], \
                                                           data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'x_v'], \
                                                           data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'y_v'], \
                                                           data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'all_data'], \
                                                           data_dict['full']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'all_labels']
            else:
                print('Training category 5(ii) models.')
                x_t, y_t, x_v, y_v, all_data, all_labels = data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'x_t'], \
                                                           data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'y_t'], \
                                                           data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'x_v'], \
                                                           data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'y_v'], \
                                                           data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'all_data'], \
                                                           data_dict['full']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'all_labels']
    else:
        if not active_learning:
            if w_hx:
                if not w_k:
                    print('Training category 3 models.')
                    x_t, y_t, x_v, y_v, all_data, all_labels = data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'x_t'], \
                                                               data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'y_t'], \
                                                               data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'x_v'], \
                                                               data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'y_v'], \
                                                               data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'all_data'], \
                                                               data_dict['test']['w/o_AL']['w/_hx']['w/o_k']['w/o_x'][
                                                                   'all_labels']
                else:
                    print('Training category 4(i) models.')
                    x_t, y_t, x_v, y_v, all_data, all_labels = data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'x_t'], \
                                                               data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'y_t'], \
                                                               data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'x_v'], \
                                                               data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'y_v'], \
                                                               data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'all_data'], \
                                                               data_dict['test']['w/o_AL']['w/_hx']['w/_k']['w/_x'][
                                                                   'all_labels']
            else:
                print('Training category 4(ii) models.')
                x_t, y_t, x_v, y_v, all_data, all_labels = data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'x_t'], \
                                                           data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'y_t'], \
                                                           data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'x_v'], \
                                                           data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'y_v'], \
                                                           data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'all_data'], \
                                                           data_dict['test']['w/o_AL']['w/o_hx']['w/_k']['w/_x'][
                                                               'all_labels']
        else:
            if w_hx:
                print('Training category 5(i) models.')
                x_t, y_t, x_v, y_v, all_data, all_labels = data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'x_t'], \
                                                           data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'y_t'], \
                                                           data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'x_v'], \
                                                           data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'y_v'], \
                                                           data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'all_data'], \
                                                           data_dict['test']['w/_AL']['w/_hx']['w/_k']['w/o_x'][
                                                               'all_labels']
            else:
                print('Training category 5(ii) models.')
                x_t, y_t, x_v, y_v, all_data, all_labels = data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'x_t'], \
                                                           data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'y_t'], \
                                                           data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'x_v'], \
                                                           data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'y_v'], \
                                                           data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'all_data'], \
                                                           data_dict['test']['w/_AL']['w/o_hx']['w/_k']['w/o_x'][
                                                               'all_labels']

    amine_list = list(x_t.keys())

    return amine_list, x_t, y_t, x_v, y_v, all_data, all_labels


if __name__ == "__main__":
    meta_batch_size = 10
    k_shot = 20
    num_batches = 10
    amine_left_out_batches, amine_cross_validate_samples, amine_test_samples, counts = import_test_dataset(
        k_shot, meta_batch_size, num_batches, verbose=True, cross_validation=True, meta=False)

    _ = process_dataset()
