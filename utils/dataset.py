import os
from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.dataset_class import Phase3DataSet, Setting

# Set up various strings corresponding to headers
distribution_header = '_raw_modelname'
amine_header = '_rxn_organic-inchikey'
score_header = '_out_crystalscore'
name_header = 'name'
bool_headers = ['_solv_GBL',
                '_solv_DMSO',
                '_solv_DMF',
                '_feat_primaryAmine',
                '_feat_secondaryAmine',
                '_rxn_plateEdgeQ']
to_exclude = [score_header, amine_header, name_header, distribution_header, '_raw_RelativeHumidity']
path = './data/raw/0057.perovskitedata_DRPFeatures_2020-07-02.csv'

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

    """
    hold_out_amines = ['CALQKRVFTWDYDG-UHFFFAOYSA-N',
                       'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                       'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                       'JMXLWMIFDJCGBV-UHFFFAOYSA-N']
    """
    #For phase 3
    hold_out_amines = ['JMXLWMIFDJCGBV-UHFFFAOYSA-N']
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
            X_sel = df[df[amine_header] != amine]
            y = X_sel[score_header].values

            # Drop these columns from the dataset
            # X = X.drop(to_exclude, axis=1).values
            X = X_sel.drop(to_exclude+bool_headers, axis=1).values

            # Standardize features since they are not yet standardized in the dataset
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            X = np.concatenate([X, X_sel[bool_headers].values], axis=1)

            amine_left_out_batches[amine] = (X, y)

        # print("hey this is {}".format(batches))

        # Now set up the cross validation data
        X_sel = df[df[amine_header] == amine]
        y = X_sel[score_header].values
        X = X_sel.drop(to_exclude+bool_headers, axis=1).values

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X = np.concatenate([X, X_sel[bool_headers].values], axis=1)

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
    # Getting scaler to scale batches
    dataset_path = Path('./data/phase3_dataset.pkl')
    phase3dataset = pickle.load(dataset_path.open('rb'))
    scaler = phase3dataset.scaler

    for _ in range(num_batches):
        # t for train, v for validate (but validate is outer loop, trying to be consistent with the PLATIPUS code)
        batch = generate_batch(df, meta_batch_size,
                               available_amines, to_exclude, k_shot, scaler=scaler)
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
                   score_header='_out_crystalscore', scaler=None):
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
        X_sel = df[df[amine_header] == np.random.choice(available_amines)]

        y = X_sel[score_header].values

        # Drop these columns from the dataset
        X = X_sel.drop(to_exclude+bool_headers, axis=1).values
        
        # Standardize features since they are not yet standardized in the dataset
        if scaler == None:
            print('Scaler not found')
            scaler = StandardScaler()
            scaler.fit(X)
        X = scaler.transform(X)
        X = np.concatenate([X, X_sel[bool_headers].values], axis=1)        

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

    """
    scaler = StandardScaler()
    scaler.fit(x_s)

    x_s = scaler.transform(x_s)
    x_q = scaler.transform(x_q)
    """
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
        X_sel = df[df[amine_header] == a]

        y = X_sel[score_header].values
        X = X_sel.drop(to_exclude+bool_headers, axis=1).values
        
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X = np.concatenate([X, X_sel[bool_headers].values], axis=1)        

        test_sample = generate_valid_test_batch(X, y, k_shot)

        amine_test_samples[a] = test_sample
    return amine_test_samples


def find_index(selected_experiements, all_experiments):
    """
    TODO: DOCUMENTATION
    :param selected_experiements:
    :param all_experiments:
    :return:
    """
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


def write_sel_to_csv(data_dict, num_draw):
    """Find the selected k amine-specific experiments and write them to two csv files

    Args:
        data_dict:          A dictionary representing the processed dataset for non-meta model training.
                                See process_dataset() for more details.
        num_draw:           An integer representing the number of random draws done when generating the dataset
    """

    # Load the original chem dataset to find specific experiments
    df = pd.read_csv(path)

    # Make a folder to store all csv files
    csv_folder = './data/selected_k'

    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    # Unpack the selected experiments
    selected_experiments = defaultdict(dict)
    sizes = ['full', 'test']
    rand_options = ['random', 'w/_success']
    set_idx = [i for i in range(num_draw)]

    for size in sizes:
        selected_experiments[size] = defaultdict(dict)
        for opt in rand_options:
            selected_experiments[size][opt] = defaultdict(dict)
            for id in set_idx:
                selected_experiments[size][opt][id] = defaultdict(dict)
                for amine in data_dict[size][opt]['k_x'][id].keys():
                    selected_experiments[size][opt][id][amine] = data_dict[size][opt]['k_x'][id][amine]

    full_dfs = []
    test_dfs = []

    # Find the experiments selected
    # For the full dataset setting
    for opt in rand_options:
        for id in set_idx:
            for amine in selected_experiments['full'][opt][id]:
                for exp in selected_experiments['full'][opt][id][amine]:
                    df_exp = df.loc[(df['_rxn_organic-inchikey'] == amine) &
                                    (df['_rxn_M_acid'] == exp[0]) &
                                    (df['_rxn_M_inorganic'] == exp[1]) &
                                    (df['_rxn_M_organic'] == exp[2])
                                    ]
                    full_dfs.append(df_exp)

            # For the test dataset setting
            for amine in selected_experiments['test'][opt][id]:
                for exp in selected_experiments['test'][opt][id][amine]:
                    df_exp = df.loc[(df['_rxn_organic-inchikey'] == amine) &
                                    (df['_rxn_M_acid'] == exp[0]) &
                                    (df['_rxn_M_inorganic'] == exp[1]) &
                                    (df['_rxn_M_organic'] == exp[2])
                                    ]
                    test_dfs.append(df_exp)

            # Concatenate all rows to a single df
            full = pd.concat(full_dfs, ignore_index=True)
            test = pd.concat(test_dfs, ignore_index=True)

            # Write each df into their own dataset file
            if opt == 'random':
                full.to_csv(f'./data/selected_k/full_dataset_random_{id}.csv')
                test.to_csv(f'./data/selected_k/test_dataset_random_{id}.csv')
            else:
                full.to_csv(
                    f'./data/selected_k/full_dataset_w_success_{id}.csv')
                test.to_csv(
                    f'./data/selected_k/test_dataset_w_success_{id}.csv')


def load_data_dict(data_dict, setting, amine, data, set_id=0):
    # x_t, y_t, x_v, y_v, all_data, all_labels = data
    data_types = ['x_t', 'y_t', 'x_v', 'y_v', 'all_data', 'all_labels']
    if set_id not in data_dict[setting]:
        data_dict[setting][set_id] = {}
    for i, dtype in enumerate(data_types):
        if dtype not in data_dict[setting][set_id]:
            data_dict[setting][set_id][dtype] = {}
        data_dict[setting][set_id][dtype][amine] = data[i]

    return data_dict


def random_draw(data, num_draws=5, k_size=10, x_size=10, success=False, min_success=2):
    """Do number of random draws to select k + x amine specific experiments

    Args:
        data:               A tuple of all the amine specific experiments to do random draws from.
        num_draws:          An integer representing the number of random draws to conduct.
        k_size:             An integer representing the number of amine specific experiments in each draw to match the k
                                in the experimental plan.
        x_size:             An integer representing the number of amine specific experiments in each draw to match the x
                                in the experimental plan. This is on top of the k experiments above.
        success:            A boolean representing if we want to include a few successful experiments in each set of k
                                experiments or not. This is used for SVM, LR, and GBT that use amine only experiments
                                for training.
        min_success:        An integer representing the minimum number of successful experiments required in each random
                                draw if we are forcing each set to have some successful experiments.

    Returns:
        draws:              A dictionary representing sets of random draws, with the following hierarchy:
                                --> set_id |--> k_x, k_y: the selected k experiments data and labels.
                                           |--> x_v, y_v: all amine experiments except for the selected k.
                                           |              Used as the pool for active learning.
                                           |--> x_qry, y_qry: the selected k+x experiments data and labels.
    """

    # Set default dictionary
    draws = defaultdict(dict)

    # Unpack amine-specific data and labels
    x, y = data[0], data[1]

    # Offset num_draws by 1
    num_draws -= 1

    # Conduct random draws
    while num_draws >= 0:
        # Initialize each draw
        draws[num_draws] = {}

        # Pick k-many experiments
        k_qry = np.random.choice(x.shape[0], size=k_size, replace=False)
        k_x = x[k_qry]
        k_y = y[k_qry]

        if not success or (success and list(k_y).count(1) >= min_success):
            # Log the k experiments
            draws[num_draws]['k_x'] = k_x
            draws[num_draws]['k_y'] = k_y

            # Update the remaining experiments
            x_remaining = np.delete(x, k_qry, axis=0)
            y_remaining = np.delete(y, k_qry)

            # Save the remaining ones for active learning models
            draws[num_draws]['x_v'] = x_remaining
            draws[num_draws]['y_v'] = y_remaining

            # Pick x-many more experiments for non-active learning models
            x_qry = np.random.choice(
                x_remaining.shape[0], size=x_size, replace=False)
            x_x = x_remaining[x_qry]
            x_y = y_remaining[x_qry]

            # Aggregate the k+x experiments and load them to dictionary
            # TODO May need better keys
            x_qry = np.append(k_x, x_x).reshape(-1, k_x.shape[1])
            y_qry = np.append(k_y, x_y)
            draws[num_draws]['x_qry'] = x_qry
            draws[num_draws]['y_qry'] = y_qry

            # One draw completed
            num_draws -= 1

    return draws


def generate_dataset(import_function, dataset_type, data_dict, num_draws=5,
                     train_size=10, active_learning_iter=10,
                     cross_validation=True, verbose=True):
    """Generate the dataset used for training and validation with numbers of random draws

    Args:
        import_function:                The function to import the dataset.
                                            It should be either import_full_dataset or import_test_dataset.
        dataset_type:                   A string representing the type of dataset we are generating.
                                            It should be either 'full' or 'test'
        data_dict:                      A dictionary that stores all the training and validation sets generated.
        num_draws:                      An integer representing the number of random draws to generate the dataset.
        train_size:                     An integer representing the number of experiments required in each random draw,
                                            corresponds to the k value in the experimental plan.
        active_learning_iter:           An integer representing the number of experiments required in each random draw,
                                            corresponds to the x value in the experimental plan.
        cross_validation:               A boolean representing if we want to generate validation set or not.
        verbose:                        A boolean representing if we want to output additional information when
                                            generating dataset or not.

    Returns:
        data_dict:                      A dictionary representing the desired dataset, with the following hierarchy:
                 --> 'full' or 'test'
                    --> 'random' or 'w/_success'
                        --> 'k_x' or 'k_y'
                            --> integer 0 to 4 (given we want 5 random draws)
                                --> amine
                                    --> data
                 --> settings as a tuple in the format of
                    ('full' / 'test', 'w/_AL' / 'w/o_AL', 'w/_hx' / 'w/o_hx', 'w/_k' / 'w/o_k', 'w/_x' / 'w/o_x',
                        'random' / 'w/_success')
                    --> integer 0 to 4 (given we want 5 random draws, only 0 if querying for category 3)
                        --> 'x_t', 'y_t', 'x_v', 'y_v', 'all_data', or 'all_labels'
                            --> amine
                                --> data
    """
    training, validation, testing, counts = import_function(
        train_size,
        meta_batch_size=25,
        num_batches=250,
        verbose=verbose,
        cross_validation=cross_validation,
        meta=False
    )

    amines = list(training.keys())

    # Preset the dictionary keys to log the k random experiments
    random_options = ['random', 'w/_success']
    data_types = ['k_x', 'k_y']
    for opt in random_options:
        if opt not in data_dict[dataset_type]:
            data_dict[dataset_type][opt] = defaultdict(dict)
            for dtype in data_types:
                if dtype not in data_dict[dataset_type][opt]:
                    data_dict[dataset_type][opt][dtype] = defaultdict(dict)

    for amine in amines:
        # Unload all data from the dataset:
        # All historical data and labels
        x_hx, y_hx = training[amine][0], training[amine][1]
        # All amine-specific data and labels
        x_amine, y_amine = validation[amine][0], validation[amine][1]

        # Category 3:
        # Full dataset under option 1 W/O ACTIVE LEARNING
        # No need for random draws here
        # Load corresponding data into dictionary
        data = (x_hx, y_hx, x_amine, y_amine, x_amine, y_amine)
        setting = (dataset_type, 'w/o_AL', 'w/_hx', 'w/o_k', 'w/o_x')
        data_dict = load_data_dict(data_dict, setting, amine, data)

        # For the remaining categories
        # Randomly draw k-many amine specific experiments num_draws many times
        # Regular random draws with no success specifications
        data = (x_amine, y_amine)
        random_draws = random_draw(data, num_draws, train_size,
                                   active_learning_iter, success=False)

        # Random draws with at least one successful experiment for each amine
        if list(y_amine).count(1) < 2:
            random_draws_w_success = random_draw(
                data,
                num_draws,
                train_size,
                active_learning_iter,
                success=True,
                min_success=1
            )
        else:
            random_draws_w_success = random_draw(
                data,
                num_draws,
                train_size,
                active_learning_iter,
                success=True
            )

        for i in range(num_draws):
            # Regular random draw with no success specification
            # Unpack the randomly selected experiments and their variants
            # Randomly selected k many experiments
            k_x, k_y = random_draws[i]['k_x'], random_draws[i]['k_y']
            # All amine-specific experiments excluding the selected k experiments
            x_v, y_v = random_draws[i]['x_v'], random_draws[i]['y_v']
            # On top of the selected k many experiments, draw x more to have k+x many
            x_qry, y_qry = random_draws[i]['x_qry'], random_draws[i]['y_qry']

            # Log the selected k experiments for later uses
            data_dict[dataset_type]['random']['k_x'][i][amine] = k_x
            data_dict[dataset_type]['random']['k_y'][i][amine] = k_y

            # Regular random draw with at least one success each
            # Unpack the randomly selected experiments and their variants
            # Randomly selected k many experiments with at least 1 success
            k_x_s, k_y_s = random_draws_w_success[i]['k_x'], random_draws_w_success[i]['k_y']
            # All amine-specific experiments excluding the selected k experiments
            x_v_s, y_v_s = random_draws_w_success[i]['x_v'], random_draws_w_success[i]['y_v']
            # On top of the selected k many experiments, draw x more to have k+x many
            x_qry_s, y_qry_s = random_draws_w_success[i]['x_qry'], random_draws_w_success[i]['y_qry']

            # Log the selected k experiments with at least one success for later uses
            data_dict[dataset_type]['w/_success']['k_x'][i][amine] = k_x_s
            data_dict[dataset_type]['w/_success']['k_y'][i][amine] = k_y_s

            # CATEGORY 4(i):
            # Full dataset using historical + k + x experiments W/O ACTIVE LEARNING
            # Aggregate the historical experiments and the k + x amine specific ones
            x_agg = np.append(x_hx, x_qry).reshape(-1, x_hx.shape[1])
            y_agg = np.append(y_hx, y_qry)

            # Load corresponding data into dictionary
            data = (x_agg, y_agg, x_amine, y_amine, x_amine, y_amine)
            setting = (dataset_type, 'w/o_AL', 'w/_hx',
                       'w/_k', 'w/_x', 'random')
            data_dict = load_data_dict(
                data_dict, setting, amine, data, set_id=i)

            # CATEGORY 4(ii):
            # Full dataset using k + x amine only experiments W/O ACTIVE LEARNING
            # Regular random draw with no success specification
            # Load corresponding data into dictionary
            data = (x_qry, y_qry, x_amine, y_amine, x_amine, y_amine)
            setting = (dataset_type, 'w/o_AL', 'w/o_hx',
                       'w/_k', 'w/_x', 'random')
            data_dict = load_data_dict(
                data_dict, setting, amine, data, set_id=i)

            # Regular random draw with at least one success each
            # Load corresponding data into dictionary
            data = (x_qry_s, y_qry_s, x_amine, y_amine, x_amine, y_amine)
            setting = (dataset_type, 'w/o_AL', 'w/o_hx',
                       'w/_k', 'w/_x', 'w/_success')
            data_dict = load_data_dict(
                data_dict, setting, amine, data, set_id=i)

            # CATEGORY 5(i):
            # Full dataset using historical + k experiments W/ ACTIVE LEARNING
            # Aggregate the historical experiments and the k amine specific ones
            x_agg = np.append(x_hx, k_x).reshape(-1, x_hx.shape[1])
            y_agg = np.append(y_hx, k_y)

            # Load corresponding data into dictionary
            data = (x_agg, y_agg, x_v, y_v, x_amine, y_amine)
            setting = (dataset_type, 'w/_AL', 'w/_hx',
                       'w/_k', 'w/o_x', 'random')
            data_dict = load_data_dict(
                data_dict, setting, amine, data, set_id=i)

            # Category 5(b):
            # Full dataset using k amine only experiments W/ ACTIVE LEARNING
            # Regular random draw with no success specification
            # Load corresponding data into dictionary
            data = (k_x, k_y, x_v, y_v, x_amine, y_amine)
            setting = (dataset_type, 'w/_AL', 'w/o_hx',
                       'w/_k', 'w/o_x', 'random')
            data_dict = load_data_dict(
                data_dict, setting, amine, data, set_id=i)

            # Regular random draw with at least one success each
            # Load corresponding data into dictionary
            data = (k_x_s, k_y_s, x_v_s, y_v_s, x_amine, y_amine)
            setting = (dataset_type, 'w/_AL', 'w/o_hx',
                       'w/_k', 'w/o_x', 'w/_success')
            data_dict = load_data_dict(
                data_dict, setting, amine, data, set_id=i)

    return data_dict


def process_dataset(num_draw=5, train_size=10, active_learning_iter=10, verbose=True, cross_validation=True, full=True,
                    active_learning=True, w_hx=True, w_k=True, success=False):
    """
    Generate and/or provide the desired training and validation data for categorical models

    Args:
        num_draw:                       An integer representing the number of random draws to generate the dataset.
        train_size:                     An integer representing the number of experiments required in each random draw,
                                            corresponds to the k value in the experimental plan.
        active_learning_iter:           An integer representing the number of experiments required in each random draw,
                                            corresponds to the x value in the experimental plan.
        cross_validation:               A boolean representing if we want to generate validation set or not.
        verbose:                        A boolean representing if we want to output additional information when
                                            generating dataset or not.
        full:                           A boolean representing if we want to import the full or the test dataset.
        active_learning:                A boolean representing if the model will conduct active learning with the given
                                            data or not.
        w_hx:                           A boolean representing if the model will be trained with historical data or not.
        w_k:                            A boolean representing if the model will be trained with the k amine-specific
                                            experiments in each set or not.
        success:                        A boolean representing if the model will be given a jump start with some
                                            successful experiments in the training set or not.

    Returns:
        A dictionary of training and validation data and labels with the following structure:
            --> integer 0 to 4 (given we want 5 random draws, only 0 if querying for category 3)
                --> 'x_t', 'y_t', 'x_v', 'y_v', 'all_data', or 'all_labels'
                    --> amine
                        --> data
    """

    data_dict = defaultdict(dict)

    data_dict_path = Path('./data/non_meta_data.pkl')
    if not data_dict_path.exists():
        # Generate full dataset
        data_dict = generate_dataset(
            import_full_dataset,
            'full',
            data_dict,
            num_draws=num_draw,
            train_size=train_size,
            active_learning_iter=active_learning_iter,
            cross_validation=cross_validation,
            verbose=verbose
        )

        # Generate test dataset
        data_dict = generate_dataset(
            import_test_dataset,
            'test',
            data_dict,
            num_draws=num_draw,
            train_size=train_size,
            active_learning_iter=active_learning_iter,
            cross_validation=cross_validation,
            verbose=verbose
        )

        with open(data_dict_path, "wb") as f:
            pickle.dump(data_dict, f)

        # Store the k experiments selected for each amine to two separate csv files
        write_sel_to_csv(data_dict, num_draw)

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
    rand_setting = 'random' if not success else 'w/_success'

    # For w_x setting
    if not active_learning:
        if w_hx:
            if not w_k:
                print(
                    'Training "H" models with historical data only and no active learning.')
                w_x_setting = 'w/o_x'
            else:
                print(
                    'Training "Hkx" models with historical data, k+x amine data and no active learning.')
                w_x_setting = 'w/_x'
        else:
            print('Training "kx" models with k+x amine data only and no active learning.')
            w_x_setting = 'w/_x'
    else:
        if w_hx:
            print(
                'Training "ALHk" models with historical data, k amine data, and active learning.')
            w_x_setting = 'w/o_x'
        else:
            print('Training "ALk" models with k amine data only and active learning.')
            w_x_setting = 'w/o_x'

    if not active_learning and w_hx and not w_k:
        setting = (dataset_size_setting, active_learning_setting,
                   w_hx_setting, w_k_setting, w_x_setting)
    else:
        setting = (dataset_size_setting, active_learning_setting,
                   w_hx_setting, w_k_setting, w_x_setting, rand_setting)

    return data_dict[setting]


if __name__ == "__main__":
    meta_batch_size = 10
    k_shot = 20
    num_batches = 10
    amine_left_out_batches, amine_cross_validate_samples, amine_test_samples, counts = import_test_dataset(
        k_shot, meta_batch_size, num_batches, verbose=True, cross_validation=True, meta=False)

    _ = process_dataset()
