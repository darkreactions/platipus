import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set up various strings corresponding to headers
distribution_header = '_raw_modelname'
amine_header = '_rxn_organic-inchikey'
score_header = '_out_crystalscore'
name_header = 'name'
to_exclude = [score_header, amine_header, name_header, distribution_header]
path = './data/0050.perovskitedata_DRP.csv'

# Successful reaction is defined as having a crystal score of...
SUCCESS = 4


def import_full_dataset(k_shot, meta_batch_size, num_batches, verbose=False,
                        cross_validation=True, meta=True):
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
                        cross_validation=True,  meta=True):
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


if __name__ == "__main__":
    meta_batch_size = 10
    k_shot = 20
    num_batches = 10
    amine_left_out_batches, amine_cross_validate_samples, amine_test_samples, counts = import_test_dataset(
        k_shot, meta_batch_size, num_batches, verbose=True, cross_validation=True, meta=False)
