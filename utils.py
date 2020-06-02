import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load in the chemistry data, it is harder than it sounds
def load_chem_dataset(k_shot, meta_batch_size=32, num_batches=100, verbose=False):
    # "I'm limited by the technology of my time."
    # Ideally the model would choose from a Uniformly distributed pool of unlabeled reactions
    # Then we would run that reaction in the lab and give it back to the model
    # The issue is the labs are closed, so we have to restrict the model to the reactions drawn 
    # from uniform distributions that we have labels for
    # The list below is a list of inchi keys for amines that have a reaction drawn from a uniform 
    # distribution with a successful outcome (no point in amines that have no successful reaction)
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
    path = '.\\data\\0050.perovskitedata_DRP.csv'

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
    amines = [a for a in amines if a not in hold_out_amines]

    # Used to set up our weighted loss function
    counts = {}
    all_train = df[df[amine_header].isin(amines)]
    print('Number of reactions in training set', all_train.shape[0])
    all_train_success = all_train[all_train[score_header] == 1]
    print('Number of successful reactions in the training set', all_train_success.shape[0])

    # [Number of failed reactions, number of successful reactions]
    counts['total'] = [all_train.shape[0] - all_train_success.shape[0], all_train_success.shape[0]]

    amine_left_out_batches = {}
    amine_cross_validate_samples = {}

    for amine in amines:
        # Since we are doing cross validation, create a training set without each amine
        print("Generating batches for amine", amine)
        available_amines = [a for a in amines if a != amine]

        all_train = df[df[amine_header].isin(available_amines)]
        print(f'Number of reactions in training set holding out {amine}', all_train.shape[0])
        all_train_success = all_train[all_train[score_header] == 1]
        print(f'Number of successful reactions in training set holding out {amine}', all_train_success.shape[0])

        counts[amine] = [all_train.shape[0] - all_train_success.shape[0], all_train_success.shape[0]]
        batches = []
        for _ in range(num_batches):
            # t for train, v for validate (but validate is outer loop, trying to be consistent with the PLATIPUS code)
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

            batches.append([np.array(x_t), np.array(y_t), np.array(x_v), np.array(y_v)])

        amine_left_out_batches[amine] = batches

        # Now set up the cross validation data
        X = df[df[amine_header] == amine]
        y = X[score_header].values
        X = X.drop(to_exclude, axis=1).values
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

        amine_cross_validate_samples[amine] = [x_s, y_s, x_q, y_q]

    amine_test_samples = {}

    for a in hold_out_amines:
        # grab task
        X = df[df[amine_header] == a]

        y = X[score_header].values
        X = X.drop(to_exclude, axis=1).values
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

        amine_test_samples[a] = [x_s, y_s, x_q, y_q]

    # amine_left_out_batches structure:
    # key is amine left out, value has following hierarchy
    # batches->x_t, y_t, x_v, y_v -> meta_batch_size number of amines -> k_shot number of reactions -> each reaction has some number of features

    # amine_cross_validate_samples structure: 
    # key is amine which the data is for, value has the following hierarchy
    # x_s, y_s, x_q, y_q -> k_shot number of reactions -> each reaction has some number of features

    # amine_test_samples has the same structure as amine_cross_validate_samples

    if verbose:
        print('Number of features to train on is', len(df.columns) - len(to_exclude))

    return amine_left_out_batches, amine_cross_validate_samples, amine_test_samples, counts


# Do not use any data for model validation this time
def load_chem_dataset_testing(k_shot, meta_batch_size=32, num_batches=100, verbose=False):
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
    path = '.\\data\\0050.perovskitedata_DRP.csv'

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

    hold_out_amines = ['CALQKRVFTWDYDG-UHFFFAOYSA-N',
                       'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                       'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                       'JMXLWMIFDJCGBV-UHFFFAOYSA-N']
    print('Holding out', hold_out_amines)

    available_amines = [a for a in amines if a not in hold_out_amines]
    # Used to set up our weighted loss function
    counts = {}
    all_train = df[df[amine_header].isin(available_amines)]
    print('Number of reactions in training set', all_train.shape[0])
    all_train_success = all_train[all_train[score_header] == 1]
    print('Number of successful reactions in the training set', all_train_success.shape[0])

    counts['total'] = [all_train.shape[0] - all_train_success.shape[0], all_train_success.shape[0]]

    batches = []
    print('Generating training batches')
    for _ in range(num_batches):
        # t for train, v for validate (but validate is outer loop, trying to be consistent with the PLATIPUS code)
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

        batches.append([np.array(x_t), np.array(y_t), np.array(x_v), np.array(y_v)])

    amine_test_samples = {}

    print('Generating testing batches')
    for a in hold_out_amines:
        # grab task
        X = df[df[amine_header] == a]

        y = X[score_header].values
        X = X.drop(to_exclude, axis=1).values
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

        amine_test_samples[a] = [x_s, y_s, x_q, y_q]

    # amine_left_out_batches structure:
    # Here we have a list of batches
    # batches->x_t, y_t, x_v, y_v -> meta_batch_size number of amines -> k_shot number of reactions -> each reaction has some number of features

    # amine_test_samples structure: 
    # key is amine which the data is for, value has the following hierarchy
    # x_s, y_s, x_q, y_q -> k_shot number of reactions -> each reaction has some number of features

    if verbose:
        print('Number of features to train on is', len(df.columns) - len(to_exclude))

    return batches, amine_test_samples, counts


if __name__ == "__main__":
    load_chem_dataset(5, meta_batch_size=32, num_batches=100, verbose=True)
