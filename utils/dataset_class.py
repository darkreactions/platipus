import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import namedtuple, defaultdict
import pickle
from abc import ABC, abstractmethod

Setting = namedtuple('Setting', ['dataset_type', 'meta', 'AL', 'H', 'k', 'x',
                                 'set_id', 'selection'])


class BaseDataSet(ABC):
    def __init__(self,
                 dataset_path='./data/raw/0057.perovskitedata_DRPFeatures_2020-07-02.csv'):
        self.dataset_path = dataset_path
        self.distribution_header = '_raw_modelname'
        self.amine_header = '_rxn_organic-inchikey'
        self.score_header = '_out_crystalscore'
        self.name_header = 'name'
        self.to_exclude = [self.score_header, self.amine_header,
                           self.name_header, self.distribution_header, '_raw_RelativeHumidity']
        self.bool_cols = ['_solv_GBL',
                          '_solv_DMSO',
                          '_solv_DMF',
                          '_feat_primaryAmine',
                          '_feat_secondaryAmine',
                          '_rxn_plateEdgeQ']
        self.SUCCESS = 4
        self.viable_amines = ['ZEVRFFCPALTVDN-UHFFFAOYSA-N',
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
        self.hold_out_amines = ['CALQKRVFTWDYDG-UHFFFAOYSA-N',
                                'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                                'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                                'JMXLWMIFDJCGBV-UHFFFAOYSA-N']
        self.phase2_amines = ['CALQKRVFTWDYDG-UHFFFAOYSA-N',
                              'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                              'FCTHQYIDLRRROX-UHFFFAOYSA-N', ]

    def _import_chemdata(self, viable_amines):
        """
            Imports data from the dataset_path. Selects only uniformly sampled
            points. Selects only amines that are in viable amines
            Args:
                viable_amines: list of amines to filter

            return: 
                df:     filtered dataframe
                amines: list of amines in the filtered dataframe

        """
        df = pd.read_csv(self.dataset_path)
        df = df[df[self.distribution_header].str.contains('Uniform')]
        df = df[df[self.amine_header].isin(viable_amines)]

        df[self.score_header] = [1 if val == self.SUCCESS else 0 for val in
                                 df[self.score_header].values]
        amines = df[self.amine_header].unique().tolist()

        return df, amines

    def _generate_batch(self, meta_batch_size, available_amines,
                        to_exclude, k_shot):
        """Generate the batch for training amines

        Args:
            df:                 The data frame of the amines data
            meta_batch_size:    An integer. Batch size for meta learning
            available_amines:   A list. The list of amines that we are
                                generating batches on
            to_exclude:         A list. The columns in the dataset that we
                                need to drop
            k_shot:             An integer. The number of unseen classes
                                in the dataset
            amine_header:       The header of the amine list in the data frame,
                                default = '_rxn_organic-inchikey'
            score_header:       The header of the score header in the data
                                frame, default = '_out_crystalscore'

        return: A list of the batch with
        training and validation features and labels in numpy arrays.
        The format is [[training_feature],[training_label],
                        [validation_feature],[validation_label]]
        """
        x_t, y_t, x_v, y_v = [], [], [], []

        for _ in range(meta_batch_size):
            # Grab the tasks
            X = self.df[self.df[self.amine_header] ==
                        np.random.choice(available_amines)]

            y = X[self.score_header].values

            # Drop these columns from the dataset
            X = X.drop(to_exclude, axis=1).values

            # Standardize features since they are not yet standardized
            # in the dataset
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

    def _generate_valid_test_batch(self, X, y, k_shot, scaler):
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

        x_s = scaler.transform(x_s)
        x_q = scaler.transform(x_q)

        return [x_s, y_s, x_q, y_q]

    def _load_data_dict(self, data_dict, setting, data, amine, data_splits):
        # data_splits = ['x_t', 'y_t', 'x_v', 'y_v',
        #               'x_vx', 'y_vx', 'k_x', 'k_y']
        if amine not in data_dict[setting]:
            data_dict[setting][amine] = {}
        for i, dtype in enumerate(data_splits):
            if dtype not in data_dict[setting][amine]:
                data_dict[setting][amine][dtype] = {}
            data_dict[setting][amine][dtype] = data[i]

        return data_dict

    def _random_draw(self, data, num_draws, k, x_size,
                     success=False, min_success=2):
        """Do number of random draws to select k + x amine specific experiments

        Args:
            data:               A tuple of all the amine specific experiments
                                to do random draws from.
            num_draws:          An integer representing the number of random
                                draws to conduct.
            k_size:             An integer representing the number of amine
                                specific experiments in each draw to match the
                                k in the experimental plan.
            x_size:             An integer representing the number of amine
                                specific experiments in each draw to match the
                                x in the experimental plan. This is on top of
                                he k experiments above.
            success:            A boolean representing if we want to include a
                                few successful experiments in each set of k
                                    experiments or not. This is used for SVM,
                                    LR, and GBT that use amine only experiments
                                    for training.
            min_success:        An integer representing the minimum number of
                                successful experiments required in each random
                                    draw if we are forcing each set to have
                                    some successful experiments.

        Returns:
            draws:              A dictionary representing sets of random draws,
                                with the following hierarchy:
                                    --> set_id |--> k_x, k_y: the selected k
                                                              experiments data
                                                              and labels.
                                            |--> x_v, y_v: all amine
                                                           experiments except
                                                           for the selected k.
                                            |              Used as the pool for
                                                           active learning.
                                            |--> x_qry, y_qry: the selected k+x
                                                               experiments data
                                                               and labels.
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
            k_qry = np.random.choice(x.shape[0], size=k, replace=False)
            k_x = x[k_qry]
            k_y = y[k_qry]

            if not success or (success and list(k_y).count(1) >= min_success):
                # Log the k experiments
                draws[num_draws]['x_vk'] = k_x
                draws[num_draws]['y_vk'] = k_y

                # Update the remaining experiments
                x_remaining = np.delete(x, k_qry, axis=0)
                y_remaining = np.delete(y, k_qry)

                # Save the remaining ones for active learning models
                # draws[num_draws]['x_vx'] = x_remaining
                # draws[num_draws]['y_vy'] = y_remaining

                # Pick x-many more experiments for non-active learning models
                x_qry = np.random.choice(
                    x_remaining.shape[0], size=x_size, replace=False)
                x_x = x_remaining[x_qry]
                x_y = y_remaining[x_qry]

                draws[num_draws]['x_vx'] = x_x
                draws[num_draws]['y_vx'] = x_y

                # Aggregate the k+x experiments and load them to dictionary
                # TODO May need better keys
                x_qry = np.append(k_x, x_x).reshape(-1, k_x.shape[1])
                y_qry = np.append(k_y, x_y)
                draws[num_draws]['x_vkx'] = x_qry
                draws[num_draws]['y_vkx'] = y_qry

                draws[num_draws]['x_vrem'] = x_remaining
                draws[num_draws]['y_vrem'] = y_remaining

                # One draw completed
                num_draws -= 1

        return draws

    def _categorize_data(self, amines, training, meta_training, validation,
                         dataset_type, num_draws, k, x_size):
        data_dict = defaultdict(dict)

        for amine in amines:
            # Unload all data from the dataset:
            # All historical data and labels
            x_h, y_h = training[amine][0], training[amine][1]
            # All amine-specific data and labels
            x_v, y_v = validation[amine][0], validation[amine][1]

            x_h_meta, y_h_meta = (meta_training[amine][0],
                                  meta_training[amine][1])

            # Category 3:
            # Full dataset under option 1 W/O ACTIVE LEARNING
            # No need for random draws here
            # Load corresponding data into dictionary

            setting = Setting(dataset_type=dataset_type, AL=False, H=True,
                              k=False, x=False, set_id=None, selection=None,
                              meta=False)
            data = (x_h, y_h, x_v, y_v)
            data_splits = ['x_t', 'y_t', 'x_v', 'y_v']
            data_dict = self._load_data_dict(data_dict, setting,
                                             data, amine, data_splits)

            # For the remaining categories
            # Randomly draw k-many amine specific experiments
            # num_draws many times. Regular random draws with no
            # success specifications
            data = (x_v, y_v)
            random_draws = self._random_draw(data, num_draws, k, x_size,
                                             success=False)
            # Random draws with at least one successful experiment for
            # each amine
            if list(y_v).count(1) < 2:
                random_draws_w_success = self._random_draw(
                    data,
                    num_draws,
                    k,
                    x_size,
                    success=True,
                    min_success=1
                )
            else:
                random_draws_w_success = self._random_draw(
                    data,
                    num_draws,
                    k,
                    x_size,
                    success=True
                )

            for i in range(num_draws):
                # Regular random draw with no success specification
                # Unpack the randomly selected experiments and their variants
                # Randomly selected k many experiments
                x_vk, y_vk = random_draws[i]['x_vk'], random_draws[i]['y_vk']
                # All amine-specific experiments excluding the selected k
                # experiments
                x_vx, y_vx = random_draws[i]['x_vx'], random_draws[i]['y_vx']
                # On top of the selected k many experiments, draw x more to
                # have k+x many
                x_vkx, y_vkx = (random_draws[i]['x_vkx'],
                                random_draws[i]['y_vkx'])
                # Experiments remaining after selecting k experiments
                x_vrem, y_vrem = (random_draws[i]['x_vrem'],
                                  random_draws[i]['y_vrem'])

                # Log the selected k experiments for later uses
                # data_dict[dataset_type]['random']['x_vk'][i][amine] = x_vk
                # data_dict[dataset_type]['random']['y_vk'][i][amine] = y_vk

                # Regular random draw with at least one success each
                # Unpack the randomly selected experiments and their variants
                # Randomly selected k many experiments with at least 1 success
                x_vk_s, y_vk_s = (random_draws_w_success[i]['x_vk'],
                                  random_draws_w_success[i]['y_vk'])
                # All amine-specific experiments excluding the selected k
                # experiments
                x_vx_s, y_vx_s = (random_draws_w_success[i]['x_vx'],
                                  random_draws_w_success[i]['y_vx'])
                # On top of the selected k many experiments, draw x more to
                # have k+x many
                x_vkx_s, y_vkx_s = (random_draws_w_success[i]['x_vkx'],
                                    random_draws_w_success[i]['y_vkx'])
                x_vrem_s, y_vrem_s = (random_draws_w_success[i]['x_vrem'],
                                      random_draws_w_success[i]['y_vrem'])

                # Log the selected k experiments with at least one success
                # for later uses
                # data_dict[dataset_type]['w/_success']['k_x'][i][amine] =
                # x_vk_s
                # data_dict[dataset_type]['w/_success']['k_y'][i][amine] =
                # y_vk_s

                # CATEGORY 4(i):
                # Full dataset using historical + k + x experiments W/O
                # ACTIVE LEARNING
                # Aggregate the historical experiments and the k + x
                #  amine specific ones
                x_agg = np.append(x_h, x_vkx).reshape(-1, x_h.shape[1])
                y_agg = np.append(y_h, y_vkx)

                # Load corresponding data into dictionary
                data = (x_agg, y_agg, x_v, y_v, x_vk, y_vk)
                data_splits = ['x_t', 'y_t', 'x_v', 'y_v', 'x_vk', 'y_vk']
                # setting = (dataset_type, 'w/o_AL', 'w/_hx',
                #        'w/_k', 'w/_x', 'random')
                setting = Setting(dataset_type=dataset_type, AL=False, H=True,
                                  k=True, x=True, selection='random', set_id=i,
                                  meta=False)
                data_dict = self._load_data_dict(data_dict, setting, data,
                                                 amine, data_splits)

                # With atleast 1 success in k
                x_agg = np.append(x_h, x_vkx_s).reshape(-1, x_h.shape[1])
                y_agg = np.append(y_h, y_vkx_s)

                # Load corresponding data into dictionary
                data = (x_agg, y_agg, x_v, y_v, x_vk_s, y_vk_s)
                setting = Setting(dataset_type=dataset_type, AL=False, H=True,
                                  k=True, x=True, selection='success',
                                  set_id=i, meta=False)

                # CATEGORY 4(ii):
                # Full dataset using k + x amine only experiments W/O ACTIVE
                #  LEARNING
                # Regular random draw with no success specification
                # Load corresponding data into dictionary
                data = (x_vkx, y_vkx, x_v, y_v)
                data_splits = ['x_t', 'y_t', 'x_v', 'y_v']
                setting = Setting(dataset_type=dataset_type, AL=False, H=False,
                                  k=True, x=True, selection='random', set_id=i,
                                  meta=False)
                data_dict = self._load_data_dict(data_dict, setting,
                                                 data, amine, data_splits)

                # Regular random draw with at least one success each
                # Load corresponding data into dictionary
                data = (x_vkx_s, y_vkx_s, x_v, y_v)
                data_splits = ['x_t', 'y_t', 'x_v', 'y_v']
                setting = Setting(dataset_type=dataset_type, AL=False, H=False,
                                  k=True, x=True, selection='success',
                                  set_id=i, meta=False)
                data_dict = self._load_data_dict(data_dict, setting,
                                                 data, amine, data_splits)

                # CATEGORY 5(i):
                # Full dataset using historical + k experiments W/ ACTIVE
                #  LEARNING
                # Aggregate the historical experiments and the k amine
                #  specific ones
                x_agg = np.append(x_h, x_vk).reshape(-1, x_h.shape[1])
                y_agg = np.append(y_h, y_vk)
                data_splits = ['x_t', 'y_t', 'x_vrem', 'y_vrem', 'x_v', 'y_v']
                # Load corresponding data into dictionary
                data = (x_agg, y_agg, x_vrem, y_vrem, x_v, y_v)
                setting = Setting(dataset_type=dataset_type, AL=True, H=True,
                                  k=True, x=False, selection='random',
                                  set_id=i, meta=False)
                data_dict = self._load_data_dict(data_dict, setting,
                                                 data, amine, data_splits)

                # Category 5(ii):
                # Full dataset using k amine only experiments W/ ACTIVE
                # LEARNING
                # Regular random draw with no success specification
                # Load corresponding data into dictionary
                data_splits = ['x_t', 'y_t', 'x_vrem', 'y_vrem', 'x_v', 'y_v']
                data = (x_vk, y_vk, x_vrem, y_vrem, x_v, y_v)
                setting = Setting(dataset_type=dataset_type, AL=True, H=False,
                                  k=True, x=False, selection='random',
                                  set_id=i, meta=False)
                data_dict = self._load_data_dict(data_dict, setting,
                                                 data, amine, data_splits)

                # Regular random draw with at least one success each
                # Load corresponding data into dictionary
                data_splits = ['x_t', 'y_t', 'x_vrem', 'y_vrem', 'x_v', 'y_v']

                data = (x_vk_s, y_vk_s, x_vrem_s, y_vrem_s, x_v, y_v)
                setting = Setting(dataset_type=dataset_type, AL=True, H=False,
                                  k=True, x=False, selection='success',
                                  set_id=i, meta=False)
                data_dict = self._load_data_dict(data_dict, setting,
                                                 data, amine, data_splits)

                # Meta model datasets
                # 5(i) for meta models
                data_splits = ['x_t', 'y_t', 'x_vk', 'y_vk', 'x_vx', 'y_vx',
                               'x_vrem', 'y_vrem', 'x_v', 'y_v'] 
                data = (x_h_meta, y_h_meta, x_vk, y_vk, x_vx, y_vx,
                        x_vrem, y_vrem, x_v, y_v)
                meta_setting = Setting(dataset_type=dataset_type, AL=True,
                                       H=True, k=True, x=False,
                                       selection='random', set_id=i, meta=True)
                data_dict = self._load_data_dict(data_dict, meta_setting, data,
                                                 amine, data_splits)

        self.data_dict = data_dict
        return data_dict

    def print_config(self):
        print(f'Dataset: {self.dataset_type}'
              f'\nCross Validation: {self.cross_validation}'
              f'\n Number of draws: {self.num_draws}'
              f'\nPretrain samples(k): {self.k}'
              f'\nActive learning steps(x): {self.x}')
        dataset_options = {'Dataset code': ['H', 'Hkx', 'kx', 'ALHk',
                                            'ALk', 'metaALHk'],
                           'Description': ['Only historical data',
                                           'Historical data and randomly \
                                               selected k and x',
                                           'k and x randomly selected \
                                               experiments',
                                           'Active learning with historical \
                                               data and k pretraining samples',
                                           'Active learning with only k \
                                               pretraining samples',
                                           'Active learning with historical \
                                               data and k pretraining samples \
                                                   for meta learning']}
        dataset_description = pd.DataFrame.from_dict(dataset_options)
        print(dataset_description)

    def get_dataset(self, code, set_id, selection):
        code_set = {'H': Setting(dataset_type=self.dataset_type,
                                 AL=False, H=True, k=False, x=False,
                                 set_id=None, selection=None, meta=False),
                    'Hkx': Setting(dataset_type=self.dataset_type, meta=False,
                                   AL=False, H=True, k=True, x=True,
                                   set_id=set_id, selection=selection),
                    'kx': Setting(dataset_type=self.dataset_type, meta=False,
                                  AL=False, H=False, k=True, x=True,
                                  set_id=set_id, selection=selection),
                    'ALHk': Setting(dataset_type=self.dataset_type, meta=False,
                                    AL=True, H=True, k=True, x=False,
                                    set_id=set_id, selection=selection),
                    'ALk': Setting(dataset_type=self.dataset_type, meta=False,
                                   AL=True, H=False, k=True, x=False,
                                   set_id=set_id, selection=selection),
                    'metaALHk': Setting(dataset_type=self.dataset_type,
                                        meta=True, AL=True, H=True, k=True,
                                        x=False, set_id=set_id,
                                        selection=selection),
                    }
        if code in code_set:
            setting = code_set[code]
            # print(setting)
            if setting in self.data_dict:
                return self.data_dict[setting]
            else:
                print(f"Setting not found: {setting}")
        else:
            return None

    @abstractmethod
    def generate_dataset(self):
        pass


class Phase2DataSet(BaseDataSet):
    def _hold_out_data(self, meta=True, num_batches=250,
                       meta_batch_size=10, k_shot=10):

        training_amines = [a for a in self.viable_amines if a not in
                           self.hold_out_amines]
        self.df, self.amines = self._import_chemdata(training_amines)
        
        amine_test_samples = {}
        amine_training_batches = {}

        x_train = self.df
        y_train = x_train[self.score_header].values

        # Drop these columns from the dataset
        x_train = x_train.drop(self.to_exclude+self.bool_cols, axis=1).values

        # Standardize features since they are not yet standardized
        # in the dataset
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)

        x_train = np.concatenate([x_train, self.df[self.bool_cols].values], axis=1)
        
        batches = []

        if meta:
            print('Holding out', self.phase2_amines)
            # Used to set up our weighted loss function
            print('Generating training batches')
            for _ in range(num_batches):
                # t for train, v for validate (but validate is outer loop,
                # trying to be consistent with the PLATIPUS code)
                batch = self._generate_batch(meta_batch_size,
                                             training_amines, self.to_exclude,
                                             k_shot)
                batches.append(batch)
            for amine in self.phase2_amines:
                amine_training_batches[amine] = batches
        
        df, p2_amines = self._import_chemdata(self.phase2_amines)
        for amine in self.phase2_amines:
            if meta:
                amine_training_batches[amine] = batches
            else:
                amine_training_batches[amine] = (x_train, y_train)

            x_test = df[df[self.amine_header] == amine]
            y_test = x_test[self.score_header].values
            x_test = x_test.drop(self.to_exclude+self.bool_cols, axis=1).values
            x_test = scaler.transform(x_test)
            x_test_bool = df[df[self.amine_header] == amine]
            x_test_bool = x_test_bool[self.bool_cols].values

            x_test = np.concatenate([x_test, x_test_bool], axis=1)

            amine_test_samples[amine] = (x_test, y_test)

        return (amine_training_batches, amine_test_samples,
                scaler)

    def _load_test_samples(self, hold_out_amines, df, to_exclude, k_shot,
                           amine_header, score_header):
        """This is a function used for loading testing samples specifically

        Args:
            hold_out_amines:    The list of all holdout amines that are used
                                for testing.
                                DO NOT TOUCH!
            df:                 The data frame of the amines data
            to_exclude:         A list. The columns in the dataset that we
                                need to drop
            k_shot:             An integer. The number of unseen classes in
                                the dataset
            amine_header:       The header of the amine list in the data frame.
            score_header:       The header of the score header in the
                                data frame.

        return: A dictionary that contains the test sample amines' batches
        """
        amine_test_samples = {}
        for a in hold_out_amines:
            # grab task
            X = df[df[amine_header] == a]

            y = X[score_header].values
            X = X.drop(to_exclude, axis=1).values
            test_sample = self._generate_valid_test_batch(X, y, k_shot)

            amine_test_samples[a] = test_sample
        return amine_test_samples

    def generate_dataset(self, dataset_type='full', num_draws=5,
                         k=10, x=10):
        self.dataset_type = dataset_type
        self.num_draws = num_draws
        self.k = k
        self.x = x

        meta_training, m_v, scalers = self._hold_out_data(k_shot=x,
                                                          meta=True)
        training, validation, scalers = self._hold_out_data(k_shot=x,
                                                            meta=False)

        amines = self.phase2_amines
        self.scaler = scalers
        
        self._categorize_data(amines, training, meta_training, validation,
                              dataset_type, num_draws, k, x)


class DataSet(BaseDataSet):
    """
    Class to handle dataset
    """

    def _cross_validation(self, meta=True, num_batches=250,
                          meta_batch_size=25, k_shot=10):
        """
            Generates dataset for cross validation for meta and non-meta models
            Args:
                meta:           Bool to indicate whether model is meta or not
                num_batches:    Number of training batches for meta learning
                meta_batch_size:Batch size for meta learning
                k_shot:         Number of training samples from new amine
            Returns:
                training_batches:   Training data based on model
                validation_batches: Validation data for the left out amine
                counts:             Count success in training data

        """
        amines = [a for a in self.viable_amines if a not in
                  self.hold_out_amines]
        self.df, self.amines = self._import_chemdata(amines)

        all_train = self.df[self.df[self.amine_header].isin(self.amines)]
        print(f'Number of reactions in training set {all_train.shape[0]}')
        all_train_success = all_train[all_train[self.score_header] == 1]
        print('Number of successful reactions in the training set',
              all_train_success.shape[0])

        training_data = {}
        validation_data = {}
        scalers = {}

        for amine in self.amines:
            # Since we are doing cross validation,
            # create a training set without each amine
            print(f"Generating batches for amine: {amine}")
            available_amines = [a for a in self.amines if a != amine]

            all_train = self.df[self.df[self.amine_header].isin(
                    available_amines)]
            y = all_train[self.score_header].values
            all_train = all_train.drop(self.to_exclude+self.bool_cols, axis=1).values
            
            

            # Standardize features since they are not yet standardized
            # in the dataset
            scaler = StandardScaler()
            scaler.fit(all_train)
            all_train = scaler.transform(all_train)
            self.bool_data = self.df[self.df[self.amine_header].isin(
                    available_amines)]
            self.bool_data = self.bool_data[self.bool_cols].values

            all_train = np.concatenate([all_train, self.bool_data], axis=1)


            scaler.fit(all_train)
            all_train = scaler.transform(all_train)
            training_data[amine] = (all_train, y)
            scalers[amine] = scaler
            
            if meta:
                batches = []
                for _ in range(num_batches):
                    # t for train, v for validate (but validate is outer loop,
                    # trying to be consistent with the PLATIPUS code)
                    batch = self._generate_batch(meta_batch_size,
                                                 available_amines,
                                                 self.to_exclude, k_shot)
                    batches.append(batch)

                training_data[amine] = batches
            

            # Now set up the cross validation data
            val_data = self.df[self.df[self.amine_header] == amine]
            y = val_data[self.score_header].values
            val_data = val_data.drop(self.to_exclude, axis=1).values
            val_data = scaler.transform(val_data)
            cross_valid = (val_data, y)

            if meta:
                cross_valid = self._generate_valid_test_batch(val_data,
                                                              y, k_shot, 
                                                              scaler)
            validation_data[amine] = cross_valid

        return (training_data, validation_data, scalers)

    def generate_dataset(self, dataset_type='full', num_draws=5,
                         k=10, x=10, save_csv=False):
        # Variables to query the current dataset
        self.dataset_type = dataset_type
        self.num_draws = num_draws
        self.k = k
        self.x = x
        self.cross_validation = True

        if dataset_type == 'test':
            self.viable_amines = ['ZEVRFFCPALTVDN-UHFFFAOYSA-N',
                                  'KFQARYBEAKAXIC-UHFFFAOYSA-N',
                                  'NLJDBTZLVTWXRG-UHFFFAOYSA-N']
            self.hold_out_amines = []

        meta_training, m_v, scalers = self._cross_validation(k_shot=x,
                                                             meta=True)
        training, validation, scalers = self._cross_validation(k_shot=x,
                                                               meta=False)
        self.scaler = scalers
        amines = list(training.keys())
        self._categorize_data(amines, training, meta_training, validation,
                              dataset_type, num_draws, k, x)
        if save_csv:
            pass
        


if __name__ == '__main__':
    dataset = Phase2DataSet(
        dataset_path='./data/raw/0057.perovskitedata_DRPFeatures_2020-07-02.csv')
    data_dict = dataset.generate_dataset(dataset_type='full')
    with open('./data/phase2_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
