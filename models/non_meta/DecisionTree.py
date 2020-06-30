import os
import pickle
import argparse
import itertools

import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from modAL.models import ActiveLearner

from utils.dataset import import_full_dataset, import_test_dataset


class ActiveDecisionTree:
    """ A class of decision tree model with active learning.
    
    Attributes: 
        amine:          A string representing the amine this model is used for.
        model:          A RandomForestClassifier object as the classifier model.
        metrics:        A dictionary to store the performance metrics locally. It has the format of
                            {'metric_name': [metric_value]}.
        verbose:        A boolean representing whether it will prints out additional information to the terminal or not.
        stats_path:     A Path object representing the directory of the stats dictionary.
        model_name:     A string representing the name of the model for future plotting.
        all_data:       A numpy array representing all the data from the dataset.
        all_labels:     A numpy array representing all the labels from the dataset.
        x_t:            A numpy array representing the training data used for model training.
        y_t:            A numpy array representing the training labels used for model training.
        x_v:            A numpy array representing the testing data used for active learning.
        y_v:            A numpy array representing the testing labels used for active learning.
        learner:        An ActiveLearner to conduct active learning with. See modAL documentation for more details.
        y_preds:        A numpy array representing the predicted labels given all data input.
    """

    def __init__(self, amine=None, config=None, verbose=True, stats_path=Path('./results/stats.pkl'),
                 model_name='Decision_Tree'):
        """initialization of the class"""

        self.amine = amine

        if config:
            self.model = DecisionTreeClassifier(max_depth=config['max_depth'],
                                                min_samples_split=config['min_samples_split'],
                                                min_samples_leaf=config['min_samples_leaf'],
                                                max_leaf_nodes=config['max_leaf_nodes'])
        else:
            self.model = DecisionTreeClassifier()

        self.metrics = defaultdict(list)
        self.verbose = verbose
        self.stats_path = stats_path
        self.model_name = model_name

    # TODO: find out what to do with this part
    def load_dataset(self, training_batches, validation_batches, meta=False):
        """Load the dataset for training and testing into model under option 1 or option 2
        
        Option 1 is train with observations of other tasks and validate on the task-specific observations.
        Option 2 is to train and validate on the task-specific observations.
        
        Args:
            training_batches:           A dictionary representing the training batches used to train.
                                            See dataset.py for specific structure.
            validation_batches:         A dictionary representing the training batches used to train.
                                            See dataset.py for specific structure.
            meta:                       A boolean representing if it will be trained under option 1 or option 2.
                                            Option 1 is train with observations of other tasks and validate on the
                                            task-specific observations.
                                            Option 2 is to train and validate on the task-specific observations.
        """

        if meta is True:
            # option 2
            self.x_t = validation_batches[self.amine][0]
            self.y_t = validation_batches[self.amine][1]
            self.x_v = validation_batches[self.amine][2]
            self.y_v = validation_batches[self.amine][3]

            self.all_data = np.concatenate((self.x_t, self.x_v))
            self.all_labels = np.concatenate((self.y_t, self.y_v))

            if self.verbose:
                print("Conducting Training under Option 2.")

        else:
            self.x_t = training_batches[self.amine][0]
            self.y_t = training_batches[self.amine][1]
            self.x_v = validation_batches[self.amine][0]
            self.y_v = validation_batches[self.amine][1]

            self.all_data = self.x_v
            self.all_labels = self.y_v

            if self.verbose:
                print("Conducting Training under Option 1.")

        if self.verbose:
            print(f'The training data has dimension of {self.x_t.shape}.')
            print(f'The training labels has dimension of {self.y_t.shape}.')
            print(f'The testing data has dimension of {self.x_v.shape}.')
            print(f'The testing labels has dimension of {self.y_v.shape}.')

    def train(self):
        """Train the decision tree model by setting up the ActiveLearner."""

        self.learner = ActiveLearner(estimator=self.model, X_training=self.x_t, y_training=self.y_t)
        # Evaluate zero-point performance
        self.evaluate()

    def active_learning(self, num_iter=None, to_params=True):
        """The active learning loop

        This is the active learning model that loops around the decision tree model
        to look for the most uncertain point and give the model the label to train

        Args:
            num_iter:   An integer that is the number of iterations.
                        Default = None
            to_params:  A boolean that decide if to store the metrics to the dictionary,
                        detail see "store_metrics_to_params" function.
                        Default = True
        """

        num_iter = num_iter if num_iter else self.x_v.shape[0]

        for _ in range(num_iter):
            # Query the most uncertain point from the active learning pool
            query_index, query_instance = self.learner.query(self.x_v)

            # Teach our ActiveLearner model the record it has requested.
            uncertain_data, uncertain_label = self.x_v[query_index].reshape(1, -1), self.y_v[query_index].reshape(1, )
            self.learner.teach(X=uncertain_data, y=uncertain_label)

            self.evaluate()

            # Remove the queried instance from the unlabeled pool.
            self.x_t = np.append(self.x_t, uncertain_data).reshape(-1, self.all_data.shape[1])
            self.y_t = np.append(self.y_t, uncertain_label)
            self.x_v = np.delete(self.x_v, query_index, axis=0)
            self.y_v = np.delete(self.y_v, query_index)

        if to_params:
            self.store_metrics_to_params()

    def evaluate(self, store=True):
        """ Evaluation of the model

        Args:
            store:  A boolean that decides if to store the metrics of the performance of the model.
                    Default = True
        """

        # Calculate and report our model's accuracy.
        accuracy = self.learner.score(self.all_data, self.all_labels)

        self.y_preds = self.learner.predict(self.all_data)

        # TODO: move the following parts into utils maybe?
        # Calculated confusion matrix
        cm = confusion_matrix(self.all_labels, self.y_preds)

        # To prevent nan value for precision, we set it to 1 and send out a warning message
        if cm[1][1] + cm[0][1] != 0:
            precision = cm[1][1] / (cm[1][1] + cm[0][1])
        else:
            precision = 1.0
            if self.verbose:
                print('WARNING: zero division during precision calculation')

        recall = cm[1][1] / (cm[1][1] + cm[1][0])
        true_negative = cm[0][0] / (cm[0][0] + cm[0][1])
        bcr = 0.5 * (recall + true_negative)

        if store:
            self.store_metrics_to_model(cm, accuracy, precision, recall, bcr)

    def store_metrics_to_model(self, cm, accuracy, precision, recall, bcr):
        """Store the performance metrics

        The metrics are specifically the confusion matrices, accuracies,
        precisions, recalls and balanced classification rates.

        Args:
            cm:             A numpy array representing the confusion matrix given our predicted labels and the actual
                                corresponding labels. It's a 2x2 matrix for the drp_chem model.
            accuracy:       A float representing the accuracy rate of the model: the rate of correctly predicted
                                reactions out of all reactions.
            precision:      A float representing the precision rate of the model: the rate of the number of actually
                                successful reactions out of all the reactions predicted to be successful.
            recall:         A float representing the recall rate of the model: the rate of the number of reactions
                                predicted to be successful out of all the actual successful reactions.
            bcr:            A float representing the balanced classification rate of the model. It's the average value
                                of recall rate and true negative rate.
        """

        self.metrics['confusion_matrices'].append(cm)
        self.metrics['accuracies'].append(accuracy)
        self.metrics['precisions'].append(precision)
        self.metrics['recalls'].append(recall)
        self.metrics['bcrs'].append(bcr)

        if self.verbose:
            print(cm)
            print('accuracy for model is', accuracy)
            print('precision for model is', precision)
            print('recall for model is', recall)
            print('balanced classification rate for model is', bcr)

    def store_metrics_to_params(self):
        """Store the metrics results to the model's parameters dictionary

        Use the same logic of saving the metrics for each model.
        Dump the cross validation statistics to a pickle file.
        """

        model = self.model_name

        if self.stats_path.exists():
            with open(self.stats_path, "rb") as f:
                stats_dict = pickle.load(f)
        else:
            stats_dict = {}

        if model not in stats_dict:
            stats_dict[model] = defaultdict(list)

        stats_dict[model]['amine'].append(self.amine)
        stats_dict[model]['accuracies'].append(self.metrics['accuracies'])
        stats_dict[model]['confusion_matrices'].append(self.metrics['confusion_matrices'])
        stats_dict[model]['precisions'].append(self.metrics['precisions'])
        stats_dict[model]['recalls'].append(self.metrics['recalls'])
        stats_dict[model]['bcrs'].append(self.metrics['bcrs'])

        # Save this dictionary in case we need it later
        with open(self.stats_path, "wb") as f:
            pickle.dump(stats_dict, f)

    def save_model(self, train_size, n_way, pretrain):
        """Save the data used to train, validate and test the model to designated folder.

        Args:
            k_shot:                 An integer representing the number of training samples per class.
            n_way:                  An integer representing the number of classes per task.
            pretrain:               A boolean representing if it will be trained under option 1 or option 2.
                                        Option 1 is train with observations of other tasks and validate on the
                                        task-specific observations.
                                        Option 2 is to train and validate on the task-specific observations.
        """

        option = 1 if pretrain else 2

        dst_root = './results/DecisionTree_few_shot/option_{0:d}'.format(option)

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
            print('No folder for decision tree model storage found')
            print(f'Make folder to store model at')

        model_folder = '{0:s}/DecisionTree_{1:d}_shot_{2:d}_way_option_{3:d}_{4:s}'.format(dst_root,
                                                                                           train_size,
                                                                                           n_way,
                                                                                           option,
                                                                                           self.amine)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print('No folder for decision tree model storage found')
            print(f'Make folder to store model of amine {self.amine} at')
        else:
            print(f'Found existing folder. Model of amine {self.amine} will be stored at')
        print(model_folder)

        # Dump the model
        file_name = "DecisionTree_{0:s}_option_{1:d}.pkl".format(self.amine, option)
        with open(os.path.join(model_folder, file_name), "wb") as f:
            pickle.dump(self, f)


def fine_tune(training_batches, validation_batches, info=False):
    """Fine tune the model based on average bcr performance to find the best model hyper-parameters.
    Args:
        training_batches:       A dictionary representing the training batches used to train.
                                    See dataset.py for specific structure.
        validation_batches:     A dictionary representing the training batches used to train.
                                    See dataset.py for specific structure.
        info:                   A boolean. Setting it to True will make the function print out additional information
                                    during the fine-tuning stage.
                                    Default to False.
    Returns:
        best_option:            A dictionary representing the hyper-parameters that yields the best performance on
                                    average. For DecisionTree, the current keys are: 'criterion', 'splitter',
                                    'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features',
                                    'max_leaf_nodes', 'class_weight', 'ccp_alpha'.
    """

    # Set all possible combinations
    params = {
        'max_depth': [i for i in range(1, 11)],
        'min_samples_split': [i for i in range(2, 11)],
        'min_samples_leaf': [i for i in range(1, 6)],
        'max_leaf_nodes': []
    }

    combinations = []

    keys, values = zip(*params.items())
    for bundle in itertools.product(*values):
        combinations.append(dict(zip(keys, bundle)))

    # Set baseline performance
    base_accuracies = []
    base_precisions = []
    base_recalls = []
    base_bcrs = []
    base_aucs = []

    for amine in training_batches:
        ADT = ActiveDecisionTree(amine=amine, verbose=False)
        ADT.load_dataset(training_batches, validation_batches)
        ADT.train()

        # Calculate AUC
        auc = roc_auc_score(ADT.all_labels, ADT.y_preds)

        base_accuracies.append(ADT.metrics['accuracies'][-1])
        base_precisions.append(ADT.metrics['precisions'][-1])
        base_recalls.append(ADT.metrics['recalls'][-1])
        base_bcrs.append(ADT.metrics['bcrs'][-1])
        base_aucs.append(auc)

    # Calculated the average baseline performances
    base_avg_accuracy = sum(base_accuracies) / len(base_accuracies)
    base_avg_precision = sum(base_precisions) / len(base_precisions)
    base_avg_recall = sum(base_recalls) / len(base_recalls)
    base_avg_bcr = sum(base_bcrs) / len(base_bcrs)
    base_avg_auc = sum(base_aucs) / len(base_aucs)

    best_metric = base_avg_auc

    if info:
        print(f'Baseline average accuracy is {base_avg_accuracy}')
        print(f'Baseline average precision is {base_avg_precision}')
        print(f'Baseline average recall is {base_avg_recall}')
        print(f'Baseline average bcr is {base_avg_bcr}')
        print(f'Baseline average auc is {base_avg_auc}')

    best_option = {}

    option_no = 1  # Debug

    # Try out each possible combinations of hyper-parameters
    print(f'There are {len(combinations)} many combinations to try.')
    for option in combinations:
        accuracies = []
        precisions = []
        recalls = []
        bcrs = []
        aucs = []

        if info:
            print(f'Trying option {option_no}')

        for amine in training_batches:
            # print("Training and cross validation on {} amine.".format(amine))
            ADT = ActiveDecisionTree(amine=amine, config=option, verbose=False)
            ADT.load_dataset(training_batches, validation_batches)
            ADT.train()

            # Calculate AUC
            auc = roc_auc_score(ADT.all_labels, ADT.y_preds)

            accuracies.append(ADT.metrics['accuracies'][-1])
            precisions.append(ADT.metrics['precisions'][-1])
            recalls.append(ADT.metrics['recalls'][-1])
            bcrs.append(ADT.metrics['bcrs'][-1])
            aucs.append(auc)

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_bcr = sum(bcrs) / len(bcrs)
        avg_auc = sum(aucs) / len(aucs)

        if avg_auc > best_metric:
            best_metric = avg_auc
            best_option = option
            if info:
                print(f'The fine-tuned average accuracy is {avg_accuracy} vs. the base accuracy {base_avg_accuracy}')
                print(
                    f'The fine-tuned average precision is {avg_precision} vs. the base precision {base_avg_precision}')
                print(f'The fine-tuned average recall rate is {avg_recall} vs. the base recall rate {base_avg_recall}')
                print(f'The fine-tuned average bcr is {avg_bcr} vs. the base bcr {base_avg_bcr}')
                print(f'The fine-tuned average auc is {avg_auc} vs. the base auc {base_avg_auc}')
                print(f'The current best setting is {best_option}')
                print()

        option_no += 1

    if info:
        print()
        print(f'The best setting for all amines is {best_option}')
        print(f'With an average auc of {best_metric}')

    return best_option


def save_used_data(training_batches, validation_batches, testing_batches, counts, pretrain):
    """Save the data used to train, validate and test the model to designated folder
    Args:
        training_batches:       A dictionary representing the training batches used to train.
                                    See dataset.py for specific structure.
        validation_batches:     A dictionary representing the training batches used to train.
                                    See dataset.py for specific structure.
        testing_batches:        A dictionary representing the training batches used to train.
                                    See dataset.py for specific structure.
        counts:                 A dictionary with 'total' and each available amines as keys and lists of length 2 as
                                    values in the format of: [# of failed reactions, # of successful reactions]
        pretrain:               A boolean representing if it will be trained under option 1 or option 2.
                                    Option 1 is train with observations of other tasks and validate on the
                                    task-specific observations.
                                    Option 2 is to train and validate on the task-specific observations.
    """

    # Indicate which option we used the data for
    option = 1 if pretrain else 2

    # Set up the destination folder to save the data
    data_folder = './results/DecisionTree_few_shot/option_{0:d}/data'.format(option)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print('No folder for decision tree model data storage found')
        print('Make folder to store data used for decision tree models at')
    else:
        print('Found existing folder. Data used for models will be stored at')
    print(data_folder)

    # Put all data into a dictionary for easier use later
    data = {
        'training_batches': training_batches,
        'validation_batches': validation_batches,
        'testing_batches': testing_batches,
        'counts': counts
    }

    # Save the file using pickle
    file_name = "DecisionTree_data.pkl"
    with open(os.path.join(data_folder, file_name), "wb") as f:
        pickle.dump(data, f)


def parse_args():
    """Set up the initial variables for running decision tree.

    Retrieves argument values from the terminal command line to create an argparse.Namespace object for
        initialization.

    Args:
        N/A

    Returns:
        args: Namespace object with the following attributes:
            datasource:         A string identifying the datasource to be used, with default datasource set to drp_chem.
            train_size          An integer representing the number of samples use for training. Default set to 1.
            train:              A train_flag attribute. Including it in the command line will set the train_flag to
                                    True by default.
            test:               A train_flag attribute. Including it in the command line will set the train_flag to
                                    False.
            cross_validate:     A boolean. Including it in the command line will run the model with cross-validation.
            pretrain:           A boolean representing if it will be trained under option 1 or option 2.
                                    Option 1 is train with observations of other tasks and validate on the
                                    task-specific observations.
                                    Option 2 is to train and validate on the task-specific observations.
            full_dataset:       A boolean representing if we want to load the full dataset or the test sample dataset.
            verbose:            A boolean. Including it in the command line will output additional information to the
                                    terminal for functions with verbose feature.
    """

    parser = argparse.ArgumentParser(description='Setup variables for active learning SVM.')
    parser.add_argument('--datasource', type=str, default='drp_chem', help='datasource to be used')

    parser.add_argument('--train_size', dest='train_size', default=1, help='number of samples used for training')

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--cross_validate', action='store_true', help='use cross-validation for training')
    parser.add_argument('--pretrain', action='store_true', help='load the dataset under option 1. Not include this will'
                                                                ' load the dataset under option 2. See documentation in'
                                                                ' codes for details.')
    parser.add_argument('--full', dest='full_dataset', action='store_true', help='load the full dataset or the test '
                                                                                 'sample dataset')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    return args


def run_model(DecisionTree_params):
    """Full-scale training, validation and testing using all amines.
    Args:
        DecisionTree_params:         A dictionary of the parameters for the decision tree model.
                                        See initialize() for more information.
    """
    '''
    # For visualization, may need deletion
    features = ['_rxn_M_acid', '_rxn_M_inorganic', '_rxn_M_organic', '_solv_GBL', '_solv_DMSO', '_solv_DMF',
                '_stoich_mmol_org', '_stoich_mmol_inorg', '_stoich_mmol_acid', '_stoich_mmol_solv', '_stoich_org/solv',
                '_stoich_inorg/solv', '_stoich_acid/solv', '_stoich_org+inorg/solv', '_stoich_org+inorg+acid/solv',
                '_stoich_org/liq', '_stoich_inorg/liq', '_stoich_org+inorg/liq', '_stoich_org/inorg',
                '_stoich_acid/inorg', '_rxn_Temperature_C', '_rxn_Reactiontime_s', '_feat_AvgPol',
                '_feat_Refractivity', '_feat_MaximalProjectionArea', '_feat_MaximalProjectionRadius',
                '_feat_maximalprojectionsize', '_feat_MinimalProjectionArea', '_feat_MinimalProjectionRadius',
                '_feat_minimalprojectionsize', '_feat_MolPol', '_feat_VanderWaalsSurfaceArea', '_feat_ASA',
                '_feat_ASA_H', '_feat_ASA_P', '_feat_ASA-', '_feat_ASA+', '_feat_ProtPolarSurfaceArea',
                '_feat_Hacceptorcount', '_feat_Hdonorcount', '_feat_RotatableBondCount', '_raw_standard_molweight',
                '_feat_AtomCount_N', '_feat_BondCount', '_feat_ChainAtomCount', '_feat_RingAtomCount',
                '_feat_primaryAmine', '_feat_secondaryAmine', '_rxn_plateEdgeQ', '_feat_maxproj_per_N',
                '_raw_RelativeHumidity']
    '''

    # Unload parameters
    config = DecisionTree_params['config']
    cross_validation = DecisionTree_params['cross_validate']
    verbose = DecisionTree_params['verbose']
    pretrain = DecisionTree_params['pretrain']
    stats_path = DecisionTree_params['stats_path']
    model_name = DecisionTree_params['model_name']
    print(model_name)

    # Set up if we want to load the meta dataset for option 2 or not
    meta = not pretrain
    # Set up the number of samples used for training under option 2
    train_size = DecisionTree_params['train_size']
    # Specify the desired operation
    fine_tuning = DecisionTree_params['fine_tuning']
    if fine_tuning:
        ft_training_batches, ft_validation_batches, ft_testing_batches, ft_counts = import_full_dataset(
            train_size, meta_batch_size=25, num_batches=250, verbose=verbose, cross_validation=cross_validation,
            meta=False)
        best_config = fine_tune(ft_training_batches, ft_validation_batches, info=True)
    else:
        # Load the full dataset for training and validation
        if DecisionTree_params['full_dataset']:
            training_batches, validation_batches, testing_batches, counts = import_full_dataset(train_size,
                                                                                                meta_batch_size=25,
                                                                                                num_batches=250,
                                                                                                verbose=verbose,
                                                                                                cross_validation=cross_validation,
                                                                                                meta=meta)
        else:
            training_batches, validation_batches, testing_batches, counts = import_test_dataset(train_size,
                                                                                                meta_batch_size=25,
                                                                                                num_batches=250,
                                                                                                verbose=verbose,
                                                                                                cross_validation=cross_validation,
                                                                                                meta=meta)
        # Save the data used for training and testing for reproducibility
        save_used_data(training_batches, validation_batches, testing_batches, counts, pretrain)
        # print(training_batches.keys())
        for amine in training_batches:
            print(f'Training and active learning on amine {amine}')
            # Create the KNN model instance for the specific amine
            ADT = ActiveDecisionTree(amine=amine, config=config, verbose=verbose, stats_path=stats_path,
                                     model_name=model_name)
            # Load the training and validation set into the model
            ADT.load_dataset(training_batches, validation_batches, meta)
            # Train the data on the training set
            ADT.train()
            # Conduct active learning with all the observations available in the pool
            ADT.active_learning(to_params=True)
            '''
            # Plot the decision tree
            file_name = './results/{}_dt_op1.dot'.format(amine) if pretrain else './results/{}_dt_op2.dot'.format(amine)
            export_graphviz(ADT.model,
                            feature_names=features,
                            class_names=['FAILURE', 'SUCCESS'],
                            out_file=file_name,
                            filled=True,
                            rounded=True,
                            special_characters=True)
            '''
            # Save the model for future reproducibility
            ADT.save_model(train_size, 2, pretrain)
            # TODO: testing part not implemented


def main():
    """Main driver function"""

    # This converts the args into a dictionary
    DecisionTree_params = vars(parse_args())

    run_model(DecisionTree_params)


if __name__ == "__main__":
    main()
