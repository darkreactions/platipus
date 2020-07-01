import os
import pickle
import argparse
import itertools

import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression

from modAL.models import ActiveLearner

from utils.dataset import import_full_dataset, import_test_dataset


class ActiveLogisticRegression:
    """
    A class of Logistic Regression model with active learning
    """

    def __init__(self, amine=None, config=None, verbose=True, stats_path=Path('./results/stats.pkl'),
                 model_name='Logistic Regression'):
        """initialization of the class

        Args:
            penalty:        A string. The norm used in the penalization for logistic regression model.
                            Can be 'l1', 'l2', 'elasticnet', 'none'.
                            Default = 'l2'
            dual:           A boolean.
                            Default = False.
            tol:            A float. Tolerance for stopping criteria.
                            Default = 1e-4
            verbose:        A boolean. Output additional information to the
                            terminal for functions with verbose feature.
                            Default = True

        """
        self.amine = amine
        if config:
            self.model = LogisticRegression(penalty=config['penalty'],
                                            dual=config['dual'],
                                            tol=config['tol'],
                                            C=config['C'],
                                            solver=config['solver'],
                                            max_iter=config['max_iter'])
        else:
            self.model = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0, solver='lbfgs', max_iter=6000)
        self.metrics = defaultdict(list)
        self.verbose = verbose
        self.stats_path = stats_path
        self.model_name = model_name

    # TODO: find out what to do with this part
    def load_dataset(self, training_batches, cross_validation_batches, meta=False):
        """TODO: Documentation

        """

        if meta is True:
            # option 2
            print("Conducting Training under Option 2.")
            self.x_t = cross_validation_batches[self.amine][0]
            self.y_t = cross_validation_batches[self.amine][1]
            self.x_v = cross_validation_batches[self.amine][2]
            self.y_v = cross_validation_batches[self.amine][3]

            self.all_data = np.concatenate((self.x_t, self.x_v))
            self.all_labels = np.concatenate((self.y_t, self.y_v))

        else:
            print("Conducting Training under Option 1.")
            self.x_t = training_batches[self.amine][0]
            self.y_t = training_batches[self.amine][1]
            self.x_v = cross_validation_batches[self.amine][0]
            self.y_v = cross_validation_batches[self.amine][1]

            self.all_data = self.x_v
            self.all_labels = self.y_v

        if self.verbose:
            print(f'The training data has dimension of {self.x_t.shape}.')
            print(f'The training labels has dimension of {self.y_t.shape}.')
            print(f'The testing data has dimension of {self.x_v.shape}.')
            print(f'The testing labels has dimension of {self.y_v.shape}.')

    def train(self):
        """Train the Random Forest model by setting up the ActiveLearner."""

        self.learner = ActiveLearner(estimator=self.model, X_training=self.x_t, y_training=self.y_t)
        # Evaluate zero-point performance
        self.evaluate()

    def active_learning(self, num_iter=None, to_params=True):
        """ The active learning loop

        This is the active learning model that loops around the Logistic Regression model
        to look for the most uncertain point and give the model the label to train

        Args:
            num_iter:   An integer that is the number of iterations.
                        Default = None
            to_params:  A boolean that decide if to store the metrics to the dictionary,
                        detail see "store_metrics_to_params" function.
                        Default = True

        return: N/A
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

        return: N/A
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
           cm:              A numpy array representing the confusion matrix given our predicted labels and the actual
                            corresponding labels. It's a 2x2 matrix for the drp_chem model.
            accuracy:       A float representing the accuracy rate of the model: the rate of correctly predicted reactions
                            out of all reactions.
            precision:      A float representing the precision rate of the model: the rate of the number of actually
                            successful reactions out of all the reactions predicted to be successful.
            recall:         A float representing the recall rate of the model: the rate of the number of reactions predicted
                            to be successful out of all the acutal successful reactions.
            bcr:            A float representing the balanced classification rate of the model. It's the average value of
                            recall rate and true negative rate.

        return: N/A
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
        """Save the data used to train, validate and test the model to designated folder
                Args:
                    k_shot:                 An integer representing the number of training samples per class.
                    n_way:                  An integer representing the number of classes per task.
                    meta:                   A boolean representing if it will be trained under option 1 or option 2.
                                                Option 1 is train with observations of other tasks and validate on the
                                                task-specific observations.
                                                Option 2 is to train and validate on the task-specific observations.
                Returns:
                    N/A
                """

        option = 1 if pretrain else 2

        dst_root = './results/LogisticRegression_few_shot/option_{0:d}'.format(option)

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
            print('No folder for Logistic Regression model storage found')
            print(f'Make folder to store Logistic Regression model at')

        model_folder = '{0:s}/LogisticRegression_{1:d}_shot_{2:d}_way_option_{3:d}_{4:s}'.format(dst_root,
                                                                                           train_size,
                                                                                           n_way,
                                                                                           option,
                                                                                           self.amine)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print('No folder for Logistic Regression model storage found')
            print(f'Make folder to store Logistic Regression model of amine {self.amine} at')
        else:
            print(f'Found existing folder. Model of amine {self.amine} will be stored at')
        print(model_folder)

        # Dump the model
        file_name = "LogisticRegression_{0:s}_option_{1:d}.pkl".format(self.amine, option)
        with open(os.path.join(model_folder, file_name), "wb") as f:
            pickle.dump(self, f)


def fine_tune(training_batches, cross_validation_batches, info=False):
    """Fine tune the model based on average bcr performance to find the best model hyper-parameters.
    Args:
        training_batches:       A dictionary representing the training batches used to train.
                                    See dataset.py for specific structure.
        cross_validation_batches:     A dictionary representing the training batches used to train.
                                    See dataset.py for specific structure.
        info:                A boolean. Setting it to True will make the function print out additional information
                                    during the fine-tuning stage.
                                    Default to False.
    Returns:
        best_option:            A dictionary representing the hyper-parameters that yields the best performance on
                                    average. For LogisticRegression, the current keys are: 'penalty', 'dual',
                                    'tol', 'C', 'solver', 'max_iter'
    """

    # Set all possible combinations

    params = {
        'penalty': ['l1','l2','elasticnet','none'],
        'dual': [True, False],
        'tol': [1e-3, 1e-4, 1e-5],
        'C': [.1 * i for i in range(3)],
        'solver': ['newton-cg','lbfgs','liblinear','sag','saga'],
        'max_iter': [4000, 5000, 6000, 7000]
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
        ALR = ActiveLogisticRegression(amine=amine, verbose=False)
        ALR.load_dataset(training_batches, cross_validation_batches)
        ALR.train()

        # Calculate AUC
        auc = roc_auc_score(ALR.all_labels, ALR.y_preds)

        base_accuracies.append(ALR.metrics['accuracies'][-1])
        base_precisions.append(ALR.metrics['precisions'][-1])
        base_recalls.append(ALR.metrics['recalls'][-1])
        base_bcrs.append(ALR.metrics['bcrs'][-1])
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
            ALR = ActiveLogisticRegression(amine=amine, config=option, verbose=False)
            ALR.load_dataset(training_batches, cross_validation_batches)
            ALR.train()

            # Calculate AUC
            auc = roc_auc_score(ALR.all_labels, ALR.y_preds)

            accuracies.append(ALR.metrics['accuracies'][-1])
            precisions.append(ALR.metrics['precisions'][-1])
            recalls.append(ALR.metrics['recalls'][-1])
            bcrs.append(ALR.metrics['bcrs'][-1])
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
    Returns:
        N/A
    """

    # Indicate which option we used the data for
    option = 1 if pretrain else 2

    # Set up the destination folder to save the data
    data_folder = './results/LogisticRegression_few_shot/option_{0:d}/data'.format(option)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print('No folder for LogisticRegression model data storage found')
        print('Make folder to store data used for LogisticRegression models at')
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
    file_name = "LogisticRegression_data.pkl"
    with open(os.path.join(data_folder, file_name), "wb") as f:
        pickle.dump(data, f)


def parse_args():
    """Set up the initial variables for running KNN.

    Retrieves argument values from the terminal command line to create an argparse.Namespace object for
        initialization.

    Args:
        N/A

    Returns:
        args: Namespace object with the following attributes:
            datasource:         A string identifying the datasource to be used, with default datasource set to drp_chem.
            k_shot:             An integer representing the number of training samples per class, with default set to 1.
            n_way:              An integer representing the number of classes per task, with default set to 1.
            num_batches:        An integer representing the number of meta-batches per task, with default set to 250.
            meta_batch_size:    An integer representing the number of tasks sampled per outer loop.
                                    It is set to 25 by default.
            train:              A train_flag attribute. Including it in the command line will set the train_flag to
                                    True by default.
            test:               A train_flag attribute. Including it in the command line will set the train_flag to
                                    False.
            cross_validate:     A boolean. Including it in the command line will run the model with cross-validation.
            pretrain:           A boolean representing if it will be trained under option 1 or option 2.
                                    Option 1 is train with observations of other tasks and validate on the
                                    task-specific observations.
                                    Option 2 is to train and validate on the task-specific observations.
            verbose:            A boolean. Including it in the command line will output additional information to the
                                    terminal for functions with verbose feature.
    """

    parser = argparse.ArgumentParser(description='Setup variables for active learning RandomForest.')
    parser.add_argument('--datasource', type=str, default='drp_chem', help='datasource to be used')

    parser.add_argument('--train_size', dest='train_size', default=1, help='number of samples used for training')

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--cross_validate', action='store_true', help='use cross-validation for training')
    parser.add_argument('--pretrain', action='store_true', help='load the dataset under option 1. Not include this will'
                                                                ' load the dataset under option 2. See documentation in'
                                                                ' codes for details.')
    parser.add_argument('--full', action='store_true', help='load the full dataset or the test sample dataset')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    return args


def run_model(LogisticRegression_params):
    """Full-scale training, validation and testing using all amines.
    Args:
        LogisticRegression_params:         A dictionary of the parameters for the LogisticRegression model.
                                    See initialize() for more information.
    Returns:
        N/A
    """
    # Unload parameters
    config = LogisticRegression_params['config']
    cross_validation = LogisticRegression_params['cross_validate']
    verbose = LogisticRegression_params['verbose']
    pretrain = LogisticRegression_params['pretrain']
    stats_path = LogisticRegression_params['stats_path']
    model_name = LogisticRegression_params['model_name']
    print(model_name)

    # Set up if we want to load the meta dataset for option 2 or not
    meta = not pretrain
    # Set up the number of samples used for training under option 2
    train_size = LogisticRegression_params['train_size']
    # Specify the desired operation
    fine_tuning = LogisticRegression_params['fine_tuning']
    if fine_tuning:
        ft_training_batches, ft_validation_batches, ft_testing_batches, ft_counts = import_full_dataset(
            train_size, meta_batch_size=25, num_batches=250, verbose=verbose, cross_validation=cross_validation,
            meta=False)
        best_config = fine_tune(ft_training_batches, ft_validation_batches, info=True)
    else:
        # Load the full dataset for training and validation
        if LogisticRegression_params['full_dataset']:
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
            ALR = ActiveLogisticRegression(amine=amine, config=config, verbose=verbose, stats_path=stats_path,
                                     model_name=model_name)
            # Load the training and validation set into the model
            ALR.load_dataset(training_batches, validation_batches, meta)
            # Train the data on the training set
            ALR.train()
            # Conduct active learning with all the observations available in the pool
            ALR.active_learning(to_params=True)
            # Save the model for future reproducibility
            ALR.save_model(train_size, 2, pretrain)
            # TODO: testing part not implemented


def main():
    """Main driver function"""

    # This converts the args into a dictionary
    LogisticRegression_params = vars(parse_args())

    run_model(LogisticRegression_params)


if __name__ == "__main__":
    main()
