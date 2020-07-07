"""
Run the following command lines for full dataset training:
python SVM.py --datasource=drp_chem --train_size=20 --cross_validate --full --verbose
python SVM.py --datasource=drp_chem --train_size=20 --cross_validate --pretrain --full --verbose

Run the following command lines for test dataset training (debug):
python SVM.py --datasource=drp_chem ---train_size=20 -cross_validate --verbose
python SVM.py --datasource=drp_chem --train_size=20 --cross_validate --pretrain --verbose
"""

import argparse
import itertools
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.svm import SVC

from modAL.models import ActiveLearner

from utils.dataset import process_dataset


class ActiveSVM:
    """A SVC machine learning model using active learning with modAL package

    Attributes: 
        amine:          A string representing the amine this model is used for.
        model:          A CalibratedClassifierCV + SVM object as the classifier model.
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

    def __init__(self, amine=None, config=None,
                 verbose=True, stats_path=Path('./results/stats.pkl'),
                 model_name='SVM'):
        """Initialization of the ActiveSVM model"""

        self.amine = amine

        # Load customized model or use the default fine-tuned setting
        if config:
            if config['kernel'] == 'poly':
                self.model = CalibratedClassifierCV(SVC(
                    C=config['C'],
                    kernel=config['kernel'],
                    degree=config['degree'],
                    gamma=config['gammas'],
                    shrinking=config['shrinking'],
                    tol=config['tol'],
                    decision_function_shape=config['decision_function_shape'],
                    break_ties=config['break_ties'],
                    class_weight=config['class_weight']
                ))

            else:
                self.model = CalibratedClassifierCV(SVC(
                    C=config['C'],
                    kernel=config['kernel'],
                    gamma=config['gammas'],
                    shrinking=config['shrinking'],
                    tol=config['tol'],
                    decision_function_shape=config['decision_function_shape'],
                    break_ties=config['break_ties'],
                    class_weight=config['class_weight']
                ))
        else:
            # Fine tuned model
            self.model = CalibratedClassifierCV(SVC(
                C=.003,
                kernel='poly',
                degree=3,
                gamma='scale',
                shrinking=True,
                tol=1,
                decision_function_shape='ovo',
                break_ties=True,
                class_weight=None
            ))

        self.metrics = defaultdict(list)
        self.verbose = verbose
        self.stats_path = stats_path
        self.model_name = model_name

    def load_dataset(self, x_t, y_t, x_v, y_v, all_data, all_labels):
        """Load the input training and validation data and labels into the model.

        Args:
            x_t:                A 2-D numpy array representing the training data.
            y_t:                A 2-D numpy array representing the training labels.
            x_v:                A 2-D numpy array representing the validation data.
            y_v:                A 2-D numpy array representing the validation labels.
            all_data:           A 2-D numpy array representing all the data in the active learning pool.
            all_labels:         A 2-D numpy array representing all the labels in the active learning pool.
        """

        self.x_t, self.y_t, self.x_v, self.y_v = x_t, y_t, x_v, y_v

        self.all_data = all_data
        self.all_labels = all_labels

        if self.verbose:
            print(f'The training data has dimension of {self.x_t.shape}.')
            print(f'The training labels has dimension of {self.y_t.shape}.')
            print(f'The testing data has dimension of {self.x_v.shape}.')
            print(f'The testing labels has dimension of {self.y_v.shape}.')

    def train(self):
        """ Train the SVM model by setting up the ActiveLearner.

        """
        self.learner = ActiveLearner(estimator=self.model, X_training=self.x_t, y_training=self.y_t)
        # Evaluate zero-point performance
        self.evaluate()

    def active_learning(self, num_iter=None, to_params=True):
        """ The active learning loop

        This is the active learning model that loops around the SVM model
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

        # Find model predictions
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
        stats_dict[model]['confusion_matrices'].append(
            self.metrics['confusion_matrices'])
        stats_dict[model]['precisions'].append(self.metrics['precisions'])
        stats_dict[model]['recalls'].append(self.metrics['recalls'])
        stats_dict[model]['bcrs'].append(self.metrics['bcrs'])

        # Save this dictionary in case we need it later
        with open(self.stats_path, "wb") as f:
            pickle.dump(stats_dict, f)

    def save_model(self, model_name):
        """Save the data used to train, validate and test the model to designated folder

        Args:
            model_name:         A string representing the name of the model.
        """

        # Set up the main destination folder for the model
        dst_root = './data/SVM/{0:s}'.format(model_name)
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
            print(f'No folder for SVM model {model_name} storage found')
            print(f'Make folder to store model at')

        # Dump the model into the designated folder
        file_name = "{0:s}_{1:s}.pkl".format(model_name, self.amine)
        with open(os.path.join(dst_root, file_name), "wb") as f:
            pickle.dump(self, f)

    def __str__(self):
        return 'A SVM model for {0:s} using active learning'.format(self.amine)


def parse_args():
    """Set up the initial variables for running SVM.

    Retrieves argument values from the terminal command line to create an argparse.Namespace object for
        initialization.

    Args:
        N/A

    Returns:
        args: Namespace object with the following attributes:
            datasource:         A string identifying the datasource to be used, with default datasource set to drp_chem.
            train_size          An integer representing the number of samples uses for training after pre-training.
                                    Default set to 10.
            pre_learn_size      An integer representing the number of samples used for training before active learning.
                                    Default set to 10.
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

    parser.add_argument('--train_size', dest='train_size', default=10, help='number of samples used for training after '
                                                                            'pre-training')
    parser.add_argument('--pre_learn_size', dest='pre_learn_size', default=10, help='number of samples used for '
                                                                                    'training before active learning')

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


def fine_tune(info=False):
    """Fine tune the model based on average bcr performance to find the best model hyper-parameters.

    Args:
        info:                A boolean. Setting it to True will make the function print out additional information
                                    during the fine-tuning stage.
                                    Default to False.

    Returns:
        best_option:            A dictionary representing the hyper-parameters that yields the best performance on
                                    average. For SVM, the current keys are: 'C', 'kernel', 'degree', 'gammas',
                                    'shrinking', 'tol', 'decision_function_shape', 'break_ties', 'class_weight'.
    """

    # Set all possible combinations
    params = {
        'C': [i / 1000 for i in range(1, 51)],
        'kernel': ['poly', 'sigmoid'],
        'degree': [0, 1, 2, 3],
        'gammas': ['scale'],
        'shrinking': [True],
        'tol': [.1, 1, 2],
        'decision_function_shape': ['ovo'],
        'break_ties': [True],
        'class_weight': [None]
    }

    combinations = []

    keys, values = zip(*params.items())
    for bundle in itertools.product(*values):
        combinations.append(dict(zip(keys, bundle)))

    if info:
        print(f'There are {len(combinations)} many combinations to try.')

    # Load the full dataset under option 1
    amine_list, train_data, train_labels, val_data, val_labels, all_data, all_labels = process_dataset(
        train_size=10,
        active_learning_iter=10,
        verbose=False,
        cross_validation=True,
        full=True,
        active_learning=False,
        w_hx=True,
        w_k=False
    )

    # Set baseline performance
    base_accuracies = []
    base_precisions = []
    base_recalls = []
    base_bcrs = []
    base_aucs = []

    for amine in amine_list:
        # Create model for a specific amine
        ASVM = ActiveSVM(amine=amine, verbose=False)

        x_t, y_t = train_data[amine], train_labels[amine]
        x_v, y_v = val_data[amine], val_labels[amine]
        all_task_data, all_task_labels = all_data[amine], all_labels[amine]

        # Load the training and validation set into the model
        ASVM.load_dataset(x_t, x_v, y_t, y_v, all_task_data, all_task_labels)

        ASVM.train()

        # Calculate AUC
        auc = roc_auc_score(ASVM.all_labels, ASVM.y_preds)

        base_accuracies.append(ASVM.metrics['accuracies'][-1])
        base_precisions.append(ASVM.metrics['precisions'][-1])
        base_recalls.append(ASVM.metrics['recalls'][-1])
        base_bcrs.append(ASVM.metrics['bcrs'][-1])
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
    for option in combinations:
        accuracies = []
        precisions = []
        recalls = []
        bcrs = []
        aucs = []

        if info:
            print(f'Trying option {option_no}')

        for amine in amine_list:
            # Create model for a specific amine
            ASVM = ActiveSVM(amine=amine, config=option, verbose=False)

            # Exact and load the training and validation set into the model
            x_t, y_t = train_data[amine], train_labels[amine]
            x_v, y_v = val_data[amine], val_labels[amine]
            all_task_data, all_task_labels = all_data[amine], all_labels[amine]

            ASVM.load_dataset(x_t, x_v, y_t, y_v, all_task_data, all_task_labels)

            ASVM.train()

            # Calculate AUC
            auc = roc_auc_score(ASVM.all_labels, ASVM.y_preds)

            accuracies.append(ASVM.metrics['accuracies'][-1])
            precisions.append(ASVM.metrics['precisions'][-1])
            recalls.append(ASVM.metrics['recalls'][-1])
            bcrs.append(ASVM.metrics['bcrs'][-1])
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


def run_model(SVM_params):
    """Full-scale training, validation and testing using all amines.

    Args:
        SVM_params:         A dictionary of the parameters for the SVM model.
                                See initialize() for more information.
     """
    
    # Unload common parameters
    config = SVM_params['config']
    verbose = SVM_params['verbose']
    stats_path = SVM_params['stats_path']

    model_name = SVM_params['model_name']
    print(f'Running model {model_name}')

    # Unload the training data specific parameters
    train_size = SVM_params['train_size']
    active_learning_iter = SVM_params['active_learning_iter']
    cross_validation = SVM_params['cross_validate']
    full = SVM_params['full_dataset']
    active_learning = SVM_params['active_learning']
    w_hx = SVM_params['with_historical_data']
    w_k = SVM_params['with_k']

    # Specify the desired operation
    fine_tuning = SVM_params['fine_tuning']
    save_model = SVM_params['save_model']
    to_params = True

    if fine_tuning:
        best_config = fine_tune(info=True)
    else:
        # Load the desired sized dataset under desired option
        amine_list, x_t, y_t, x_v, y_v, all_data, all_labels = process_dataset(
            train_size=train_size,
            active_learning_iter=active_learning_iter,
            verbose=verbose,
            cross_validation=cross_validation,
            full=full,
            active_learning=active_learning,
            w_hx=w_hx,
            w_k=w_k
        )

        # print(amine_list)
        for amine in amine_list:
            if cross_validation:
                print("Training and cross validation on {} amine.".format(amine))

                # Create the SVM model instance for the specific amine
                ASVM = ActiveSVM(amine=amine, config=config, verbose=verbose, stats_path=stats_path, model_name=model_name)

                # Load the training and validation set into the model
                ASVM.load_dataset(x_t[amine], y_t[amine], x_v[amine], y_v[amine], all_data[amine], all_labels[amine])

                # Train the data on the training set
                ASVM.train()

                # Conduct active learning with all the observations available in the pool
                if active_learning:
                    ASVM.active_learning(num_iter=active_learning_iter, to_params=to_params)
                else:
                    ASVM.store_metrics_to_params()

                # Save the model for future reproducibility
                if save_model:
                    ASVM.save_model(model_name)

            # TODO: testing part not implemented: might need to change the logic loading things in


def main():
    """Main driver function"""

    # This converts the args into a dictionary
    SVM_params = vars(parse_args())

    run_model(SVM_params)


if __name__ == "__main__":
    main()