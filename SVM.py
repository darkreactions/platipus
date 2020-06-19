import itertools
import os
import pickle

import dataset
import numpy as np
from modAL.models import ActiveLearner
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


class ActiveSVM:
    """A SVC machine learning model using active learning with modAL package

    Attributes: TODO: more attributes
        model:          A KNeighborClassifier object as the classifier model given the number of neighbors to classify
                            with.
        metrics:        A dictionary to store the performance metrics locally. It has the format of
                            {'metric_name': [metric_value]}.
        verbose:        A boolean representing whether it will prints out additional information to the terminal or not.
        all_data:       A numpy array representing all the data from the dataset.
        all_labels:     A numpy array representing all the labels from the dataset.
        x_t:            A numpy array representing the training data used for model training.
        y_t:            A numpy array representing the training labels used for model training.
        x_v:            A numpy array representing the testing data used for active learning.
        y_v:            A numpy array representing the testing labels used for active learning.
        learner:        An ActiveLearner to conduct active learning with. See modAL documentation for more details.
    """

    def __init__(self, amine, option=None, verbose=True):
        """TODO: Documentation

        """
        self.amine = amine

        if option:
            if option['kernel'] == 'poly':
                self.model = CalibratedClassifierCV(SVC(
                    C=option['C'],
                    kernel=option['kernel'],
                    degree=option['degree'],
                    gamma=option['gammas'],
                    shrinking=option['shrinking'],
                    tol=option['tol'],
                    decision_function_shape=option['decision_function_shape'],
                    break_ties=option['break_ties']
                ))

            else:
                self.model = CalibratedClassifierCV(SVC(
                    C=option['C'],
                    kernel=option['kernel'],
                    gamma=option['gammas'],
                    shrinking=option['shrinking'],
                    tol=option['tol'],
                    decision_function_shape=option['decision_function_shape'],
                    break_ties=option['break_ties']
                ))
        else:
            # Simple baseline model
            self.model = CalibratedClassifierCV(SVC())

        self.metrics = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'bcrs': [],
            'confusion_matrices': []
        }

        self.verbose = verbose

    def load_dataset(self, training_batches, cross_validation_batches, meta=False):
        """TODO: Change this to accommodate drp+chem dataset

        TODO: Documentation

        """

        if meta is True:
            # option 2
            self.x_t = cross_validation_batches[self.amine][0]
            self.y_t = cross_validation_batches[self.amine][1]
            self.x_v = cross_validation_batches[self.amine][2]
            self.y_v = cross_validation_batches[self.amine][3]

            self.all_data = np.concatenate((self.x_t, self.x_v))
            self.all_labels = np.concatenate((self.y_t, self.y_v))

            if self.verbose:
                print("Conducting Training under Option 2.")

        else:
            self.x_t = training_batches[self.amine][0]
            self.y_t = training_batches[self.amine][1]
            self.x_v = cross_validation_batches[self.amine][0]
            self.y_v = cross_validation_batches[self.amine][1]

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

        return: N/A
        """
        num_iter = num_iter if num_iter else self.x_v.shape[0]

        for _ in range(num_iter):
            # TODO: Comment
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

        # Find model predictions
        self.y_preds = self.learner.predict(self.all_data)

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

        model = 'SVM'

        with open(os.path.join("./data", "cv_statistics.pkl"), "rb") as f:
            stats_dict = pickle.load(f)

        stats_dict[model]['accuracies'].append(self.metrics['accuracies'])
        stats_dict[model]['confusion_matrices'].append(self.metrics['confusion_matrices'])
        stats_dict[model]['precisions'].append(self.metrics['precisions'])
        stats_dict[model]['recalls'].append(self.metrics['recalls'])
        stats_dict[model]['bcrs'].append(self.metrics['bcrs'])

        # Save this dictionary in case we need it later
        with open(os.path.join("./data", "cv_statistics.pkl"), "wb") as f:
            pickle.dump(stats_dict, f)

    def save_model(self, k_shot, n_way, meta):
        """TODO: Documentation"""
        option = 2 if meta else 1

        dst_root = './SVM_few_shot/option_{0:d}'.format(option)

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
            print('No folder for SVM model storage found')
            print(f'Make folder to store SVM model at')

        model_folder = '{0:s}/SVM_{1:d}_shot_{2:d}_way_option_{3:d}_{4:s}'.format(dst_root,
                                                                                  k_shot,
                                                                                  n_way,
                                                                                  option,
                                                                                  self.amine)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print('No folder for SVM model storage found')
            print(f'Make folder to store SVM model of amine {self.amine} at')
        else:
            print(f'Found existing folder. Model of amine {self.amine} will be stored at')
        print(model_folder)

        # Dump the model
        file_name = "SVM_{0:s}_option_{1:d}.pkl".format(self.amine, option)
        with open(os.path.join(model_folder, file_name), "wb") as f:
            pickle.dump(self, f)

    # TODO: Unofficial
    def __str__(self):
        return 'A SVM model for {0:s} using active learning'.format(self.amine)


def fine_tune(training_batches, cross_validation_batches, verbose=False):
    """TODO: Documentation and comments

    """

    # Set all possible combinations
    params = {
        'C': [.1, 1.0, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [0, 1, 2, 3, 4, 5, 6],
        'gammas': ['scale', 'auto'],
        'shrinking': [True, False],
        'tol': [1e-3, 1e-2, .1, 1, 10],
        'decision_function_shape': ['ovo', 'ovr'],
        'break_ties': [True, False]
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

    for amine in training_batches:
        ASVM = ActiveSVM(amine=amine, verbose=verbose)
        ASVM.load_dataset(training_batches, cross_validation_batches, meta=meta)
        ASVM.train()

        base_accuracies.append(ASVM.metrics['accuracies'][-1])
        base_precisions.append(ASVM.metrics['precisions'][-1])
        base_recalls.append(ASVM.metrics['recalls'][-1])
        base_bcrs.append(ASVM.metrics['bcrs'][-1])

    base_avg_accuracy = sum(base_accuracies) / len(base_accuracies)
    base_avg_precision = sum(base_precisions) / len(base_precisions)
    base_avg_recall = sum(base_recalls) / len(base_recalls)
    base_avg_bcr = sum(base_bcrs) / len(base_bcrs)

    if verbose:
        print(f'Baseline average bcr is {base_avg_bcr}')
    best_bcr = base_avg_bcr
    best_option = {}

    option_no = 1

    for option in combinations:
        accuracies = []
        precisions = []
        recalls = []
        bcrs = []

        # print(f'Trying option {option_no}')
        for amine in training_batches:
            # print("Training and cross validation on {} amine.".format(amine))
            ASVM = ActiveSVM(amine=amine, option=option, verbose=verbose)
            ASVM.load_dataset(training_batches, cross_validation_batches, meta=meta)
            ASVM.train()

            accuracies.append(ASVM.metrics['accuracies'][-1])
            precisions.append(ASVM.metrics['precisions'][-1])
            recalls.append(ASVM.metrics['recalls'][-1])
            bcrs.append(ASVM.metrics['bcrs'][-1])

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_bcr = sum(bcrs) / len(bcrs)

        if avg_bcr > best_bcr:
            best_bcr = avg_bcr
            best_option = option
            if verbose:
                print(f'The current average accuracy is {avg_accuracy} vs. the base accuracy {base_avg_accuracy}')
                print(f'The current average precision is {avg_precision} vs. the base precision {base_avg_precision}')
                print(f'The current average recall rate is {avg_recall} vs. the base recall rate {base_avg_recall}')
                print(f'The best average bcr by this setting is {avg_bcr} vs. the base bcr {base_avg_bcr}')
                print(f'The current best setting for amine {amine} is {best_option}')
                print()

        option_no += 1

    if verbose:
        print()
        print(f'The best setting for all amines is {best_option}')
        print(f'With an average bcr of {best_bcr}')

    return best_option


def save_used_data(training_batches, validation_batches, testing_batches, counts, meta):
    """TODO: Dcoumentation"""

    option = 2 if meta else 1

    data_folder = './SVM_few_shot/option_{0:d}/data'.format(option)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print('No folder for SVM model data storage found')
        print('Make folder to store data used for SVM models at')
    else:
        print('Found existing folder. Data used for models will be stored at')
    print(data_folder)

    data = {
        'training_batches': training_batches,
        'validation_batches': validation_batches,
        'testing_batches': testing_batches,
        'counts': counts
    }

    file_name = "SVM_data.pkl"
    with open(os.path.join(data_folder, file_name), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    meta_batch_size = 10
    k_shot = 20
    num_batches = 10
    verbose = True
    cross_validation = True
    meta = True

    # training_batches, cross_validation_batches, testing_batches, counts = dataset.import_test_dataset(k_shot, meta_batch_size, num_batches, verbose=verbose, cross_validation=cross_validation, meta=meta)

    # best_hyper_params = fine_tune(training_batches, cross_validation_batches)

    training_batches, cross_validation_batches, testing_batches, counts = dataset.import_full_dataset(
        k_shot, meta_batch_size, num_batches, verbose=verbose, cross_validation=cross_validation, meta=meta)

    save_used_data(training_batches, cross_validation_batches, testing_batches, counts, meta)

    for amine in training_batches:
        print("Training and cross validation on {} amine.".format(amine))
        ASVM = ActiveSVM(amine=amine, option=None)  # TODO: fine tune
        ASVM.load_dataset(training_batches, cross_validation_batches, meta=meta)
        ASVM.train()
        ASVM.active_learning(to_params=False)  # TODO: delete this when running full models
        ASVM.save_model(k_shot=k_shot, n_way=2, meta=meta)
