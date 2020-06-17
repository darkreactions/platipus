import os
import pickle

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from modAL.models import ActiveLearner


class ActiveSVM:
    """A SVC machine learning model using active learning with modAL package

    Attributes:
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

    def __init__(self, verbose=True):
        """TODO: Documentation

        """
        self.model = CalibratedClassifierCV(SVC())
        self.metrics = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'bcrs': [],
            'confusion_matrices': []
        }
        self.verbose = verbose

    '''def load_dataset(self, data, labels, test_size=None, random_state=None):
        """TODO: Documentation

        """
        print('Loading and normalizing dataset')
        scaler = StandardScaler()
        scaler.fit(data)
        normalized_data = scaler.transform(data)

        self.all_data = normalized_data
        self.all_labels = labels

        self.x_t, self.x_v, self.y_t, self.y_v = train_test_split(
            self.all_data,
            self.all_labels,
            test_size=test_size,
            random_state=random_state)

        if self.verbose:
            print(f'The training data has dimension of {self.x_t.shape}.')
            print(f'The training labels has dimension of {self.y_t.shape}.')
            print(f'The testing data has dimension of {self.x_v.shape}.')
            print(f'The testing labels has dimension of {self.y_v.shape}.')'''
    def load_dataset(self, amine, amine_left_out_batches, amine_cross_validate_samples, meta=False):
        """TODO: Change this to accommodate drp+chem dataset

        TODO: Documentation

        """


        if meta == True:
            # option 2
            pass

        else:
            self.x_t = amine_left_out_batches[amine][0]
            self.y_t = amine_left_out_batches[amine][1]
            self.x_v = amine_cross_validate_samples[amine][0]
            self.y_v = amine_cross_validate_samples[amine][1]

            self.all_data = np.concatenate((self.x_t, self.x_v))
            self.all_labels = np.concatenate((self.y_t, self.y_v))

        if self.verbose:
            print(f'The training data has dimension of {self.x_t.shape}.')
            print(f'The training labels has dimension of {self.y_t.shape}.')
            print(f'The testing data has dimension of {self.x_v.shape}.')
            print(f'The testing labels has dimension of {self.y_v.shape}.')


    def train(self):
        """ Train the SVM model by setting up the ActiveLearner.

        """
        self.learner = ActiveLearner(estimator=self.model, X_training=self.x_t, y_training=self.y_t)
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

        preds = self.learner.predict(self.all_data)
        cm = confusion_matrix(self.all_labels, preds)

        # TODO: Also in sklearn package
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
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

    # TODO: Unofficial
    def __str__(self):
        return 'A SVM model using active learning'


if __name__ == "__main__":
    '''iris = load_iris()
    X = iris['data']
    y = iris['target']'''
    meta_batch_size = 10
    k_shot = 20
    num_batches = 10
    amine_left_out_batches, amine_cross_validate_samples, amine_test_samples, counts = dataset.import_full_dataset(
        k_shot, meta_batch_size, num_batches, verbose=True, cross_validation=True, meta=True)
    ASVM = ActiveSVM()
    '''print(amine_left_out_batches)
    amine = amine_left_out_batches.keys()
    print(amine[0])
    x_t = amine_left_out_batches[amine][0]
    y_t = amine_left_out_batches[amine][1]
    x_v = amine_cross_validate_samples[amine][0]
    y_v = amine_cross_validate_samples[amine][1]'''
    for amine in amine_left_out_batches:
        print("testing on {}".format(amine))
        ASVM.load_dataset(amine, amine_left_out_batches, amine_cross_validate_samples, meta=True)
        ASVM.train()
        ASVM.active_learning(to_params=False)
