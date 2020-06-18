import os
import pickle
import argparse


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from modAL.models import ActiveLearner

import dataset

'''def parse_args():
    parser = argparse.ArgumentParser(
        description='Setup variables for RandomForest.')
    parser.add_argument('--meta_batch_size', type=int, default=25,
                        help='Number of tasks sampled per outer loop')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='Number of training samples per class')
    parser.add_argument('--num_batches', type=int, default=250,
                        help='Number of tasks sampled per inner loop')'''


class ActiveRandomForest:
    """
    A class of Random Forest model with active learning
    """

    def __init__(self, amine, n_estimator=100, criterion="gini", max_depth=7, verbose=True):
        """initialization of the class

        Args:
            n_estimator:    An integer. The number of estimators in the random forest model.
                            Default = 100
            criterion:      A string. The criterion used for the random forest model,
                            can be "gini" or "entropy".
                            Default = "gini".
            max_depth:      An integer. The max_depth of the decision tress in the random forest model.
                            Default = 7
            verbose:        A boolean. Output additional information to the
                            terminal for functions with verbose feature.
                            Default = True

        """
        self.amine = amine
        self.n_estimator = n_estimator
        self.criterion = criterion
        self.max_depth = max_depth
        self.model = RandomForestClassifier(n_estimators=self.n_estimator,criterion=self.criterion, max_depth=self.max_depth)
        self.metrics = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'bcrs': [],
            'confusion_matrices': []
        }
        self.verbose = verbose

    # TODO: should we load dataset here? what if we are using the model for another dataset
    def load_dataset(self, training_batches, cross_validation_batches, meta=False):
        """TODO: Change this to accommodate drp+chem dataset

        TODO: Documentation

        """
        if meta is True:
            # option 2
            print("Conducting Training under Option 2.")
            self.x_t = cross_validation_batches[self.amine][0]
            self.y_t = cross_validation_batches[self.amine][1]
            self.x_v = cross_validation_batches[self.amine][2]
            self.y_v = cross_validation_batches[self.amine][3]

            self.all_data = np.concatenate((self.x_t,self.x_v))
            self.all_labels = np.concatenate((self.y_t,self.y_v))

        else:
            print("Conducting Training under Option 1.")
            self.x_t = training_batches[self.amine][0]
            self.y_t = training_batches[self.amine][1]
            self.x_v = cross_validation_batches[self.amine][0]
            self.y_v = cross_validation_batches[self.amine][1]

            self.all_data = np.concatenate((self.x_t, self.x_v))
            self.all_labels = np.concatenate((self.y_t, self.y_v))

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

        This is the active learning model that loops around the Random Forest model
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

        model = "RandomForest"

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


if __name__ == "__main__":
    # parser the under params
    meta_batch_size = 10
    k_shot = 20
    num_batches = 10
    verbose = True
    cross_validation = True
    meta = False

    training_batches, validation_batches, testing_batches, counts = dataset.import_full_dataset(
        k_shot, meta_batch_size, num_batches, verbose=verbose, cross_validation=cross_validation, meta=meta)

    for amine in training_batches:
        print("Training and cross validation on {} amine.".format(amine))
        ARF = ActiveRandomForest(amine, n_estimator=100, criterion="gini", max_depth=8)
        ARF.load_dataset(training_batches,validation_batches, meta=meta)
        ARF.train()
        ARF.active_learning(to_params=False)
