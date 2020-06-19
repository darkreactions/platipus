import itertools
import os
import pickle

import dataset
import numpy as np
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

    def __init__(self, amine, option=None, verbose=True):
        """initialization of the class

        Args:
            amine:          A string representing the amine this model is used for.
            option:         A dictionary representing the hyper-parameters chosen. For RandomForest, the keys are:
                                'n_estimators', 'criterion', 'max_depth', 'max_features', 'bootstrap',
                                'min_samples_leaf', 'min_samples_split', 'ccp_alpha'.
                                Default = None
            verbose:        A boolean. Output additional information to the
                            terminal for functions with verbose feature.
                            Default = True
        """

        self.amine = amine

        if option:
            self.model = RandomForestClassifier(
                n_estimators=option['n_estimators'],
                criterion=option['criterion'],
                max_depth=option['max_depth'],
                min_samples_split=option['min_samples_split'],
                min_samples_leaf=option['min_samples_leaf'],
                max_features=option['max_features'],
                bootstrap=option['bootstrap'],
                ccp_alpha=option['ccp_alpha']
            )
        else:
            # Simple baseline model
            self.model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=7)

        self.metrics = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'bcrs': [],
            'confusion_matrices': []
        }

        self.verbose = verbose

    def load_dataset(self, training_batches, cross_validation_batches, meta=False):
        """TODO: Documentation

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
            # Option 1
            self.x_t = training_batches[self.amine][0]
            self.y_t = training_batches[self.amine][1]
            self.x_v = cross_validation_batches[self.amine][0]
            self.y_v = cross_validation_batches[self.amine][1]

            self.all_data = np.concatenate((self.x_t, self.x_v))
            self.all_labels = np.concatenate((self.y_t, self.y_v))

            if self.verbose:
                print("Conducting Training under Option 1.")

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

    def save_model(self, k_shot, n_way, meta):
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

        option = 2 if meta else 1

        dst_root = './RandomForest_few_shot/option_{0:d}'.format(option)

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
            print('No folder for RandomForest model storage found')
            print(f'Make folder to store RandomForest model at')

        model_folder = '{0:s}/RandomForest_{1:d}_shot_{2:d}_way_option_{3:d}_{4:s}'.format(dst_root,
                                                                                           k_shot,
                                                                                           n_way,
                                                                                           option,
                                                                                           self.amine)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print('No folder for RandomForest model storage found')
            print(f'Make folder to store RandomForest model of amine {self.amine} at')
        else:
            print(f'Found existing folder. Model of amine {self.amine} will be stored at')
        print(model_folder)

        # Dump the model
        file_name = "RandomForest_{0:s}_option_{1:d}.pkl".format(self.amine, option)
        with open(os.path.join(model_folder, file_name), "wb") as f:
            pickle.dump(self, f)


def fine_tune(training_batches, cross_validation_batches, verbose=False):
    """Fine tune the model based on average bcr performance to find the best model hyper-parameters.

    Args:
        training_batches:       A dictionary representing the training batches used to train.
                                    See dataset.py for specific structure.
        validation_batches:     A dictionary representing the training batches used to train.
                                    See dataset.py for specific structure.
        verbose:                A boolean. Setting it to True will make the function print out additional information
                                    during the fine-tuning stage.
                                    Default to False.

    Returns:
        best_option:            A dictionary representing the hyper-parameters that yields the best performance on
                                    average. For RandomForest, the current keys are: 'n_estimators', 'criterion',
                                    'max_depth', 'max_features', 'bootstrap', 'min_samples_leaf', 'min_samples_split',
                                    'ccp_alpha'.
    """

    # Set all possible combinations
    params = {
        'n_estimators': [10, 50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'criterion': ['gini', 'entropy'],
        'max_depth': [i for i in range(1, 11)],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'bootstrap': [True, False],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'ccp_alpha': [.1 * i for i in range(10)]
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
        ASVM = ActiveRandomForest(amine=amine, verbose=verbose)
        ASVM.load_dataset(training_batches, cross_validation_batches, meta=meta)
        ASVM.train()

        base_accuracies.append(ASVM.metrics['accuracies'][-1])
        base_precisions.append(ASVM.metrics['precisions'][-1])
        base_recalls.append(ASVM.metrics['recalls'][-1])
        base_bcrs.append(ASVM.metrics['bcrs'][-1])

    # Calculated the average baseline performances
    base_avg_accuracy = sum(base_accuracies) / len(base_accuracies)
    base_avg_precision = sum(base_precisions) / len(base_precisions)
    base_avg_recall = sum(base_recalls) / len(base_recalls)
    base_avg_bcr = sum(base_bcrs) / len(base_bcrs)

    if verbose:
        print(f'Baseline average bcr is {base_avg_bcr}')
    best_bcr = base_avg_bcr
    best_option = {}

    # Try out each possible combinations of hyper-parameters
    print(f'There are {len(combinations)} many combinations to try.')
    for option in combinations:
        accuracies = []
        precisions = []
        recalls = []
        bcrs = []

        for amine in training_batches:
            # print("Training and cross validation on {} amine.".format(amine))
            ASVM = ActiveRandomForest(amine=amine, option=option, verbose=verbose)
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
            if verbose:
                print(f'The previous best option was {best_option}')
            best_bcr = avg_bcr
            best_option = option
            if verbose:
                print(f'The current average accuracy is {avg_accuracy} vs. the base accuracy {base_avg_accuracy}')
                print(f'The current average precision is {avg_precision} vs. the base precision {base_avg_precision}')
                print(f'The current average recall rate is {avg_recall} vs. the base recall rate {base_avg_recall}')
                print(f'The best average bcr by this setting is {avg_bcr} vs. the base bcr {base_avg_bcr}')
                print(f'The current best setting for amine {amine} is {best_option}')

    if verbose:
        print()
        print(f'The best setting for all amines is {best_option}')
        print(f'With an average bcr of {best_bcr}')

    return best_option


def save_used_data(training_batches, validation_batches, testing_batches, counts, meta):
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
    option = 2 if meta else 1

    # Set up the destination folder to save the data
    data_folder = './RandomForest_few_shot/option_{0:d}/data'.format(option)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print('No folder for RandomForest model data storage found')
        print('Make folder to store data used for RandomForest models at')
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
    file_name = "RandomForest_data.pkl"
    with open(os.path.join(data_folder, file_name), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # parser the under params
    meta_batch_size = 10
    k_shot = 20
    num_batches = 10
    verbose = True
    cross_validation = True
    meta = True

    """
    training_batches, cross_validation_batches, testing_batches, counts = dataset.import_test_dataset(k_shot,
                                                                                                      meta_batch_size,
                                                                                                      num_batches,
                                                                                                      verbose=verbose,
                                                                                                      cross_validation=cross_validation,
                                                                                                      meta=meta)

    best_hyper_params = fine_tune(training_batches=training_batches, cross_validation_batches=cross_validation_batches)
    """

    # training_batches, validation_batches, testing_batches, counts = dataset.import_full_dataset(
    # k_shot, meta_batch_size, num_batches, verbose=verbose, cross_validation=cross_validation, meta=meta)

    training_batches, validation_batches, testing_batches, counts = dataset.import_full_dataset(k_shot, meta_batch_size,
                                                                                                num_batches,
                                                                                                verbose=verbose,
                                                                                                cross_validation=cross_validation,
                                                                                                meta=meta)

    save_used_data(training_batches, validation_batches, testing_batches, counts, meta)

    # best_hyper_params = fine_tune(training_batches=training_batches, cross_validation_batches=validation_batches)

    for amine in training_batches:
        print("Training and cross validation on {} amine.".format(amine))
        ARF = ActiveRandomForest(amine, option=None)  # TODO: Change to best_hyper_params for fine tuned models
        ARF.load_dataset(training_batches, validation_batches, meta=meta)
        ARF.train()
        ARF.active_learning(to_params=False)
        ARF.save_model(k_shot=k_shot, n_way=2, meta=meta)  # TODO: delete this when running full models
