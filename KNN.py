"""
Run the following command lines for full dataset training:
python KNN.py --datasource=drp_chem --k_shot=20 --n_way=2 --meta_batch_size=10 --num_batches=250 --cross_validate --full --verbose
python KNN.py --datasource=drp_chem --k_shot=20 --n_way=2 --meta_batch_size=10 --num_batches=250 --cross_validate --meta --full --verbose

Run the following command lines for test dataset training (debug):
python KNN.py --datasource=drp_chem --k_shot=20 --n_way=2 --meta_batch_size=10 --num_batches=250 --cross_validate --verbose
python KNN.py --datasource=drp_chem --k_shot=20 --n_way=2 --meta_batch_size=10 --num_batches=250 --cross_validate --meta --verbose
"""

import argparse
import os
import pickle
import sys

import numpy as np
from dataset import import_full_dataset, import_test_dataset
from modAL.models import ActiveLearner
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class ActiveKNN:
    """A KNN machine learning model using active learning with modAL package

    Attributes:
        amine:          A string representing the amine that the KNN model is used for predictions.
        n_neighbors:    An integer representing the number of neighbors to classify using KNN model.
        model:          A KNeighborClassifier object as the classifier model given the number of neighbors to classify
                            with.
        metrics:        A dictionary to store the performance metrics locally. It has the format of
                            {'metric_name': [metric_value]}.
        verbose:        A boolean representing whether it will prints out additional information to the terminal or not.
        pool_data:      A numpy array representing all the data from the dataset.
        pool_labels:    A numpy array representing all the labels from the dataset.
        x_t:            A numpy array representing the training data used for model training.
        y_t:            A numpy array representing the training labels used for model training.
        x_v:            A numpy array representing the testing data used for active learning.
        y_v:            A numpy array representing the testing labels used for active learning.
        learner:        An ActiveLearner to conduct active learning with. See modAL documentation for more details.
    """

    def __init__(self, amine=None, n_neighbors=2, verbose=True):
        """Initialize the ActiveKNN object."""
        self.amine = amine
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.metrics = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'bcrs': [],
            'confusion_matrices': []
        }
        self.verbose = verbose

    def load_dataset(self, x_t, y_t, x_v, y_v, all_data, all_labels):
        """Load the input training and validation data and labels into the model.

        Args:
            x_t:                A 2-D numpy array representing the training data.
            y_t:                A 2-D numpy array representing the training labels.
            x_v:                A 2-D numpy array representing the validation data.
            y_v:                A 2-D numpy array representing the validation labels.
            all_data:           A 2-D numpy array representing all the data in the active learning pool.
            all_labels:         A 2-D numpy array representing all the labels in the active learning pool.

        Returns:
            N/A
        """

        self.x_t, self.x_v, self.y_t, self.y_v = x_t, y_t, x_v, y_v

        self.pool_data = all_data
        self.pool_labels = all_labels

        if self.verbose:
            print(f'The training data has dimension of {self.x_t.shape}.')
            print(f'The training labels has dimension of {self.y_t.shape}.')
            print(f'The testing data has dimension of {self.x_v.shape}.')
            print(f'The testing labels has dimension of {self.y_v.shape}.')

    def train(self):
        """Train the KNN model by setting up the ActiveLearner."""

        self.learner = ActiveLearner(estimator=self.model, X_training=self.x_t, y_training=self.y_t)
        # Evaluate zero-point performance
        self.evaluate()

    def active_learning(self, num_iter=None, to_params=True):
        """ The active learning loop

        This is the active learning model that loops around the KNN model
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
            self.x_t = np.append(self.x_t, uncertain_data).reshape(-1, self.pool_data.shape[1])
            self.y_t = np.append(self.y_t, uncertain_label)
            self.x_v = np.delete(self.x_v, query_index, axis=0)
            self.y_v = np.delete(self.y_v, query_index)

        if to_params:
            self.store_metrics_to_params()

    def evaluate(self, store=True):
        """Evaluation of the model

        Args:
            store:  A boolean that decides if to store the metrics of the performance of the model.
                    Default = True

        return: N/A
        """

        # Calculate and report our model's accuracy.
        accuracy = self.learner.score(self.pool_data, self.pool_labels)

        preds = self.learner.predict(self.pool_data)
        cm = confusion_matrix(self.pool_labels, preds)

        # To prevent nan value for precision, we set it to 1 and send out a warning message
        if cm[1][1] + cm[0][1] != 0:
            precision = cm[1][1] / (cm[1][1] + cm[0][1])
        else:
            precision = 1.0     # TODO: I still think 0.0 is better
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

        model = 'KNN'   # TODO: Implement a way to identify models under meta or not

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

        # Indicate which option we used the data for
        option = 2 if meta else 1

        # Set up the main destination folder for the model
        dst_root = './KNN_few_shot/option_{0:d}'.format(option)
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
            print('No folder for KNN model storage found')
            print(f'Make folder to store KNN model at')

        # Set up the model specific folder
        model_folder = '{0:s}/KNN_{1:d}_shot_{2:d}_way_option_{3:d}_{4:s}'.format(dst_root,
                                                                                  k_shot,
                                                                                  n_way,
                                                                                  option,
                                                                                  self.amine)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print('No folder for KNN model storage found')
            print(f'Make folder to store KNN model of amine {self.amine} at')
        else:
            print(f'Found existing folder. Model of amine {self.amine} will be stored at')
        print(model_folder)

        # Dump the model into the designated folder
        file_name = "KNN_{0:s}_option_{1:d}.pkl".format(self.amine, option)
        with open(os.path.join(model_folder, file_name), "wb") as f:
            pickle.dump(self, f)

    def __str__(self):
        return 'A {0:d}-neighbor KNN model for amine {1:s} using active learning'.format(self.n_neighbors, self.amine)


# TODO: may have to change this
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
            meta:               A boolean representing if it will be trained under option 1 or option 2.
                                        Option 1 is train with observations of other tasks and validate on the
                                        task-specific observations.
                                        Option 2 is to train and validate on the task-specific observations.
            verbose:            A boolean. Including it in the command line will output additional information to the
                                    terminal for functions with verbose feature.
    """

    parser = argparse.ArgumentParser(
        description='Setup variables for active learning KNN.')
    parser.add_argument('--datasource', type=str, default='drp_chem',
                        help='datasource to be used, defaults to drp_chem')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='Number of training samples per class')
    parser.add_argument('--n_way', type=int, default=1,
                        help='Number of classes per task, this is 2 for the chemistry data')

    parser.add_argument('--num_batches', type=int, default=250,
                        help='Number of meta-batches per task')
    parser.add_argument('--meta_batch_size', type=int, default=25,
                        help='Number of tasks sampled per outer loop')

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--cross_validate', action='store_true')
    parser.add_argument('--meta', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    return args


# TODO: may have to change this
def initialize():
    """Initializes a dictionary of parameters corresponding to the arguments

    Args:
        N/A

    Returns:
        KNN_params: A dictionary of the parameters for the KNN model with the following keys:
            datasource:             A string representing the data used for KNN.
            k_shot:                 An integer representing the number of training samples per class.
            n_way:                  An integer representing the number of classes per task.
            num_batches:            An integer representing the number of meta-batches per task.
            meta_batch_size:        An integer representing the size of a meta-batch per task.
            train_flag:             A boolean representing if it will be training the model or testing.
            full_dataset:           A boolean representing if it will be running on the full dataset or not.
            meta:                   A boolean representing if it will be trained under option 1 or option 2.
                                        Option 1 is train with observations of other tasks and validate on the
                                        task-specific observations.
                                        Option 2 is to train and validate on the task-specific observations.
            cross_validate:         A boolean representing if it will be conducting cross-validation or not.
            verbose:                A boolean. representing whether it will output additional information to the
                                        terminal for functions with verbose feature.
    """
    args = parse_args()
    KNN_params = {}

    # Initialize the datasource
    print(f'Dataset = {args.datasource}')
    KNN_params['datasource'] = args.datasource

    # Set up the value of k in k-shot learning
    print(f'{args.k_shot}-shot')
    KNN_params['k_shot'] = args.k_shot

    # n-way: 2 for the chemistry data
    print(f"{args.n_way}-way")
    KNN_params['n_way'] = args.n_way

    # Number of meta-batches per task
    KNN_params['num_batches'] = args.num_batches

    # Number of tasks sampled per outer loop
    KNN_params['meta_batch_size'] = args.meta_batch_size

    # Set up KNN actions from the user input
    KNN_params['train_flag'] = args.train_flag

    print('Using all experiments') if args.full else print('Using small scale dataset for debugging')
    KNN_params['full_dataset'] = args.full

    print('Training and validate the model with option 2') if args.meta else print('Training and validate the model with option 1')
    KNN_params['meta'] = args.meta

    KNN_params['cross_validate'] = args.cross_validate
    KNN_params['verbose'] = args.verbose

    return KNN_params


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
    data_folder = './KNN_few_shot/option_{0:d}/data'.format(option)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print('No folder for KNN model data storage found')
        print('Make folder to store data used for KNN models at')
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
    file_name = "KNN_data.pkl"
    with open(os.path.join(data_folder, file_name), "wb") as f:
        pickle.dump(data, f)


def iris_test():
    """Test the model on the iris data provided in the scikit-learn package.

    This is used only to test new implementations.

    Args:
          N/A

    Returns:
         N/A
    """
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    scaler = StandardScaler()
    scaler.fit(X)
    normalized_data = scaler.transform(X)

    x_t, x_v, y_t, y_v = train_test_split(
        normalized_data,
        y,
        random_state=2)

    KNN = ActiveKNN(n_neighbors=3)
    KNN.load_dataset(x_t, x_v, y_t, y_v, normalized_data, y)

    KNN.train()
    KNN.active_learning(to_params=False)


def small_scale(KNN_params):
    """A small scale training and validation for debugging using only 3 amines.

    Args:
        KNN_params:         A dictionary of the parameters for the KNN model.
                                See initialize() for more information.

    Returns:
        N/A
    """

    # Unload parameters
    k_shot = KNN_params['k_shot']
    n_way = KNN_params['n_way']
    meta_batch_size = KNN_params['meta_batch_size']
    num_batches = KNN_params['num_batches']
    meta = KNN_params['meta']
    verbose = KNN_params['verbose']

    # Load the small-scale 3-amine dataset for training and validation
    training_batches, validation_batches, testing_batches, counts = import_test_dataset(k_shot,
                                                                                        meta_batch_size,
                                                                                        num_batches,
                                                                                        verbose=verbose,
                                                                                        cross_validation=True,
                                                                                        meta=meta)
    
    save_used_data(training_batches, validation_batches, testing_batches, counts, meta)
    
    for amine in training_batches:
        print(f'Training and cross-validation amine {amine}')
        # Create the KNN model instance for the specific amine
        KNN = ActiveKNN(amine=amine, n_neighbors=n_way, verbose=verbose)

        # Option 1:
        if not meta:
            print('Conducting training under option 1.')
            x_t, y_t = training_batches[amine][0], training_batches[amine][1]
            x_v, y_v = validation_batches[amine][0], validation_batches[amine][1]
            all_data, all_labels = x_v, y_v
        # Option 2:
        else:
            print('Conducting training under option 2.')
            x_t, y_t = validation_batches[amine][0], validation_batches[amine][1]
            x_v, y_v = validation_batches[amine][2], validation_batches[amine][3]
            all_data, all_labels = np.concatenate((x_t, x_v)), np.concatenate((y_t, y_v))
            print(all_data.shape, all_labels.shape)

        # Load the training and validation set into the model
        KNN.load_dataset(x_t, x_v, y_t, y_v, all_data, all_labels)

        # Train the data on the training set
        KNN.train()

        # Conduct active learning with all the observations available in the pool
        KNN.active_learning(to_params=False)    # TODO: delete this when running full models


def full_dataset(KNN_params):
    """Full-scale training, validation and testing using all amines.

    Args:
        KNN_params:         A dictionary of the parameters for the KNN model.
                                See initialize() for more information.

    Returns:
        N/A
    """

    # Unload parameters
    k_shot = KNN_params['k_shot']
    n_way = KNN_params['n_way']
    meta_batch_size = KNN_params['meta_batch_size']
    num_batches = KNN_params['num_batches']
    cross_validation = KNN_params['cross_validate']
    meta = KNN_params['meta']
    verbose = KNN_params['verbose']

    # Load the full dataset for training and validation
    training_batches, validation_batches, testing_batches, counts = import_full_dataset(k_shot,
                                                                                        meta_batch_size,
                                                                                        num_batches,
                                                                                        verbose=verbose,
                                                                                        cross_validation=cross_validation,
                                                                                        meta=meta)

    save_used_data(training_batches, validation_batches, testing_batches, counts, meta)
    
    for amine in training_batches:
        print(f'Training and active learning on amine {amine}')
        # Create the KNN model instance for the specific amine
        KNN = ActiveKNN(amine=amine, n_neighbors=n_way, verbose=verbose)

        if cross_validation:
            # Option 1:
            if not meta:
                print('Conducting training under option 1.')
                x_t, y_t = training_batches[amine][0], training_batches[amine][1]
                x_v, y_v = validation_batches[amine][0], validation_batches[amine][1]
                all_data, all_labels = x_v, y_v
            # Option 2
            else:
                print('Conducting training under option 2.')
                x_t, y_t = validation_batches[amine][0], validation_batches[amine][1]
                x_v, y_v = validation_batches[amine][2], validation_batches[amine][3]
                all_data, all_labels = np.concatenate((x_t, x_v)), np.concatenate((y_t, y_v))
        else:
            sys.exit('Testing portion not implemented yet.')

        # Load the training and validation set into the model
        KNN.load_dataset(x_t, x_v, y_t, y_v, all_data, all_labels)

        # Train the data on the training set
        KNN.train()

        # Conduct active learning with all the observations available in the pool
        KNN.active_learning(to_params=False)    # TODO: delete this when running full models

        # Save model to designated folder
        KNN.save_model(k_shot=k_shot, n_way=2, meta=meta)

    # TODO: model saving and testing part not implemented


def main():
    """Main driver function"""

    KNN_params = initialize()

    if KNN_params['full_dataset']:
        full_dataset(KNN_params)
    else:
        small_scale(KNN_params)


if __name__ == "__main__":
    main()
