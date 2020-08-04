from collections import defaultdict
import os
from pathlib import Path
import pickle

from modAL.models import ActiveLearner
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from utils.utils import grid_search
from utils.dataset import process_dataset


class ActiveLogisticRegression:
    """A Logistic Regression machine learning model using active learning with modAL package

    Attributes:
        amine:          A string representing the amine that the Logistic Regression model is used for predictions.
        model:          A LogisticRegression object as the classifier model.
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
    """

    def __init__(self, amine=None, config=None, verbose=True, stats_path=Path('./results/stats.pkl'),
                 model_name='Logistic Regression'):
        """initialization of the class

        Args:
            verbose:        A boolean. Output additional information to the
                            terminal for functions with verbose feature.
                            Default = True

        """
        self.amine = amine

        if config:
            self.model = LogisticRegression(**config)
        else:
            self.model = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0, solver='lbfgs', max_iter=6000)

        self.metrics = defaultdict(dict)
        self.verbose = verbose
        self.stats_path = stats_path
        self.model_name = model_name

    def load_dataset(self, set_id, x_t, y_t, x_v, y_v, all_data, all_labels):
        """Load the input training and validation data and labels into the model.

        Args:
            set_id:             An integer representing the id of the random draw that we are loading.
            x_t:                A 2-D numpy array representing the training data.
            y_t:                A 2-D numpy array representing the training labels.
            x_v:                A 2-D numpy array representing the validation data.
            y_v:                A 2-D numpy array representing the validation labels.
            all_data:           A 2-D numpy array representing all the data in the active learning pool.
            all_labels:         A 2-D numpy array representing all the labels in the active learning pool.
        """
        self.draw_id = set_id
        self.metrics[self.draw_id] = defaultdict(list)

        self.x_t, self.y_t, self.x_v, self.y_v = x_t, y_t, x_v, y_v

        self.all_data = all_data
        self.all_labels = all_labels

        if self.verbose:
            print(f'The training data has dimension of {self.x_t.shape}.')
            print(f'The training labels has dimension of {self.y_t.shape}.')
            print(f'The testing data has dimension of {self.x_v.shape}.')
            print(f'The testing labels has dimension of {self.y_v.shape}.')

    def train(self, warning=True):
        """Train the KNN model by setting up the ActiveLearner."""

        self.learner = ActiveLearner(estimator=self.model, X_training=self.x_t, y_training=self.y_t)
        # Evaluate zero-point performance
        self.evaluate(warning=warning)

    def active_learning(self, num_iter=None, warning=True):
        """The active learning loop

        This is the active learning model that loops around the decision tree model
        to look for the most uncertain point and give the model the label to train

        Args:
            num_iter:   An integer that is the number of iterations.
                        Default = None
            warning:    A boolean that decide if to declare zero division warning or not.
                        Default = True.
        """

        num_iter = num_iter if num_iter else self.x_v.shape[0]

        for _ in range(num_iter):
            # Query the most uncertain point from the active learning pool
            query_index, query_instance = self.learner.query(self.x_v)

            # Teach our ActiveLearner model the record it has requested.
            uncertain_data, uncertain_label = self.x_v[query_index].reshape(1, -1), self.y_v[query_index].reshape(1, )
            self.learner.teach(X=uncertain_data, y=uncertain_label)

            self.evaluate(warning=warning)

            # Remove the queried instance from the unlabeled pool.
            self.x_t = np.append(self.x_t, uncertain_data).reshape(-1, self.all_data.shape[1])
            self.y_t = np.append(self.y_t, uncertain_label)
            self.x_v = np.delete(self.x_v, query_index, axis=0)
            self.y_v = np.delete(self.y_v, query_index)

    def evaluate(self, warning=True, store=True):
        """Evaluation of the model

        Args:
            warning:    A boolean that decides if to warn about the zero division issue or not.
                            Default = True
            store:      A boolean that decides if to store the metrics of the performance of the model.
                            Default = True
        """

        # Calculate and report our model's accuracy.
        accuracy = self.learner.score(self.all_data, self.all_labels)

        self.y_preds = self.learner.predict(self.all_data)

        cm = confusion_matrix(self.all_labels, self.y_preds)

        # To prevent nan value for precision, we set it to 1 and send out a warning message
        if cm[1][1] + cm[0][1] != 0:
            precision = cm[1][1] / (cm[1][1] + cm[0][1])
        else:
            precision = 1.0
            if warning:
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

        self.metrics[self.draw_id]['confusion_matrices'].append(cm)
        self.metrics[self.draw_id]['accuracies'].append(accuracy)
        self.metrics[self.draw_id]['precisions'].append(precision)
        self.metrics[self.draw_id]['recalls'].append(recall)
        self.metrics[self.draw_id]['bcrs'].append(bcr)

        if self.verbose:
            print(cm)
            print('accuracy for model is', accuracy)
            print('precision for model is', precision)
            print('recall for model is', recall)
            print('balanced classification rate for model is', bcr)

    def find_inner_avg(self):
        """Find the average across all random draws"""
        metric_names = ['accuracies', 'precisions', 'recalls', 'bcrs']
        rand_draws = list(self.metrics.keys())

        for metric in metric_names:
            lst_of_metrics = []
            for set_id in rand_draws:
                lst_of_metrics.append(self.metrics[set_id][metric])
            self.metrics['average'][metric] = list(np.average(lst_of_metrics, axis=0))

        lst_of_confusion_matrices = []
        for set_id in rand_draws:
            lst_of_confusion_matrices.append(self.metrics[set_id]['confusion_matrices'])
        self.metrics['average']['confusion_matrices'] = lst_of_confusion_matrices

    def store_metrics_to_params(self):
        """Store the metrics results to the model's parameters dictionary

        Use the same logic of saving the metrics for each model.
        Dump the cross validation statistics to a pickle file.
        """
        self.find_inner_avg()

        model = self.model_name

        if self.stats_path.exists():
            with open(self.stats_path, "rb") as f:
                stats_dict = pickle.load(f)
        else:
            stats_dict = {}

        if model not in stats_dict:
            stats_dict[model] = defaultdict(list)

        stats_dict[model]['amine'].append(self.amine)
        stats_dict[model]['accuracies'].append(self.metrics['average']['accuracies'])
        stats_dict[model]['confusion_matrices'].append(
            self.metrics['average']['confusion_matrices'])
        stats_dict[model]['precisions'].append(self.metrics['average']['precisions'])
        stats_dict[model]['recalls'].append(self.metrics['average']['recalls'])
        stats_dict[model]['bcrs'].append(self.metrics['average']['bcrs'])

        # Save this dictionary in case we need it later
        with open(self.stats_path, "wb") as f:
            pickle.dump(stats_dict, f)

    def save_model(self, model_name):
        """Save the data used to train, validate and test the model to designated folder
        Args:
            model_name:         A string representing the name of the model.
        """

        # Set up the main destination folder for the model
        dst_root = './data/LogisticRegression/{0:s}'.format(model_name)
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
            print(f'No folder for LogisticRegression model {model_name} storage found')
            print(f'Make folder to store model at')

        # Dump the model into the designated folder
        file_name = "{0:s}_{1:s}.pkl".format(model_name, self.amine)
        with open(os.path.join(dst_root, file_name), "wb") as f:
            pickle.dump(self, f)


def run_model(LogisticRegression_params, category):
    """Full-scale training, validation and testing using all amines.
    Args:
        LogisticRegression_params:          A dictionary of the parameters for the LogisticRegression model.
                                                See initialize() for more information.
        category:                           A string representing the category the model is classified under.
    """

    # Unload common parameters
    config = LogisticRegression_params['config'][category] if LogisticRegression_params['config'] else None
    verbose = LogisticRegression_params['verbose']
    warning = LogisticRegression_params['warning']
    stats_path = LogisticRegression_params['stats_path']

    model_name = LogisticRegression_params['model_name']
    print(f'Running model {model_name}')

    # Unload the training data specific parameters
    num_draws = LogisticRegression_params['num_draws']
    train_size = LogisticRegression_params['train_size']
    cross_validation = LogisticRegression_params['cross_validate']
    active_learning = LogisticRegression_params['active_learning']
    w_hx = LogisticRegression_params['with_historical_data']
    w_k = LogisticRegression_params['with_k']
    active_learning_iter = LogisticRegression_params['active_learning_iter']
    full = LogisticRegression_params['full_dataset']
    draw_success = LogisticRegression_params['draw_success']

    # Specify the desired operation
    fine_tuning = LogisticRegression_params['fine_tuning']
    save_model = LogisticRegression_params['save_model']
    to_params = True

    if fine_tuning:
        class_weights = [{0: i, 1: 1.0 - i} for i in np.linspace(.05, .95, num=50)]
        class_weights.append('balanced')
        class_weights.append(None)

        ft_params = {
            'penalty': ['l2', 'none'],
            'dual': [False],
            'tol': [1e-4, 1e-5],
            'C': [.1 * i for i in range(1, 3)],
            'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [4000, 5000, 6000, 7000, 9000],
            'class_weight': class_weights
        }

        result_path = './results/ft_{}.pkl'.format(model_name)

        grid_search(
            ActiveLogisticRegression,
            ft_params,
            result_path,
            num_draws,
            train_size,
            active_learning_iter,
            active_learning=active_learning,
            w_hx=w_hx,
            w_k=w_k,
            draw_success=draw_success
        )

    else:
        # Load the desired sized dataset under desired option
        dataset = process_dataset(
            num_draw=num_draws,
            train_size=train_size,
            active_learning_iter=active_learning_iter,
            verbose=verbose,
            cross_validation=cross_validation,
            full=full,
            active_learning=active_learning,
            w_hx=w_hx,
            w_k=w_k,
            success=draw_success
        )

        draws = list(dataset.keys())
        amine_list = list(dataset[0]['x_t'].keys())

        for amine in amine_list:
            if amine == 'XZUCBFLUEBDNSJ-UHFFFAOYSA-N' and draw_success:
                # Skipping the amine with only 1 successful experiment overall
                # Can't run 4-ii and 5-ii models on this amine
                continue
            else:
                # Create the LogisticRegression model instance for the specific amine
                ALR = ActiveLogisticRegression(amine=amine, config=config, verbose=verbose, stats_path=stats_path,
                                               model_name=model_name)

                for set_id in draws:
                    # Unload the randomly drawn dataset values
                    x_t, y_t, x_v, y_v, all_data, all_labels = dataset[set_id]['x_t'], \
                                                               dataset[set_id]['y_t'], \
                                                               dataset[set_id]['x_v'], \
                                                               dataset[set_id]['y_v'], \
                                                               dataset[set_id]['all_data'], \
                                                               dataset[set_id]['all_labels']
                    # Load the training and validation set into the model
                    ALR.load_dataset(set_id, x_t[amine], y_t[amine], x_v[amine], y_v[amine], all_data[amine], all_labels[amine])

                    # Train the data on the training set
                    ALR.train(warning=warning)

                    # Conduct active learning with all the observations available in the pool
                    if active_learning:
                        ALR.active_learning(num_iter=active_learning_iter, warning=warning)

                if to_params:
                    ALR.store_metrics_to_params()

                # Save the model for future reproducibility
                if save_model:
                    ALR.save_model(model_name)

            # TODO: testing part not implemented
