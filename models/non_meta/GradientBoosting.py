from collections import defaultdict
import os
from pathlib import Path
import pickle

from modAL.models import ActiveLearner
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from models.non_meta.BaseClassifier import ActiveLearningClassifier
from utils.utils import grid_search
from utils.dataset import process_dataset


class ActiveGradientBoosting(ActiveLearningClassifier):
    """ A class of Gradient Boosting model with active learning

    Attributes:
        amine:              A string representing the amine that the Logistic Regression model is used for predictions.
        config:             A dictionary representing the hyper-parameters of the model
        metrics:            A dictionary to store the performance metrics locally. It has the format of
                                {'metric_name': [metric_value]}.
        verbose:            A boolean representing whether it will prints out additional information to the terminal
                                or not.
        stats_path:         A Path object representing the directory of the stats dictionary if we are not running
                                multi-processing.
        result_dict:        A dictionary representing the result dictionary used during multi-thread processing.
        classifier_name:    A string representing the name of the generic classifier.
        model_name:         A string representing the name of the specific model for future plotting.
        all_data:           A numpy array representing all the data from the dataset.
        all_labels:         A numpy array representing all the labels from the dataset.
        x_t:                A numpy array representing the training data used for model training.
        y_t:                A numpy array representing the training labels used for model training.
        x_v:                A numpy array representing the testing data used for active learning.
        y_v:                A numpy array representing the testing labels used for active learning.
        learner:            An ActiveLearner to conduct active learning with. See modAL documentation for more details.
    """

    def __init__(self, amine=None, config=None, verbose=True, stats_path=Path('./results/stats.pkl'), result_dict=None,
                 classifier_name='Gradient_Boosting', model_name='Gradient_Boosting'):
        """Initialize the ActiveGradientBoosting object."""
        super().__init__(
            amine=amine,
            config=config,
            verbose=verbose,
            stats_path=stats_path,
            result_dict=result_dict,
            classifier_name=classifier_name,
            model_name=model_name
        )

        if config:
            self.model = GradientBoostingClassifier(**config)
        else:
            self.model = GradientBoostingClassifier(
                loss='deviance',
                learning_rate=0.1,
                n_estimators=100,
                criterion='friedman_mse',
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=None,
                ccp_alpha=0.0)


def run_model(GradientBoosting_params, category):
    """Full-scale training, validation and testing using all amines.
    Args:
        GradientBoosting_params:         A dictionary of the parameters for the Gradient Boosting model.
                                            See initialize() for more information.
        category:                        A string representing the category the model is classified under.
    """

    # Unload common parameters
    config = GradientBoosting_params['config'][category] if GradientBoosting_params['config'] else None
    verbose = GradientBoosting_params['verbose']
    warning = GradientBoosting_params['warning']
    stats_path = GradientBoosting_params['stats_path']
    result_dict = GradientBoosting_params['result_dict']

    model_name = GradientBoosting_params['model_name']
    print(f'Running model {model_name}')

    # Unload the training data specific parameters
    num_draws = GradientBoosting_params['num_draws']
    train_size = GradientBoosting_params['train_size']
    active_learning_iter = GradientBoosting_params['active_learning_iter']
    active_learning = GradientBoosting_params['active_learning']
    cross_validation = GradientBoosting_params['cross_validate']
    full = GradientBoosting_params['full_dataset']
    w_hx = GradientBoosting_params['with_historical_data']
    w_k = GradientBoosting_params['with_k']
    draw_success = GradientBoosting_params['draw_success']

    # Specify the desired operation
    fine_tuning = GradientBoosting_params['fine_tuning']
    save_model = GradientBoosting_params['save_model']
    to_file = True

    if fine_tuning:
        ft_params = {
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [100, 200, 500, 1000],
            'criterion': ['friedman_mse', 'mse', 'mae'],
            'max_depth': [i for i in range(1, 9)],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'min_samples_leaf': [1, 2, 3],
            'min_samples_split': [2, 5, 10],
            'ccp_alpha': [.1 * i for i in range(1)]
        }

        result_path = './results/ft_{}.pkl'.format(model_name)

        grid_search(
            ActiveGradientBoosting,
            ft_params,
            result_path,
            num_draws,
            train_size,
            active_learning_iter,
            active_learning=active_learning,
            w_hx=w_hx,
            w_k=w_k,
            draw_success=draw_success,
            result_dict=result_dict,
            model_name=model_name
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

        # print(training_batches.keys())
        for amine in amine_list:
            if amine == 'XZUCBFLUEBDNSJ-UHFFFAOYSA-N' and draw_success:
                # Skipping the amine with only 1 successful experiment overall
                # Can't run 4-ii and 5-ii models on this amine
                continue
            else:
                # Create the GradientBoosting model instance for the specific amine
                AGB = ActiveGradientBoosting(amine=amine, config=config, verbose=verbose, stats_path=stats_path,
                                             result_dict=result_dict, model_name=model_name)
                for set_id in draws:
                    # Unload the randomly drawn dataset values
                    x_t, y_t, x_v, y_v, all_data, all_labels = dataset[set_id]['x_t'], \
                                                               dataset[set_id]['y_t'], \
                                                               dataset[set_id]['x_v'], \
                                                               dataset[set_id]['y_v'], \
                                                               dataset[set_id]['all_data'], \
                                                               dataset[set_id]['all_labels']
                    # Load the training and validation set into the model
                    AGB.load_dataset(
                        set_id,
                        x_t[amine],
                        y_t[amine],
                        x_v[amine],
                        y_v[amine],
                        all_data[amine],
                        all_labels[amine])

                    # Train the data on the training set
                    AGB.train(warning=warning)

                    # Conduct active learning with all the observations available in the pool
                    if active_learning:
                        AGB.active_learning(num_iter=active_learning_iter, warning=warning)

                if to_file:
                    AGB.store_metrics_to_file()

                # Save the model for future reproducibility
                if save_model:
                    AGB.save_model(model_name)

            # TODO: testing part not implemented
