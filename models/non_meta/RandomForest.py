from pathlib import Path
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from models.non_meta.BaseClassifier import ActiveLearningClassifier
from utils.utils import grid_search
from utils.dataset import process_dataset


class ActiveRandomForest(ActiveLearningClassifier):
    """ A class of Random Forest model with active learning
    
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
                 classifier_name='Random_Forest', model_name='Random_Forest'):
        """Initialize the ActiveRandomForest object."""
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
            self.model = RandomForestClassifier(**config)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                criterion='gini',
                max_depth=8,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=None,
                bootstrap=True,
                ccp_alpha=0.0)


def run_model(RandomForest_params, category):
    """Full-scale training, validation and testing using all amines.
    Args:
        RandomForest_params:         A dictionary of the parameters for the random forest model.
                                        See initialize() for more information.
        category:                    A string representing the category the model is classified under.
    """

    # Unload common parameters
    config = RandomForest_params['config'][category] if RandomForest_params['config'] else None
    verbose = RandomForest_params['verbose']
    warning = RandomForest_params['warning']
    stats_path = RandomForest_params['stats_path']
    result_dict = RandomForest_params['result_dict']

    model_name = RandomForest_params['model_name']
    print(f'Running model {model_name}')

    # Unload the training data specific parameters
    num_draws = RandomForest_params['num_draws']
    train_size = RandomForest_params['train_size']
    cross_validation = RandomForest_params['cross_validate']
    active_learning = RandomForest_params['active_learning']
    w_hx = RandomForest_params['with_historical_data']
    w_k = RandomForest_params['with_k']
    active_learning_iter = RandomForest_params['active_learning_iter']
    full = RandomForest_params['full_dataset']
    draw_success = RandomForest_params['draw_success']

    # Specify the desired operation
    fine_tuning = RandomForest_params['fine_tuning']
    save_model = RandomForest_params['save_model']
    to_file = True

    if fine_tuning:
        class_weights = [{0: i, 1: 1.0 - i} for i in np.linspace(.05, .95, num=50)]
        class_weights.append('balanced')
        class_weights.append(None)

        ft_params = {
            'n_estimators': [100, 200, 500, 1000],
            'criterion': ['gini','entropy'],
            'max_depth': [i for i in range(1, 9)],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'bootstrap': [True],
            'min_samples_leaf': [i for i in range(1, 6)],
            'min_samples_split': [i for i in range(2, 11)],
            'ccp_alpha': [.1 * i for i in range(1)],
            'class_weight': class_weights
        }

        result_path = './results/ft_{}.pkl'.format(model_name)

        grid_search(
            ActiveRandomForest,
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

        for amine in amine_list:

            # Create the RandomForest model instance for the specific amine
            ARF = ActiveRandomForest(
                amine=amine,
                config=config,
                verbose=verbose,
                stats_path=stats_path,
                result_dict=result_dict,
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
                ARF.load_dataset(set_id, x_t[amine],y_t[amine],x_v[amine],y_v[amine],all_data[amine],all_labels[amine])

                # Train the data on the training set
                ARF.train(warning=warning)

                # Conduct active learning with all the observations available in the pool
                if active_learning:
                    ARF.active_learning(num_iter=active_learning_iter, warning=warning)

            if to_file:
                ARF.store_metrics_to_file()

            # Save the model for future reproducibility
            if save_model:
                ARF.save_model(model_name)

            # TODO: testing part not implemented
