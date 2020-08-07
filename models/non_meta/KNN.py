from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier

from models.non_meta.BaseClassifier import ActiveLearningClassifier
from utils.utils import grid_search
from utils.dataset import process_dataset


class ActiveKNN(ActiveLearningClassifier):
    """A KNN machine learning model using active learning with modAL package

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
                 classifier_name='KNN', model_name='KNN'):
        """Initialize the ActiveKNN object."""
        super().__init__(
            amine=amine,
            config=config,
            verbose=verbose,
            stats_path=stats_path,
            result_dict=result_dict,
            classifier_name=classifier_name,
            model_name=model_name
        )

        # Load customized model or use the default fine-tuned setting
        if config:
            self.model = KNeighborsClassifier(**config)
            self.n_neighbors = config['n_neighbors']
        else:
            self.model = KNeighborsClassifier(n_neighbors=3, p=1)
            self.n_neighbors = 3


def run_model(KNN_params, category):
    """Full-scale training, validation and testing using all amines.

    Args:
        KNN_params:         A dictionary of the parameters for the KNN model.
                                See initialize() for more information.
        category:           A string representing the category the model is classified under.
    """

    # Unload common parameters
    config = KNN_params['configs'][category] if KNN_params['configs'] else None
    verbose = KNN_params['verbose']
    warning = KNN_params['warning']
    stats_path = KNN_params['stats_path']
    result_dict = KNN_params['result_dict']

    model_name = KNN_params['model_name']
    print(f'Running model {model_name}')

    # Unload the training data specific parameters
    num_draws = KNN_params['num_draws']
    train_size = KNN_params['train_size']
    active_learning_iter = KNN_params['active_learning_iter']
    cross_validation = KNN_params['cross_validate']
    full = KNN_params['full_dataset']
    active_learning = KNN_params['active_learning']
    w_hx = KNN_params['with_historical_data']
    w_k = KNN_params['with_k']
    draw_success = KNN_params['draw_success']

    # Specify the desired operation
    fine_tuning = KNN_params['fine_tuning']
    save_model = KNN_params['save_model']
    to_file = True

    if fine_tuning:
        # Set all possible combinations
        ft_params = {
            'n_neighbors': [i for i in range(1, 10)],
            'leaf_size': [i for i in range(1, 51)],
            'p': [i for i in range(1, 4)]
        }

        result_path = './results/ft_{}.pkl'.format(model_name)

        grid_search(
            ActiveKNN,
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
            model_name=model_name,
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
            success=draw_success,
        )

        draws = list(dataset.keys())
        amine_list = list(dataset[0]['x_t'].keys())

        for amine in amine_list:
            # Create the KNN model instance for the specific amine
            KNN = ActiveKNN(amine=amine, config=config, verbose=verbose, stats_path=stats_path, result_dict=result_dict,
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
                KNN.load_dataset(set_id, x_t[amine], y_t[amine], x_v[amine], y_v[amine], all_data[amine],
                                 all_labels[amine])

                # Train the data on the training set
                KNN.train(warning=warning)

                # Conduct active learning with all the observations available in the pool
                if active_learning:
                    KNN.active_learning(num_iter=active_learning_iter, warning=warning)

            if to_file:
                KNN.store_metrics_to_file()

            # Save the model for future reproducibility
            if save_model:
                KNN.save_model(model_name)

            # TODO: testing part not implemented
