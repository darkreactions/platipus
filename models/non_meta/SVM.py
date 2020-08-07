from pathlib import Path
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from models.non_meta.BaseClassifier import ActiveLearningClassifier
from utils.utils import grid_search, PUK_kernel
from utils.dataset import process_dataset


class ActiveSVM(ActiveLearningClassifier):
    """A SVM machine learning model using active learning with modAL package

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
                 classifier_name='SVM', model_name='SVM'):
        """Initialize the ActiveSVM object."""
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
            self.model = CalibratedClassifierCV(SVC(**config, cache_size=7000))
        else:
            # Fine tuned model
            self.model = CalibratedClassifierCV(SVC(
                C=.003,
                kernel='poly',
                degree=3,
                gamma='scale',
                tol=1,
                decision_function_shape='ovo',
                break_ties=True,
            ))


def run_model(SVM_params, category):
    """Full-scale training, validation and testing using all amines.

    Args:
        SVM_params:         A dictionary of the parameters for the SVM model.
                                See initialize() for more information.
        category:           A string representing the category the model is classified under.
     """
    
    # Unload common parameters
    config = SVM_params['configs'][category] if SVM_params['configs'] else None
    verbose = SVM_params['verbose']
    warning = SVM_params['warning']
    stats_path = SVM_params['stats_path']
    result_dict = SVM_params['result_dict']

    model_name = SVM_params['model_name']
    print(f'Running model {model_name}')

    # Unload the training data specific parameters
    num_draws = SVM_params['num_draws']
    train_size = SVM_params['train_size']
    active_learning_iter = SVM_params['active_learning_iter']
    cross_validation = SVM_params['cross_validate']
    full = SVM_params['full_dataset']
    active_learning = SVM_params['active_learning']
    w_hx = SVM_params['with_historical_data']
    w_k = SVM_params['with_k']
    draw_success = SVM_params['draw_success']

    # Specify the desired operation
    fine_tuning = SVM_params['fine_tuning']
    save_model = SVM_params['save_model']
    to_file = True

    if fine_tuning:
        class_weights = [{0: i, 1: 1.0-i} for i in np.linspace(.1, .9, num=9)]
        class_weights.append('balanced')
        class_weights.append(None)

        ft_params = {
            'kernel': ['poly', 'sigmoid', 'rbf', PUK_kernel],
            'C': [.01, .1, 1, 10, 100],
            'degree': [i for i in range(1, 6)],
            'gamma': ['auto', 'scale'],
            'tol': [.001, .01, .1, 1],
            'decision_function_shape': ['ovo'],
            'break_ties': [True],
            'class_weight': class_weights
        }

        result_path = './results/ft_{}.pkl'.format(model_name)

        grid_search(
            ActiveSVM,
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
            if amine == 'XZUCBFLUEBDNSJ-UHFFFAOYSA-N' and draw_success:
                # Skipping the amine with only 1 successful experiment overall
                # Can't run 4-ii and 5-ii models on this amine
                continue
            else:
                # Create the SVM model instance for the specific amine
                ASVM = ActiveSVM(
                    amine=amine,
                    config=config,
                    verbose=verbose,
                    stats_path=stats_path,
                    result_dict=result_dict,
                    model_name=model_name
                )

                for set_id in draws:
                    if cross_validation:
                        # Unload the randomly drawn dataset values
                        x_t, y_t, x_v, y_v, all_data, all_labels = dataset[set_id]['x_t'], \
                                                                   dataset[set_id]['y_t'], \
                                                                   dataset[set_id]['x_v'], \
                                                                   dataset[set_id]['y_v'], \
                                                                   dataset[set_id]['all_data'], \
                                                                   dataset[set_id]['all_labels']

                        # Load the training and validation set into the model
                        ASVM.load_dataset(
                            set_id,
                            x_t[amine],
                            y_t[amine],
                            x_v[amine],
                            y_v[amine],
                            all_data[amine],
                            all_labels[amine]
                        )

                        # Train the data on the training set
                        ASVM.train(warning=warning)

                        # Conduct active learning with all the observations available in the pool
                        if active_learning:
                            ASVM.active_learning(num_iter=active_learning_iter, warning=warning)

                if to_file:
                    ASVM.store_metrics_to_file()

                # Save the model for future reproducibility
                if save_model:
                    ASVM.save_model(model_name)

            # TODO: testing part not implemented: might need to change the logic loading things in
