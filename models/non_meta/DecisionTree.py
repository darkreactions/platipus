from pathlib import Path
import numpy as np

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from models.non_meta.BaseClassifier import ActiveLearningClassifier
from utils.utils import grid_search
from utils.dataset import process_dataset


class ActiveDecisionTree(ActiveLearningClassifier):
    """ A class of decision tree model with active learning.
    
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
                 classifier_name='Decision_Tree', model_name='Decision_Tree'):
        """Initialize the ActiveDecisionTree object."""
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
            self.model = DecisionTreeClassifier(**config)
        else:
            self.model = DecisionTreeClassifier()


def run_model(DecisionTree_params, category):
    """Full-scale training, validation and testing using all amines.
    Args:
        DecisionTree_params:         A dictionary of the parameters for the decision tree model.
                                        See initialize() for more information.
        category:                    A string representing the category the model is classified under.
    """

    # Feature names hard-coded for decision tree visualization
    features = ['_rxn_M_acid', '_rxn_M_inorganic', '_rxn_M_organic', '_solv_GBL', '_solv_DMSO', '_solv_DMF',
                '_stoich_mmol_org', '_stoich_mmol_inorg', '_stoich_mmol_acid', '_stoich_mmol_solv', '_stoich_org/solv',
                '_stoich_inorg/solv', '_stoich_acid/solv', '_stoich_org+inorg/solv', '_stoich_org+inorg+acid/solv',
                '_stoich_org/liq', '_stoich_inorg/liq', '_stoich_org+inorg/liq', '_stoich_org/inorg',
                '_stoich_acid/inorg', '_rxn_Temperature_C', '_rxn_Reactiontime_s', '_feat_AvgPol',
                '_feat_Refractivity', '_feat_MaximalProjectionArea', '_feat_MaximalProjectionRadius',
                '_feat_maximalprojectionsize', '_feat_MinimalProjectionArea', '_feat_MinimalProjectionRadius',
                '_feat_minimalprojectionsize', '_feat_MolPol', '_feat_VanderWaalsSurfaceArea', '_feat_ASA',
                '_feat_ASA_H', '_feat_ASA_P', '_feat_ASA-', '_feat_ASA+', '_feat_ProtPolarSurfaceArea',
                '_feat_Hacceptorcount', '_feat_Hdonorcount', '_feat_RotatableBondCount', '_raw_standard_molweight',
                '_feat_AtomCount_N', '_feat_BondCount', '_feat_ChainAtomCount', '_feat_RingAtomCount',
                '_feat_primaryAmine', '_feat_secondaryAmine', '_rxn_plateEdgeQ', '_feat_maxproj_per_N',
                '_raw_RelativeHumidity']

    # Unload common parameters
    config = DecisionTree_params['configs'][category] if DecisionTree_params['configs'] else None
    verbose = DecisionTree_params['verbose']
    warning = DecisionTree_params['warning']
    stats_path = DecisionTree_params['stats_path']
    result_dict = DecisionTree_params['result_dict']

    model_name = DecisionTree_params['model_name']
    print(f'Running model {model_name}')

    # Unload the training data specific parameters
    num_draws = DecisionTree_params['num_draws']
    train_size = DecisionTree_params['train_size']
    active_learning_iter = DecisionTree_params['active_learning_iter']
    cross_validation = DecisionTree_params['cross_validate']
    full = DecisionTree_params['full_dataset']
    active_learning = DecisionTree_params['active_learning']
    w_hx = DecisionTree_params['with_historical_data']
    w_k = DecisionTree_params['with_k']
    draw_success = DecisionTree_params['draw_success']

    # Specify the desired operation
    fine_tuning = DecisionTree_params['fine_tuning']
    save_model = DecisionTree_params['save_model']
    visualize = DecisionTree_params['visualize']
    to_file = True

    if fine_tuning:
        class_weights = [{0: i, 1: 1.0 - i} for i in np.linspace(.05, .95, num=50)]
        class_weights.append('balanced')
        class_weights.append(None)

        max_depths = [i for i in range(9, 26)]
        max_depths.append(None)

        ft_params = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': max_depths,
            'min_samples_split': [i for i in range(2, 11)],
            'min_samples_leaf': [i for i in range(1, 4)],
            'class_weight': class_weights
        }

        result_path = './results/ft_{}.pkl'.format(model_name)

        grid_search(
            ActiveDecisionTree,
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
            # Create the decision tree model instance for the specific amine
            ADT = ActiveDecisionTree(amine=amine, config=config, verbose=verbose, stats_path=stats_path,
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
                ADT.load_dataset(set_id, x_t[amine], y_t[amine], x_v[amine], y_v[amine], all_data[amine], all_labels[amine])

                # Train the data on the training set
                ADT.train(warning=warning)

                # Conduct active learning with all the observations available in the pool
                if active_learning:
                    ADT.active_learning(num_iter=active_learning_iter, warning=warning)

                if visualize:
                    # Plot the decision tree
                    # To compile the graph, use the following command in terminal
                    # dot -Tpng "{dt_file_name}.dot" -o "{desired file name}.png"
                    # If using Jupyter Notebook, add ! in front to run command lines
                    file_name = './results/{0:s}_dt_{1:s}_{2:d}.dot'.format(model_name, amine, set_id)
                    export_graphviz(ADT.model,
                                    feature_names=features,
                                    class_names=['FAILURE', 'SUCCESS'],
                                    out_file=file_name,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)

            if to_file:
                ADT.store_metrics_to_file()

            # Save the model for future reproducibility
            if save_model:
                ADT.save_model(model_name)

            # TODO: testing part not implemented
