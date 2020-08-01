import itertools
from pathlib import Path
import pickle
import torch

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform

from utils.dataset import process_dataset, import_test_dataset, import_full_dataset

# TODO DELETE
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


def write_pickle(path, data):
    """Write pickle file

    Save for reproducibility

    Args:
        path: The path we want to write the pickle file at
        data: The data we want to save in the pickle file
    """
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def read_pickle(path):
    """Read pickle file

    Make sure we don't overwrite our batches if we are validating and testing

    Args:
        path: The path we want to check the batches

    return: Data that we already stored in the pickle file
    """
    path = Path(path)
    data = None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_chem_dataset(k_shot, cross_validation=True, meta_batch_size=32,
                      num_batches=100, verbose=False, test=False):
    """Load in the chemistry data for training

    "I'm limited by the technology of my time."
    Ideally the model would choose from a Uniformly distributed pool of unlabeled reactions
    Then we would run that reaction in the lab and give it back to the model
    The issue is the labs are closed, so we have to restrict the model to the reactions drawn
    from uniform distributions that we have labels for
    The list below is a list of inchi keys for amines that have a reaction drawn from a uniform
    distribution with a successful outcome (no point in amines that have no successful reaction)

    Args:
        k_shot:             An integer. The number of unseen classes in the dataset
        params:             A dictionary. The dictionary that is initialized with parameters.
                            Use the key "cross_validate" in the dictionary to separate
                            loading training data and testing data
        meta_batch_size:    An integer. The batch size for meta learning, default is 32
        num_batches:        An integer. The batch size for training, default is 100
        verbose:            A boolean that gives information about
                            the number of features to train on is

    return:
        amine_left_out_batches:         A dictionary of batches with structure:
                                        key is amine left out,
                                        value has following hierarchy
                                        batches -> x_t, y_t, x_v, y_v -> meta_batch_size number of amines ->
                                        k_shot number of reactions -> each reaction has some number of features
        amine_cross_validate_samples:   A dictionary of batches with structure:
                                        key is amine which the data is for,
                                        value has the following hierarchy
                                        x_s, y_s, x_q, y_q -> k_shot number of reactions ->
                                        each reaction has some number of features
        amine_test_samples:             A dictionary that has the same structure as
                                        amine_cross_validate_samples
        counts:                         A dictionary to record the number of
                                        successful and failed reactions in the format of
                                        {"total": [# of failed reactions, # of successful reactions]}
    """
    if test:
        print('Getting Test dataset')
        return import_test_dataset(k_shot, meta_batch_size, num_batches, verbose=verbose,
                                   cross_validation=cross_validation)
    else:
        print('Getting FULL dataset')
        return import_full_dataset(k_shot, meta_batch_size, num_batches, verbose=verbose,
                                   cross_validation=cross_validation)


def find_avg_metrics(stats_dict, models, min_length=None):
    """Calculate the average metrics of several models' performances

    Args:
        stats_dict:         A dictionary representing the performance metrics of the machine learning models.
                                It has the format of {model_name: {metric_name: [[metric_values for each amine]]}}
        modes:              A list of the models that we want to find the average metrics for
        min_length:         An integer representing the fewest number of points to start metrics calculations.

    Returns:
        avg_stat:           A dictionary representing the average performance metrics of each model.
                                It has the format of {model_name: {metric_name: [avg_metric_values]}}.
    """

    # Set up default dictionary to store average metrics for each model
    avg_stat = {}

    all_models = models
    """random_model = all_models[0]

    if not min_length:
        min_length = len(stats_dict[random_model]['accuracies'][0])
        for model in all_models:
            length = len(min(stats_dict[model]['accuracies'], key=len))
            if min_length > length:
                min_length = length"""

    # Calculate average metrics by model
    for model in all_models:
        # Pre-fill each model's value with standard metrics dictionary
        avg_stat[model] = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'bcrs': []
        }
        max_length = len(max(stats_dict[model]['accuracies'], key=len))

        for i in range(max_length):

            total = 0
            num_points = 0
            for amine_metric in stats_dict[model]['accuracies']:
                if i < len(amine_metric):
                    total += amine_metric[i]
                    num_points += 1
            # avg_acc = total / len(stats_dict[model]['accuracies'])
            avg_acc = total / num_points
            avg_stat[model]['accuracies'].append(avg_acc)

            total = 0
            num_points = 0
            for amine_metric in stats_dict[model]['precisions']:
                if i < len(amine_metric):
                    total += amine_metric[i]
                    num_points += 1
            # avg_prec = total / len(stats_dict[model]['precisions'])
            avg_prec = total / num_points
            avg_stat[model]['precisions'].append(avg_prec)

            total = 0
            num_points = 0
            for amine_metric in stats_dict[model]['recalls']:
                if i < len(amine_metric):
                    total += amine_metric[i]
                    num_points += 1
            # avg_rec = total / len(stats_dict[model]['recalls'])
            avg_rec = total / num_points
            avg_stat[model]['recalls'].append(avg_rec)

            total = 0
            num_points = 0
            for amine_metric in stats_dict[model]['bcrs']:
                if i < len(amine_metric):
                    total += amine_metric[i]
                    num_points += 1
            # avg_bcr = total / len(stats_dict[model]['bcrs'])
            avg_bcr = total / num_points
            avg_stat[model]['bcrs'].append(avg_bcr)

    return avg_stat


def find_winning_models(avg_stat,all_cat):
    """Find the winning models for each category of models

    Args:
        avg_stat:           A dictionary representing the average performance metrics of the machine learning models.
                                It has the format of {model_name: {metric_name: [[avg_metric_values for model]]}}
        all_cat:            A dictionary representing the different non_meta models in each category
                                It has the format of {category: [all the models in this category]}

    Returns:
        best_model_bcr:     A dictionary representing the best performing models according to their auc_to_bcr.
                                It has the format of {model_name: {metric_name: [metric_values]}}.
    """
    best_model_bcr = {}
    cats = list(all_cat.keys())
    # Find the best performing model in each category
    for cat in cats:
        cat_models = all_cat[cat]
        cat_best_bcr = {}
        for model in list(avg_stat.keys()):
            if model in cat_models and 'PLATIPUS' not in model:
                cat_best_bcr[model] = avg_stat[model]['bcrs']
        cat_best_bcr = max(cat_best_bcr, key=lambda x: np.trapz(cat_best_bcr[x]) / len(cat_best_bcr[x]))
        best_model_bcr[cat_best_bcr] = avg_stat[cat_best_bcr]
    # add the platipus line into the graph to compare with the best performing ones
    for model in list(avg_stat.keys()):
        if 'PLATIPUS' in model:
            best_model_bcr[model] = avg_stat[model]
    return best_model_bcr


def save_model(model, params, amine=None):
    """This is to save models

    Create specific folders and store the model in the folder

    Args:
        model:  A string that indicates which model we are using
        params: A dictionary of the initialized parameters
        amine:  The specific amine that we want to store models for.
                Default is None

    return: The path for dst_folder
    """
    # Make sure we are creating directory for all models
    dst_folder_root = './results'
    dst_folder = ""
    k_shot = params['k_shot']
    if amine and (amine in params['training_batches'] or amine in params['validation_batches']):
        dst_folder = Path(f'{dst_folder_root}/{model}_{k_shot}_shot/{amine}')
        # return dst_folder
    else:
        dst_folder = Path(f'{dst_folder_root}/{model}_{k_shot}_shot/testing')

    dst_folder.mkdir(parents=True, exist_ok=True)
    return dst_folder


def initialise_dict_of_dict(key_list):
    """Initialize a dictionary within a dictionary

    Helps us create a data structure to store model weights during gradient updates

    Args:
        key_list: A list of keys that will be initialized
        in each of the keys in the outer-most dictionary

    return:
        q: A dictionary that has a dictionary as the index of each key.
        In the format of {"key":{"key":0}}
    """

    q = dict.fromkeys(['mean', 'logSigma'])
    for para in q.keys():
        q[para] = {}
        for key in key_list:
            q[para][key] = 0
    return q


def create_stats_dict(models):
    """Creating the stats dictionary

    Args:
        model: A list of the models that we are creating metrics for

    return: A dictionary with format: {"model":{"metric1":[],"metric2":[], etc}, "model":{"metric1":[], etc}}
    """
    stats_dict = {}
    metrics = ['accuracies', 'confusion_matrices',
               'precisions', 'recalls', 'bcrs']
    for model in models:
        stats_dict[model] = {}
        for key in metrics:
            stats_dict[model][key] = []
    return stats_dict


def create_cv_stats_dict(models):
    """Creates a stats dictionary that stores the performance metrics during the cross-validation stage of all models on
            a specific amine.

    Args:
        models:         A list representing all the models to be evaluated.

    Returns:
        cv_stats_dict:  A dictionary that stores the performance metrics during the cross-validation stage of a specific
                            amine. It has the format of {model_name:{metric_name: [metric_value]}}.
    """
    metrics = {
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'bcrs': [],
        'corrects': [],
        'confusion_matrices': [],
        'prob_pred': []
    }

    cv_stats_dict = {}

    for model in models:
        # Pre-fill each model's value with standard metrics dictionary
        cv_stats_dict.setdefault(model, metrics)

    return cv_stats_dict


def update_cv_stats_dict(cv_stats_dict, model, correct, cm, accuracy, precision, recall, bcr, prob_pred=None,
                         verbose=True):
    """Update the stats dictionary that stores the performance metrics during the cross-validation stage of a specific
            amine
    Args:
        cv_stats_dict:  A dictionary that stores the performance metrics during the cross-validation stage of a specific
                            amine. It has the format of {model_name:{metric_name: [metric_value]}}.
        model:          A string representing the ML model the metrics are evaluating. Should be either 'PLATIPUS' or
                            'MAML'.
        correct:        An torch.Tensor object representing an array-like element-wise comparison between the actual
                            labels and predicted labels.
        cm:             A numpy array representing the confusion matrix given our predicted labels and the actual
                            corresponding labels. It's a 2x2 matrix for the drp_chem model.
        accuracy:       A float representing the accuracy rate of the model: the rate of correctly predicted reactions
                            out of all reactions.
        precision:      A float representing the precision rate of the model: the rate of the number of actually
                            successful reactions out of all the reactions predicted to be successful.
        recall:         A float representing the recall rate of the model: the rate of the number of reactions predicted
                            to be successful out of all the acutal successful reactions.
        bcr:            A float representing the balanced classification rate of the model. It's the average value of
                            recall rate and true negative rate.
        prob_pred:      A torch.Tensor object representing all the probabilities of the current predictions of all data
                            points w/o active learning. Default to None since the model can be MAML.
        verbose:        A string representing whether we want to print the metrics out or not.

    Returns:
        cv_stats_dict:  A dictionary that stores the performance metrics during the cross-validation stage of a specific
                            amine. It has the same format as above, with values updated given the input metrics.
    """
    # Display and update individual performance metric
    cv_stats_dict[model]['corrects'].extend(correct.detach().cpu().numpy())
    cv_stats_dict[model]['accuracies'].append(accuracy)
    cv_stats_dict[model]['confusion_matrices'].append(cm)
    cv_stats_dict[model]['precisions'].append(precision)
    cv_stats_dict[model]['recalls'].append(recall)
    cv_stats_dict[model]['bcrs'].append(bcr)

    if isinstance(prob_pred, torch.Tensor):
        cv_stats_dict[model]['prob_pred'].extend(
            prob_pred.detach().cpu().numpy())

    if verbose:
        print('accuracy for model is', accuracy)
        print(cm)
        print('precision for model is', precision)
        print('recall for model is', recall)
        print('balanced classification rate for model is', bcr)

    return cv_stats_dict


def find_success_rate():
    missing_from_inventory = ['UMDDLGMCNFAZDX-UHFFFAOYSA-O']

    missing_from_volumes = ['KFQARYBEAKAXIC-UHFFFAOYSA-N',
                            'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                            'FJFIJIDZQADKEE-UHFFFAOYSA-N',
                            'BAMDIFIROXTEEM-UHFFFAOYSA-N',
                            'XZUCBFLUEBDNSJ-UHFFFAOYSA-N']

    missing_from_viable = ['QHJPGANWSLEMTI-UHFFFAOYSA-N',
                           'GGYGJCFIYJVWIP-UHFFFAOYSA-N',
                           'PXWSKGXEHZHFJA-UHFFFAOYSA-N',
                           'DMFMZFFIQRMJQZ-UHFFFAOYSA-N',
                           'NOHLSFNWSBZSBW-UHFFFAOYSA-N']

    distribution_header = '_raw_modelname'
    amine_header = '_rxn_organic-inchikey'
    score_header = '_out_crystalscore'
    name_header = 'name'
    SUCCESS = 4
    viable_amines = ['ZEVRFFCPALTVDN-UHFFFAOYSA-N',
                     'KFQARYBEAKAXIC-UHFFFAOYSA-N',
                     'NLJDBTZLVTWXRG-UHFFFAOYSA-N',
                     'LCTUISCIGMWMAT-UHFFFAOYSA-N',
                     'JERSPYRKVMAEJY-UHFFFAOYSA-N',
                     'JMXLWMIFDJCGBV-UHFFFAOYSA-N',
                     'VAWHFUNJDMQUSB-UHFFFAOYSA-N',
                     'WGYRINYTHSORGH-UHFFFAOYSA-N',
                     'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                     'VNAAUNTYIONOHR-UHFFFAOYSA-N',
                     'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                     'FJFIJIDZQADKEE-UHFFFAOYSA-N',
                     'XFYICZOIWSBQSK-UHFFFAOYSA-N',
                     'HBPSMMXRESDUSG-UHFFFAOYSA-N',
                     'NXRUEVJQMBGVAT-UHFFFAOYSA-N',
                     'CALQKRVFTWDYDG-UHFFFAOYSA-N',
                     'LLWRXQXPJMPHLR-UHFFFAOYSA-N',
                     'BAMDIFIROXTEEM-UHFFFAOYSA-N',
                     'XZUCBFLUEBDNSJ-UHFFFAOYSA-N']

    held_out = ['CALQKRVFTWDYDG-UHFFFAOYSA-N',
                'KOAGKPNEVYEZDU-UHFFFAOYSA-N',
                'FCTHQYIDLRRROX-UHFFFAOYSA-N',
                'JMXLWMIFDJCGBV-UHFFFAOYSA-N']

    viable_amines = [a for a in viable_amines if a not in held_out]

    df = pd.read_csv('./data/0050.perovskitedata_DRP.csv')
    df = df[df[distribution_header].str.contains('Uniform')]
    df = df[df[amine_header].isin(viable_amines)]
    df[score_header] = [1 if val == SUCCESS else 0 for val in df[score_header].values]
    amines = df[amine_header].unique().tolist()
    inventory = pd.read_csv('./data/inventory.csv')
    percent_volume = pd.read_csv('./data/percent.csv', sep=',')

    percent_volume['Full Name'].str.lower()
    inventory['Chemical Name'].str.lower()

    total = percent_volume.set_index('Full Name').join(inventory.set_index('Chemical Name'))

    # percent_vol = []

    filtered = percent_volume[percent_volume['Inchi'].isin(amines)].sort_values(['Success'], axis=0, ascending=False)
    amines_new = filtered['Inchi'].values

    success_rate = filtered['Success'].to_numpy()
    success_rate = np.expand_dims(success_rate, axis=1)

    counts1 = []
    counts0 = []
    names = []
    percent_success = []
    for amine in amines_new:
        results = df[df[amine_header] == amine][score_header].value_counts()
        name = inventory[inventory['InChI Key (ID)'] == amine]['InChI Key (ID)'].values
        names.append(name[0])
        counts0.append(results[0])
        counts1.append(results[1])
        percent_success.append(results[1] * 100 / (results[1] + results[0]))
    percent_success = np.array(percent_success)
    percent_success = np.expand_dims(percent_success, axis=1)

    return names, success_rate, percent_success


def find_bcr(category, cv_stats, amine_names):
    """
    for plotting the success rate graph
    """
    wanted_index = []
    for amine in amine_names:
        i = 0
        found = False
        while i < len(cv_stats[category]['amine']) and not found:
            if cv_stats[category]['amine'][i] == amine:
                wanted_index.append(i)
                found = True
            else:
                i += 1
    wanted_bcrs = []
    wanted_names = []
    for index in wanted_index:
        wanted_bcrs.append(cv_stats[category]['bcrs'][index][-1])
        wanted_names.append(cv_stats[category]['amine'][index])
    # print(wanted_bcrs)
    # print(wanted_names)
    return wanted_bcrs


def define_non_meta_model_name(model_name, active_learning, w_hx, w_k):
    """ Function to define the suffix of non-meta model
    Args:
        model_name:         A string representing the base model name of the non-meta model.
        active_learning:    A boolean representing if the model will conduct active learning or not.
        w_hx:               A boolean representing if the model will be trained with historical data or not.
        w_k:                A boolean representing if the model will be trained with k additional experiments of the
                                task-specific experiments or not.
    returns:
        A string representing the model name with proper suffix
    """

    suffix = ''

    if not active_learning:
        if w_hx:
            if not w_k:
                suffix = '_historical_only'
            else:
                suffix = '_historical_amine'
        else:
            if w_k:
                suffix = '_amine_only'
            else:
                print('Invalid combination of parameters.')
                print("Can't find appropriate category for the model.")
                print("Using default name instead.")
    else:
        if w_hx:
            if w_k:
                suffix = '_historical_amine_AL'
            else:
                print('Invalid combination of parameters.')
                print("Can't find appropriate category for the model.")
                print("Using default name instead.")
        else:
            if w_k:
                suffix = '_amine_only_AL'
            else:
                print('Invalid combination of parameters.')
                print("Can't find appropriate category for the model.")
                print("Using default name instead.")

    return model_name + suffix


def run_non_meta_model(base_model, common_params, model_params, category):
    """Run non-meta models under desired category

    Args:
        base_model:         The base non-meta machine learning model to be run.
        common_params:      A dictionary representing the common parameters used across all models.
        model_params:       A dictionary representing the base-model specific parameters.
        category:           A string representing the category of the model to be run.
    """

    # Set up the settings of each category
    # The entries correspond to: With active learning? With historical data? With amine data?
    settings = {
        'category_3': [False, True, False],
        'category_4_i': [False, True, True],
        'category_4_ii': [False, False, True],
        'category_5_i': [True, True, True],
        'category_5_ii': [True, False, True],
    }

    # Set up the aggregated parameter dictionary
    base_model_params = {**common_params, **model_params}

    # Change the three settings based on the input category
    base_model_params['active_learning'] = settings[category][0]
    base_model_params['with_historical_data'] = settings[category][1]
    base_model_params['with_k'] = settings[category][2]

    # Define the model's name given the category it is in
    base_model_params['model_name'] = define_non_meta_model_name(
        base_model_params['model_name'],
        base_model_params['active_learning'],
        base_model_params['with_historical_data'],
        base_model_params['with_k'])

    # Run the non-meta models
    base_model.run_model(base_model_params, category)


def grid_search(clf, params, train_size, active_learning_iter, active_learning=True, w_hx=True, w_k=True, random=False,
                random_size=10, info=False):
    """Fine tune the model based on average bcr performance to find the best model hyper-parameters.

    Similar to GridSearchCV in scikit-learn package, we try out all the combinations and evaluate performance
        across all amine-specific models under different categories.

    Args:
        clf:                        A class object representing the classifier being fine tuned.
        params:                     A dictionary representing the possible hyper-parameter values to try out.
        train_size:                 An integer representing the number of amine-specific experiments used for training.
                                        Corresponds to the k in the category description.
        active_learning_iter:       An integer representing the number of iterations in an active learning loop.
                                        Corresponds to the x in the category description.
        active_learning:            A boolean representing if active learning will be involved in testing or not.
        w_hx:                       A boolean representing if the models are trained with historical data or not.
        w_k:                        A boolean representing if the modes are trained with amine-specific experiments.
        random:                     A boolean representing if we want to do random search or not.
        random_size:                An integer representing the number of random combinations to try and compare.
        info:                       A boolean. Setting it to True will make the function print out additional
                                        information during the fine-tuning stage.
                                        Default to False.
    Returns:
        best_option:                A dictionary representing the hyper-parameters that yields the best performance on
                                        average. The keys may vary for models.
    """

    # Set all possible combinations
    combinations = []

    keys, values = zip(*params.items())
    for bundle in itertools.product(*values):
        combinations.append(dict(zip(keys, bundle)))

    # In case we want to decrease the run time
    # by doing random search
    if random:
        combinations = list(np.random.choice(combinations, size=random_size))

    # Load the full dataset under specific categorical option
    amine_list, train_data, train_labels, val_data, val_labels, all_data, all_labels = process_dataset(
        train_size=train_size,
        active_learning_iter=active_learning_iter,
        verbose=False,
        cross_validation=True,
        full=True,
        active_learning=active_learning,
        w_hx=w_hx,
        w_k=w_k
    )

    # Set baseline performance
    base_accuracies = []
    base_precisions = []
    base_recalls = []
    base_bcrs = []
    base_aucs = []

    for amine in amine_list:
        ACLF = clf(amine=amine, verbose=False)

        # Exact and load the training and validation set into the model
        x_t, y_t = train_data[amine], train_labels[amine]
        x_v, y_v = val_data[amine], val_labels[amine]
        all_task_data, all_task_labels = all_data[amine], all_labels[amine]
        ACLF.load_dataset(x_t, y_t, x_v, y_v, all_task_data, all_task_labels)

        ACLF.train(warning=False)

        # Calculate AUC
        auc = roc_auc_score(ACLF.all_labels, ACLF.y_preds)

        base_accuracies.append(ACLF.metrics['accuracies'][-1])
        base_precisions.append(ACLF.metrics['precisions'][-1])
        base_recalls.append(ACLF.metrics['recalls'][-1])
        base_bcrs.append(ACLF.metrics['bcrs'][-1])
        base_aucs.append(auc)

    # Calculated the average baseline performances
    base_avg_accuracy = sum(base_accuracies) / len(base_accuracies)
    base_avg_precision = sum(base_precisions) / len(base_precisions)
    base_avg_recall = sum(base_recalls) / len(base_recalls)
    base_avg_bcr = sum(base_bcrs) / len(base_bcrs)
    base_avg_auc = sum(base_aucs) / len(base_aucs)

    best_metric = base_avg_auc

    if info:
        print(f'Baseline average accuracy is {base_avg_accuracy}')
        print(f'Baseline average precision is {base_avg_precision}')
        print(f'Baseline average recall is {base_avg_recall}')
        print(f'Baseline average bcr is {base_avg_bcr}')
        print(f'Baseline average auc is {base_avg_auc}')

    best_option = {}

    option_no = 1

    # Try out each possible combinations of hyper-parameters
    print(f'There are {len(combinations)} many combinations to try.')
    for option in combinations:
        accuracies = []
        precisions = []
        recalls = []
        bcrs = []
        aucs = []

        print(f'Trying option {option_no}')
        option_no += 1
        for amine in amine_list:
            # print("Training and cross validation on {} amine.".format(amine))
            ACLF = clf(amine=amine, config=option, verbose=False)

            # Exact and load the training and validation set into the model
            x_t, y_t = train_data[amine], train_labels[amine]
            x_v, y_v = val_data[amine], val_labels[amine]
            all_task_data, all_task_labels = all_data[amine], all_labels[amine]

            ACLF.load_dataset(x_t, y_t, x_v, y_v, all_task_data, all_task_labels)
            ACLF.train(warning=False)

            # Calculate AUC
            auc = roc_auc_score(ACLF.all_labels, ACLF.y_preds)

            accuracies.append(ACLF.metrics['accuracies'][-1])
            precisions.append(ACLF.metrics['precisions'][-1])
            recalls.append(ACLF.metrics['recalls'][-1])
            bcrs.append(ACLF.metrics['bcrs'][-1])
            aucs.append(auc)

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_bcr = sum(bcrs) / len(bcrs)
        avg_auc = sum(aucs) / len(aucs)

        if avg_auc > best_metric:
            if info:
                print(f'The previous best option is {best_option}')
                print(f'The current best setting is {option}')
                print(f'The fine-tuned average accuracy is {avg_accuracy} vs. the base accuracy {base_avg_accuracy}')
                print(
                    f'The fine-tuned average precision is {avg_precision} vs. the base precision {base_avg_precision}')
                print(f'The fine-tuned average recall rate is {avg_recall} vs. the base recall rate {base_avg_recall}')
                print(f'The fine-tuned average bcr is {avg_bcr} vs. the base bcr {base_avg_bcr}')
                print(f'The fine-tuned average auc is {avg_auc} vs. the base auc {base_avg_auc}')
                print()

            best_metric = avg_auc
            best_option = option

    if info:
        print()
        print(f'The best setting for all amines is {best_option}')
        print(f'With an average auc of {best_metric}')

    return best_option


# Credit: https://github.com/rlphilli/sklearn-PUK-kernel
def PUK_kernel(X1, X2, sigma=1.0, omega=1.0):
    # Compute the kernel matrix between two arrays using the Pearson VII function-based universal kernel.

    # Compute squared euclidean distance between each row element pair of the two matrices
    if X1 is X2:
        kernel = squareform(pdist(X1, 'sqeuclidean'))
    else:
        kernel = cdist(X1, X2, 'sqeuclidean')

    kernel = (1 + (kernel * 4 * np.sqrt(2**(1.0/omega)-1)) / sigma**2) ** omega
    kernel = 1/kernel

    return kernel


if __name__ == "__main__":
    params = {"cross_validate": True}
    load_chem_dataset(5, params, meta_batch_size=32,
                      num_batches=100, verbose=True)
