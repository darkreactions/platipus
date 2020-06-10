import torch
import numpy as np
import pickle

import os
import sys

from utils import *
from FC_net import FCNet
from platipus import *
from maml import *


from sklearn.metrics import confusion_matrix


def main(params):
    """Main driver code

    The main function to conduct PLATIPUS meta-training for each available amine.

    Args:
        N/A

    Returns:
        N/A
    """

    # Do the massive initialization and get back a dictionary instead of using global variables

    if params['train_flag']:
        if params['datasource'] == 'drp_chem':
            if params['cross_validate']:
                # TODO: This is going to be insanely nasty, basically reinitialize for each amine
                for amine in params['training_batches']:
                    print("Starting training for amine", amine)
                    # Change the path to save models
                    params['dst_folder'] = save_model("PLATIPUS", params, amine)

                    # Adjust the loss function for each amine
                    amine_counts = params['counts'][amine]
                    if amine_counts[0] >= amine_counts[1]:
                        weights = [amine_counts[0] / amine_counts[0],
                                   amine_counts[0] / amine_counts[1]]
                    else:
                        weights = [amine_counts[1] / amine_counts[0],
                                   amine_counts[1] / amine_counts[1]]

                    print('Using the following weights for loss function:', weights)
                    class_weights = torch.tensor(
                        weights, device=params['device'])
                    params['loss_fn'] = torch.nn.CrossEntropyLoss(
                        class_weights)

                    # Train the model then reinitialize a new one
                    params = meta_train_platipus(params, amine)
                    Theta = initialzie_theta_platipus(params)
                    params['Theta'] = Theta
                    params['op_Theta'] = set_optim_platipus(Theta, params["meta_lr"])
            else:
                params = meta_train_platipus(params)

    elif params['resume_epoch'] > 0:
        if params['datasource'] == 'drp_chem' and params['cross_validate']:
            # I am saving this dictionary in case things go wrong
            # It will get added to in the active learning code
            model_list = ["PLATIPUS", "MAML"]
            stats_dict = create_stats_dict(model_list)
            params['cv_statistics'] = stats_dict
            # Test performance of each individual cross validation model
            for amine in params['validation_batches']:
                print("Starting validation for amine", amine)
                # Change the path to save models
                params['dst_folder'] = save_model("PLATIPUS",params, amine)

                # Here we are loading a previously trained model
                saved_checkpoint = load_previous_model_platipus(params)

                Theta = saved_checkpoint['Theta']
                params['Theta'] = Theta

                # Adjust the loss function for each amine
                amine_counts = params['counts'][amine]
                if amine_counts[0] >= amine_counts[1]:
                    weights = [amine_counts[0] / amine_counts[0],
                               amine_counts[0] / amine_counts[1]]
                else:
                    weights = [amine_counts[1] / amine_counts[0],
                               amine_counts[1] / amine_counts[1]]
                print('Using the following weights for loss function:', weights)
                class_weights = torch.tensor(weights, device=params['device'])
                params['loss_fn'] = torch.nn.CrossEntropyLoss(class_weights)

                # Run forward pass on the validation data
                forward_pass_validate_platipus(params, amine)
                test_model_actively(params, amine)

            # Save this dictionary in case we need it later
            with open(os.path.join("./data", "cv_statistics.pkl"), "wb") as f:
                pickle.dump(params['cv_statistics'], f)

            with open(os.path.join("./data", "cv_statistics.pkl"), "rb") as f:
                params['cv_statistics'] = pickle.load(f)
            # Now plot the big graph
            stats_dict = params['cv_statistics']

            # Do some deletion so the average graph has no nan's
            # changed this to the new stats dict format
            for key in stats_dict.keys():
                for k in stats_dict[key].keys():
                    del stats_dict[key][k][10]
                    del stats_dict[key][k][11]

            for key in stats_dict.keys():
                for stat_list in stats_dict[key]["precisions"]:
                    print(stat_list[0])
                    if np.isnan(stat_list[0]):
                        stat_list[0] = 0

            # for key in stats_dict.keys():
            #     for stat_list in stats_dict[key]:
            #         del stat_list[0]

            # Find the minimum number of points for performance evaluation
            min_length = len(min(stats_dict["PLATIPUS"]['accuracies'], key=len))
            print(f'Minimum number of points we have to work with is {min_length}')

            # Evaluate all models' performances
            avg_stat = find_avg_metrics(stats_dict, min_length)

            # Graph all model's performances
            num_examples = [i for i in range(min_length)]
            plot_metrics_graph(num_examples, avg_stat, params['graph_folder'])

        # TEST CODE, SHOULD NOT BE RUN
        elif params['datasource'] == 'drp_chem' and not params['cross_validate']:
            print('If this code is running, you are doing something wrong. DO NOT TEST.')
            test_batches = params['testing_batches']
            for amine in test_batches:
                print("Checking for amine", amine)
                val_batch = test_batches[amine]
                x_t, y_t, x_v, y_v = torch.from_numpy(val_batch[0]).float().to(params['device']), torch.from_numpy(
                    val_batch[1]).long().to(params['device']), \
                    torch.from_numpy(val_batch[2]).float().to(params['device']), torch.from_numpy(
                    val_batch[3]).long().to(params['device'])

                accuracies = []
                corrects = []
                probability_pred = []
                sm_loss = params['sm_loss']
                preds = get_task_prediction_platipus(x_t, y_t, x_v, params)
                # print('raw task predictions', preds)
                y_pred_v = sm_loss(torch.stack(preds))
                # print('after applying softmax', y_pred_v)
                y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)
                # print('after calling torch mean', y_pred)

                prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
                # print('print training labels', y_t)
                print('print labels predicted', labels_pred)
                print('print true labels', y_v)
                print('print probability of prediction', prob_pred)
                correct = (labels_pred == y_v)
                corrects.extend(correct.detach().cpu().numpy())

                # print('length of validation set', len(y_v))
                accuracy = torch.sum(correct, dim=0).item() / len(y_v)
                accuracies.append(accuracy)

                probability_pred.extend(prob_pred.detach().cpu().numpy())

                print('accuracy for model is', accuracies)
                # print('probabilities for predictions are', probability_pred)
                test_model_actively(params, amine)

    else:
        sys.exit('Unknown action')


def test_model_actively(params, amine=None):
    """train and test the model with active learning

    Choosing the most uncertain point in the validation dataset to the training set for PLATIPUS
    and choose a random point in the validation dataset to the training set for MAML
    since MAML cannot reason about uncertainty. Then plot out the accuracy, precision, recall and
    balanced classification rate graph for each amine being trained or tested.
    """

    # Start by unpacking any necessary parameters
    device = params['device']
    gpu_id = params['gpu_id']
    num_training_samples_per_class = params['num_training_samples_per_class']
    num_classes_per_task = params['num_classes_per_task']
    # Assume root directory is current directory
    dst_folder_root = '.'
    num_epochs_save = params['num_epochs_save']

    # Running for a specific amine
    if params['cross_validate']:
        dst_folder_root = '.'
        sm_loss = params['sm_loss']

        # Load MAML data for comparison
        # This requires that we already trained a MAML model, see maml.py
        maml_folder = '{0:s}/MAML_few_shot/MAML_{1:s}_{2:d}way_{3:d}shot_{4:s}'.format(
            dst_folder_root,
            params['datasource'],
            params['num_classes_per_task'],
            params['num_training_samples_per_class'],
            amine
        )
        maml_filename = 'drp_chem_{0:d}way_{1:d}shot_{2:s}.pt'.format(num_classes_per_task,
                                                                      num_training_samples_per_class, '{0:d}')

        # Try to find the model that was training for the most epochs
        # If you get an error here, your MAML model was not saved at epochs of increment num_epochs_save
        # By default that value is 1000 and is set in initialize()
        i = num_epochs_save
        maml_checkpoint_filename = os.path.join(
            maml_folder, maml_filename.format(i))
        while (os.path.exists(maml_checkpoint_filename)):
            i = i + num_epochs_save
            maml_checkpoint_filename = os.path.join(
                maml_folder, maml_filename.format(i))

        # This is overshooting, just have it here as a sanity check
        print('loading from', maml_checkpoint_filename)

        if torch.cuda.is_available():
            maml_checkpoint = torch.load(
                os.path.join(maml_folder, maml_filename.format(
                    i - num_epochs_save)),
                map_location=lambda storage,
                loc: storage.cuda(gpu_id)
            )
        else:
            maml_checkpoint = torch.load(
                os.path.join(maml_folder, maml_filename.format(
                    i - num_epochs_save)),
                map_location=lambda storage,
                loc: storage
            )

        Theta_maml = maml_checkpoint['theta']

        validation_batches = params['validation_batches']
        val_batch = validation_batches[amine]
        x_t, y_t, x_v, y_v = torch.from_numpy(val_batch[0]).float().to(params['device']), torch.from_numpy(
            val_batch[1]).long().to(params['device']), \
            torch.from_numpy(val_batch[2]).float().to(params['device']), torch.from_numpy(
            val_batch[3]).long().to(params['device'])

        all_labels = torch.cat((y_t, y_v))
        all_data = torch.cat((x_t, x_v))

        accuracies = []
        corrects = []
        probability_pred = []
        confusion_matrices = []
        precisions = []
        recalls = []
        balanced_classification_rates = []

        num_examples = [0]

        # Set up the number of active learning iterations
        # Starting from 1 so that we can compare PLATIPUS/MAML with other models such as SVM and KNN that
        # have valid results from the first validation point.
        # For testing, overwrite with iters = 1
        iters = len(x_t) + len(x_v) - 1

        # Zero point prediction for PLATIPUS model
        print('Getting the model baseline before training on zero points')
        preds = get_naive_prediction_platipus(all_data, params)

        # Evaluate zero point performance for PLATIPUS
        prob_pred, correct, cm, accuracy, precision, recall, bcr = zero_point_platipus(preds, sm_loss, all_labels)

        # Display and update individual performance metric
        probability_pred.extend(prob_pred.detach().cpu().numpy())

        corrects.extend(correct.detach().cpu().numpy())

        print('accuracy for model is', accuracy)
        accuracies.append(accuracy)

        print(cm)
        confusion_matrices.append(cm)

        print('precision for model is', precision)
        precisions.append(precision)

        print('recall for model is', recall)
        recalls.append(recall)

        print('balanced classification rate for model is', bcr)
        balanced_classification_rates.append(bcr)

        # Randomly pick a point to start active learning with
        rand_index = np.random.choice(iters+1)

        # Reset the training set and validation set
        x_t, x_v = all_data[rand_index].view(-1, 51), torch.cat(
            [all_data[0:rand_index], all_data[rand_index + 1:]])
        y_t, y_v = all_labels[rand_index].view(1), torch.cat(
            [all_labels[0:rand_index], all_labels[rand_index + 1:]])

        for _ in range(iters):
            print(f'Doing active learning with {len(x_t)} examples')
            num_examples.append(len(x_t))
            preds = get_task_prediction_platipus(x_t, y_t, all_data, params)

            # Update available datapoints in the pool and evaluate current model performance
            x_t, y_t, x_v, y_v, prob_pred, correct, cm, accuracy, precision, recall, bcr = active_learning_platipus(
                preds, sm_loss, all_labels, params, x_t, y_t, x_v, y_v)

            # Display and update individual performance metric
            probability_pred.extend(prob_pred.detach().cpu().numpy())
            corrects.extend(correct.detach().cpu().numpy())

            print(cm)
            confusion_matrices.append(cm)

            print('accuracy for model is', accuracy)
            accuracies.append(accuracy)

            print('precision for model is', precision)
            precisions.append(precision)

            print('recall for model is', recall)
            recalls.append(recall)

            print('balanced classification rate for model is', bcr)
            balanced_classification_rates.append(bcr)

        # Now do it again but for the MAML model
        accuracies_MAML = []
        corrects_MAML = []
        confusion_matrices_MAML = []
        precisions_MAML = []
        recalls_MAML = []
        balanced_classification_rates_MAML = []

        num_examples = [0]

        # Set up softmax loss function for maml
        sm_loss_maml = torch.nn.Softmax(dim=1)

        # Zero point prediction for MAML model
        print('Getting the MAML model baseline before training on zero points')
        preds = get_naive_task_prediction_maml(all_data, Theta_maml, params)

        # Evaluate zero point performance for MAML
        correct, accuracy, cm, precision, recall, bcr = zero_point_maml(preds, sm_loss_maml, all_labels)

        # Display and update individual performance metric
        corrects_MAML.extend(correct.detach().cpu().numpy())

        print(cm)
        confusion_matrices_MAML.append(cm)

        print('accuracy for model is', accuracy)
        accuracies_MAML.append(accuracy)

        print('precision for model is', precision)
        precisions_MAML.append(precision)

        print('recall for model is', recall)
        recalls_MAML.append(recall)

        print('balanced classification rate for model is', bcr)
        balanced_classification_rates_MAML.append(bcr)

        # Reset the training and validation data for MAML
        x_t, x_v = all_data[rand_index].view(1, 51), torch.cat(
            [all_data[0:rand_index], all_data[rand_index + 1:]])
        y_t, y_v = all_labels[rand_index].view(1), torch.cat(
            [all_labels[0:rand_index], all_labels[rand_index + 1:]])

        for i in range(iters):
            print(f'Doing MAML learning with {len(x_t)} examples')
            num_examples.append(len(x_t))
            preds = get_task_prediction_maml(x_t=x_t, y_t=y_t, x_v=all_data, meta_params=Theta_maml, params=params)

            # Update available datapoints in the pool and evaluate current model performance
            x_t, y_t, x_v, y_v, correct, cm, accuracy, precision, recall, bcr = active_learning_maml(
                preds, sm_loss_maml, all_labels, x_t, y_t, x_v, y_v
            )

            # Display and update individual performance metric
            corrects_MAML.extend(correct.detach().cpu().numpy())

            print(cm)
            confusion_matrices_MAML.append(cm)

            print('accuracy for model is', accuracy)
            accuracies_MAML.append(accuracy)

            print('precision for model is', precision)
            precisions_MAML.append(precision)

            print('recall for model is', recall)
            recalls_MAML.append(recall)

            print('balanced classification rate for model is', bcr)
            balanced_classification_rates_MAML.append(bcr)

        # TODO: make sure the stats_dict passed in has the hierachy: {model_name: metric_name: [metrics]}
        plot_metrics_graph(num_examples, stats_dict, params['active_learning_graph_folder'], amine=amine)

        params['cv_statistics']['accuracies'].append(accuracies)
        params['cv_statistics']['confusion_matrices'].append(
            confusion_matrices)
        params['cv_statistics']['precisions'].append(precisions)
        params['cv_statistics']['recalls'].append(recalls)
        params['cv_statistics']['balanced_classification_rates'].append(
            balanced_classification_rates)
        params['cv_statistics']['accuracies_MAML'].append(accuracies_MAML)
        params['cv_statistics']['confusion_matrices_MAML'].append(
            confusion_matrices_MAML)
        params['cv_statistics']['precisions_MAML'].append(precisions_MAML)
        params['cv_statistics']['recalls_MAML'].append(recalls_MAML)
        params['cv_statistics']['balanced_classification_rates_MAML'].append(
            balanced_classification_rates_MAML)

    # Here we are testing, this code should NOT be run yet
    elif params['datasource'] == 'drp_chem' and not params['cross_validate']:
        testing_batches = params['testing_batches']
        test_batch = testing_batches[amine]
        x_t, y_t, x_v, y_v = torch.from_numpy(test_batch[0]).float().to(params['device']), torch.from_numpy(
            test_batch[1]).long().to(params['device']), \
            torch.from_numpy(test_batch[2]).float().to(params['device']), torch.from_numpy(
            test_batch[3]).long().to(params['device'])

        all_labels = torch.cat((y_t, y_v))
        all_data = torch.cat((x_t, x_v))

        accuracies = []
        corrects = []
        probability_pred = []
        confusion_matrices = []
        precisions = []
        recalls = []
        balanced_classification_rates = []
        sm_loss = params['sm_loss']

        num_examples = []
        iters = len(x_v)

        for i in range(iters):
            print(f'Doing active learning with {len(x_t)} examples')
            num_examples.append(len(x_t))
            preds = get_task_prediction_platipus(x_t, y_t, all_data, params)

            # Update available datapoints in the pool and evaluate current model performance
            x_t, y_t, x_v, y_v, prob_pred, correct, cm, accuracy, precision, recall, bcr = active_learning_platipus(
                preds, sm_loss, all_labels, params, x_t, y_t, x_v, y_v)

            # Display and update individual performance metric
            probability_pred.extend(prob_pred.detach().cpu().numpy())
            corrects.extend(correct.detach().cpu().numpy())

            print(cm)
            confusion_matrices.append(cm)

            print('accuracy for model is', accuracy)
            accuracies.append(accuracy)

            print('precision for model is', precision)
            precisions.append(precision)

            print('recall for model is', recall)
            recalls.append(recall)

            print('balanced classification rate for model is', bcr)
            balanced_classification_rates.append(bcr)

        # TODO: make sure the stats_dict get passed
        plot_metrics_graph(num_examples, stats_dict, params['active_learning_graph_folder'], amine=amine)


def create_stats_dict(models):
    """Creating the stats dictionary

    Args:
        model: A list of the models that we are creating metrics for

    return: A dictionary with format: {"model":{"metric1":[],"metric2":[], etc}, "model":{"metric1":[], etc}}
    """
    stats_dict = {}
    metrics = ['accuracies', 'confusion_matrices', 'precisions', 'recalls', 'bcrs']
    for model in models:
        stats_dict[model] = {}
        for key in metrics:
            stats_dict[model][key] = []
    return stats_dict



if __name__ == "__main__":
    main()
