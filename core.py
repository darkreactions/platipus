from maml import initialize
import torch
import numpy as np
import pickle

import os
import sys

from utils import (load_chem_dataset, plot_metrics_graph, create_cv_stats_dict,
                   update_cv_stats_dict)

from FC_net import FCNet
from platipus import (initialzie_theta_platipus, save_model,
                      meta_train_platipus, set_optim_platipus,
                      load_previous_model_platipus, forward_pass_validate_platipus,
                      find_avg_metrics, get_task_prediction_platipus,
                      get_naive_prediction_platipus, zero_point_platipus,
                      active_learning_platipus)
from maml import (initialize, load_previous_model_maml,
                  get_naive_task_prediction_maml, zero_point_maml,
                  get_task_prediction_maml, active_learning_maml)


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
                    params['dst_folder'] = save_model(
                        "PLATIPUS", params, amine)

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
                    params['op_Theta'] = set_optim_platipus(
                        Theta, params["meta_lr"])
            else:
                params = meta_train_platipus(params)

    elif params['resume_epoch'] > 0:
        if params['datasource'] == 'drp_chem' and params['cross_validate']:
            # I am saving this dictionary in case things go wrong
            # It will get added to in the active learning code
            # Test performance of each individual cross validation model
            for amine in params['validation_batches']:
                print("Starting validation for amine", amine)
                # Change the path to save models
                params['dst_folder'] = save_model("PLATIPUS", params, amine)

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
            min_length = len(
                min(stats_dict["PLATIPUS"]['accuracies'], key=len))
            print(
                f'Minimum number of points we have to work with is {min_length}')

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

    # Assume root directory is current directory
    dst_folder_root = '.'

    # Running cross validation for a specific amine
    if params['cross_validate']:
        # List out the models we want to conduct active learning
        models = ['PLATIPUS', 'MAML']

        # Create the stats dictionary to store performance metrics
        cv_stats_dict = create_cv_stats_dict(models)

        # Set up softmax loss function for PLATIPUS and MAML
        sm_loss_platipus = params['sm_loss']
        sm_loss_maml = torch.nn.Softmax(dim=1)

        # Assign the validation batch to be used given an amine
        validation_batches = params['validation_batches']
        val_batch = validation_batches[amine]

        for model in models:
            # Initialize the training and the active learning pool for model
            x_t, y_t, x_v, y_v = torch.from_numpy(val_batch[0]).float().to(params['device']), torch.from_numpy(
                val_batch[1]).long().to(params['device']), \
                torch.from_numpy(val_batch[2]).float().to(params['device']), torch.from_numpy(
                val_batch[3]).long().to(params['device'])

            all_labels = torch.cat((y_t, y_v))
            all_data = torch.cat((x_t, x_v))

            # Pre-fill num_examples for zero-point evaluation
            num_examples = [0]

            # Set up the number of active learning iterations
            # Starting from 1 so that we can compare PLATIPUS/MAML with other models such as SVM and KNN that
            # have valid results from the first validation point.
            # For testing, overwrite with iters = 1
            iters = len(x_t) + len(x_v) - 1

            # Randomly pick a point to start active learning with
            rand_index = np.random.choice(iters + 1)

            if model == 'PLATIPUS':

                # Zero point prediction for PLATIPUS model
                print(
                    'Getting the PLATIPUS model baseline before training on zero points')
                preds = get_naive_prediction_platipus(all_data, params)

                # Evaluate zero point performance for PLATIPUS
                prob_pred, correct, cm, accuracy, precision, recall, bcr = zero_point_platipus(preds, sm_loss_platipus,
                                                                                               all_labels)

                # Display and update individual performance metric
                cv_stats_dict = update_cv_stats_dict(cv_stats_dict, model, correct, cm, accuracy, precision,
                                                     recall, bcr, prob_pred=prob_pred, verbose=params['verbose'])

                # Reset the training set and validation set
                x_t, x_v = all_data[rand_index].view(-1, 51), torch.cat(
                    [all_data[0:rand_index], all_data[rand_index + 1:]])
                y_t, y_v = all_labels[rand_index].view(1), torch.cat(
                    [all_labels[0:rand_index], all_labels[rand_index + 1:]])

                for _ in range(iters):
                    print(f'Doing active learning with {len(x_t)} examples')
                    num_examples.append(len(x_t))
                    preds = get_task_prediction_platipus(
                        x_t, y_t, all_data, params)

                    # Update available datapoints in the pool and evaluate current model performance
                    x_t, y_t, x_v, y_v, prob_pred, correct, cm, accuracy, precision, recall, bcr = active_learning_platipus(
                        preds, sm_loss_platipus, all_labels, params, x_t, y_t, x_v, y_v)

                    # Display and update individual performance metric
                    cv_stats_dict = update_cv_stats_dict(cv_stats_dict, model, correct, cm, accuracy, precision,
                                                         recall, bcr, prob_pred=prob_pred, verbose=params['verbose'])

            elif model == 'MAML':
                # Load pre-trained MAML model
                maml_checkpoint = load_previous_model_maml(
                    dst_folder_root, params, amine=None)
                Theta_maml = maml_checkpoint['theta']

                # Zero point prediction for MAML model
                print('Getting the MAML model baseline before training on zero points')
                preds = get_naive_task_prediction_maml(
                    all_data, Theta_maml, params)

                # Evaluate zero point performance for MAML
                correct, cm, accuracy, precision, recall, bcr = zero_point_maml(
                    preds, sm_loss_maml, all_labels)

                # Display and update individual performance metric
                cv_stats_dict = update_cv_stats_dict(cv_stats_dict, model, correct, cm, accuracy, precision,
                                                     recall, bcr, verbose=params['verbose'])

                # Reset the training and validation data for MAML
                x_t, x_v = all_data[rand_index].view(1, 51), torch.cat(
                    [all_data[0:rand_index], all_data[rand_index + 1:]])
                y_t, y_v = all_labels[rand_index].view(1), torch.cat(
                    [all_labels[0:rand_index], all_labels[rand_index + 1:]])

                for i in range(iters):
                    print(f'Doing MAML learning with {len(x_t)} examples')
                    num_examples.append(len(x_t))
                    preds = get_task_prediction_maml(x_t=x_t, y_t=y_t, x_v=all_data, meta_params=Theta_maml,
                                                     params=params)

                    # Update available datapoints in the pool and evaluate current model performance
                    x_t, y_t, x_v, y_v, correct, cm, accuracy, precision, recall, bcr = active_learning_maml(
                        preds, sm_loss_maml, all_labels, x_t, y_t, x_v, y_v
                    )

                    # Display and update individual performance metric
                    cv_stats_dict = update_cv_stats_dict(cv_stats_dict, model, correct, cm, accuracy, precision,
                                                         recall, bcr, verbose=params['verbose'])

            else:
                sys.exit('Unidentified model')

            # TODO: Check format
            # Update the main stats dictionary stored in params
            # This is bulky but it's good for future debugging
            params['cv_statistics'][model]['accuracies'].append(
                cv_stats_dict[model]['accuracies'])
            params['cv_statistics'][model]['confusion_matrices'].append(
                cv_stats_dict[model]['confusion_matrices'])
            params['cv_statistics'][model]['precisions'].append(
                cv_stats_dict[model]['precisions'])
            params['cv_statistics'][model]['recalls'].append(
                cv_stats_dict[model]['recalls'])
            params['cv_statistics'][model]['bcrs'].append(
                cv_stats_dict[model]['bcrs'])

        # Plot the metric graphs and save it in the designated folder
        plot_metrics_graph(num_examples, cv_stats_dict,
                           params['active_learning_graph_folder'], amine=amine)

    # Here we are testing, this code should NOT be run yet
    elif not params['cross_validate']:
        testing_batches = params['testing_batches']
        test_batch = testing_batches[amine]
        x_t, y_t, x_v, y_v = torch.from_numpy(test_batch[0]).float().to(params['device']), torch.from_numpy(
            test_batch[1]).long().to(params['device']), \
            torch.from_numpy(test_batch[2]).float().to(params['device']), torch.from_numpy(
            test_batch[3]).long().to(params['device'])

        all_labels = torch.cat((y_t, y_v))
        all_data = torch.cat((x_t, x_v))

        # List out the models we want to conduct active learning
        # For this portion, we are only using PLATIPUS
        models = ['PLATIPUS']

        # Create the stats dictionary to store performance metrics
        cv_stats_dict = create_cv_stats_dict(models)

        # Set up softmax loss function for PLATIPUS
        sm_loss_platipus = params['sm_loss']

        num_examples = []
        iters = len(x_v)

        for i in range(iters):
            print(f'Doing active learning with {len(x_t)} examples')
            num_examples.append(len(x_t))
            preds = get_task_prediction_platipus(x_t, y_t, all_data, params)

            # Update available data points in the pool and evaluate current model performance
            x_t, y_t, x_v, y_v, prob_pred, correct, cm, accuracy, precision, recall, bcr = active_learning_platipus(
                preds, sm_loss_platipus, all_labels, params, x_t, y_t, x_v, y_v)

            # Display and update individual performance metric
            cv_stats_dict = update_cv_stats_dict(cv_stats_dict, 'PLATIPUS', correct, cm, accuracy, precision,
                                                 recall, bcr, prob_pred=prob_pred, verbose=params['verbose'])

        # Plot the metric graphs and save it in the designated folder
        plot_metrics_graph(num_examples, cv_stats_dict,
                           params['active_learning_graph_folder'], amine=amine)


if __name__ == "__main__":
    # np.random.seed(2)
    # main()
    pass
