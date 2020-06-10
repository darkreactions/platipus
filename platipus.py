import torch
import numpy as np
import random
import itertools
import pickle
import copy

import os
import sys

from utils import *
from FC_net import FCNet

from sklearn.metrics import confusion_matrix


# Originally "reinitialzie model params" Core line 147 in main function
def initialzie_theta_platipus(params):
    """This function is to initialize Theta

    Args:
        params: A dictionary of initialized parameters

    return: A dictionary of initialized meta parameters used for cross validation
    """
    Theta = {}
    Theta['mean'] = {}
    Theta['logSigma'] = {}
    Theta['logSigma_q'] = {}
    Theta['gamma_q'] = {}
    Theta['gamma_p'] = {}
    for key in params['w_shape'].keys():
        if 'b' in key:
            Theta['mean'][key] = torch.zeros(
                params['w_shape'][key], device=params['device'], requires_grad=True)
        else:
            Theta['mean'][key] = torch.empty(
                params['w_shape'][key], device=params['device'])
            # Could also opt for Kaiming Normal here
            torch.nn.init.xavier_normal_(tensor=Theta['mean'][key], gain=1.)
            Theta['mean'][key].requires_grad_()

        # Subtract 4 to get us into appropriate range for log variances
        Theta['logSigma'][key] = torch.rand(
            params['w_shape'][key], device=params['device']) - 4
        Theta['logSigma'][key].requires_grad_()

        Theta['logSigma_q'][key] = torch.rand(
            params['w_shape'][key], device=params['device']) - 4
        Theta['logSigma_q'][key].requires_grad_()

        Theta['gamma_q'][key] = torch.tensor(
            1e-2, device=params['device'], requires_grad=True)
        Theta['gamma_q'][key].requires_grad_()
        Theta['gamma_p'][key] = torch.tensor(
            1e-2, device=params['device'], requires_grad=True)
        Theta['gamma_p'][key].requires_grad_()
    return Theta

def set_optim_platipus(Theta,meta_lr):
    """This function is to set up the optimizer for Theta

    Args:
        Theta:      A dictionary containing the meta parameters
        meta_lr:    The defined learning rate for meta_learning

    return: The optimizer setted up for Theta
    """
    op_Theta = torch.optim.Adam(
        [
            {
                'params': Theta['mean'].values()
            },
            {
                'params': Theta['logSigma'].values()
            },
            {
                'params': Theta['logSigma_q'].values()
            },
            {
                'params': Theta['gamma_p'].values()
            },
            {
                'params': Theta['gamma_q'].values()
            }
        ],
        lr=meta_lr
    )
    return op_Theta


def load_previous_model_platipus(params):
    """This function is to load in previous model

    This is for PLATIPUS model specifically.

    Args:
        params: A dictionary of initialzed parameters

    return: The saved checkpoint
    """
    print('Restore previous Theta...')
    print('Resume epoch {0:d}'.format(params['resume_epoch']))
    checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt') \
        .format(params['datasource'],
                params['num_classes_per_task'],
                params['num_training_samples_per_class'],
                params['resume_epoch'])
    checkpoint_file = os.path.join(params["dst_folder"], checkpoint_filename)
    print('Start to load weights from')
    print('{0:s}'.format(checkpoint_file))
    if torch.cuda.is_available():
        saved_checkpoint = torch.load(
            checkpoint_file,
            map_location=lambda storage,
                                loc: storage.cuda(params['gpu_id'])
        )
    else:
        saved_checkpoint = torch.load(
            checkpoint_file,
            map_location=lambda storage,
                                loc: storage
        )
    return saved_checkpoint


def meta_train_platipus(params, amine=None):
    """The meta-training section of PLATIPUS

    Lots of cool stuff happening in this function.

    Args:
        params:     A dictionary of parameters used for this model. See documentation in initialize() for details.
        amine:      A string representing the specific amine that the model will be trained on.

    Returns:
        N/A
    """

    # Start by unpacking the params we need
    # Messy but it does eliminate global variables and make things easier to trace
    datasource = params['datasource']
    num_total_samples_per_class = params['num_total_samples_per_class']
    device = params['device']
    num_classes_per_task = params['num_classes_per_task']
    num_training_samples_per_class = params['num_training_samples_per_class']
    num_tasks_save_loss = params['num_tasks_save_loss']

    num_epochs = params['num_epochs']
    resume_epoch = params['resume_epoch']
    num_tasks_per_epoch = params['num_tasks_per_epoch']

    Theta = params['Theta']
    op_Theta = params['op_Theta']
    KL_reweight = params['KL_reweight']

    num_epochs_save = params['num_epochs_save']
    # How often should we do a printout?
    num_meta_updates_print = 1

    for epoch in range(resume_epoch, resume_epoch + num_epochs):
        print(f"Starting epoch {epoch}")

        # Load chemistry data
        if datasource == 'drp_chem':
            training_batches = params['training_batches']
            if params['cross_validate']:
                b_num = np.random.choice(len(training_batches[amine]))
                batch = training_batches[amine][b_num]
            else:
                b_num = np.random.choice(len(training_batches))
                batch = training_batches[b_num]
            x_train, y_train, x_val, y_val = torch.from_numpy(batch[0]).float().to(params['device']), torch.from_numpy(
                batch[1]).long().to(params['device']), \
                torch.from_numpy(batch[2]).float().to(params['device']), torch.from_numpy(
                batch[3]).long().to(params['device'])
        # variables used to store information of each epoch for monitoring purpose
        meta_loss_saved = []  # meta loss to save
        kl_loss_saved = []
        val_accuracies = []
        train_accuracies = []

        meta_loss = 0  # accumulate the loss of many ensambling networks to descent gradient for meta update
        num_meta_updates_count = 0

        meta_loss_avg_print = 0  # compute loss average to print

        kl_loss = 0
        kl_loss_avg_print = 0

        meta_loss_avg_save = []  # meta loss to save
        kl_loss_avg_save = []

        task_count = 0  # a counter to decide when a minibatch of task is completed to perform meta update

        # Treat batches as tasks for the chemistry data
        while task_count < num_tasks_per_epoch:
            if datasource == 'drp_chem':
                x_t, y_t, x_v, y_v = x_train[task_count], y_train[task_count], x_val[task_count], y_val[task_count]
            else:
                sys.exit('Unknown dataset')

            loss_i, KL_q_p = get_training_loss(x_t, y_t, x_v, y_v, params)
            KL_q_p = KL_q_p * KL_reweight

            if torch.isnan(loss_i).item():
                sys.exit('NaN error')

            # accumulate meta loss
            meta_loss = meta_loss + loss_i + KL_q_p
            kl_loss = kl_loss + KL_q_p

            task_count = task_count + 1

            if task_count % num_tasks_per_epoch == 0:
                meta_loss = meta_loss / num_tasks_per_epoch
                kl_loss /= num_tasks_per_epoch

                # accumulate into different variables for printing purpose
                meta_loss_avg_print += meta_loss.item()
                kl_loss_avg_print += kl_loss.item()

                op_Theta.zero_grad()
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=Theta['mean'].values(),
                    max_norm=3
                )
                torch.nn.utils.clip_grad_norm_(
                    parameters=Theta['logSigma'].values(),
                    max_norm=3
                )
                op_Theta.step()

                # Printing losses
                num_meta_updates_count += 1
                if (num_meta_updates_count % num_meta_updates_print == 0):
                    meta_loss_avg_save.append(
                        meta_loss_avg_print / num_meta_updates_count)
                    kl_loss_avg_save.append(
                        kl_loss_avg_print / num_meta_updates_count)
                    print('{0:d}, {1:2.4f}, {2:2.4f}'.format(
                        task_count,
                        meta_loss_avg_save[-1],
                        kl_loss_avg_save[-1]
                    ))

                    num_meta_updates_count = 0
                    meta_loss_avg_print = 0
                    kl_loss_avg_print = 0

                if (task_count % num_tasks_save_loss == 0):
                    meta_loss_saved.append(np.mean(meta_loss_avg_save))
                    kl_loss_saved.append(np.mean(kl_loss_avg_save))

                    meta_loss_avg_save = []
                    kl_loss_avg_save = []

                # reset meta loss
                meta_loss = 0
                kl_loss = 0

        if ((epoch + 1) % num_epochs_save == 0):
            checkpoint = {
                'Theta': Theta,
                'kl_loss': kl_loss_saved,
                'meta_loss': meta_loss_saved,
                'val_accuracy': val_accuracies,
                'train_accuracy': train_accuracies,
                'op_Theta': op_Theta.state_dict()
            }
            print('SAVING WEIGHTS...')

            checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt') \
                .format(datasource,
                        num_classes_per_task,
                        num_training_samples_per_class,
                        epoch + 1)
            print(checkpoint_filename)
            dst_folder = params['dst_folder']
            torch.save(checkpoint, os.path.join(
                dst_folder, checkpoint_filename))
        print()