'''
NOTE: Before running the following commands, make sure to run command lines in the order of training PLATIPUS, training MAML, and cross-validating both models.

Small scale testing:
python main.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --Lt=1 --num_inner_updates=10 --Lv=10 --kl_reweight=.0001 --num_epochs=4 --num_epochs_save=2 --cross_validate --resume_epoch=0 --verbose --p_dropout_base=0.4

python maml.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --num_epochs=4 --num_epochs_save=2 --cross_validate --resume_epoch=0 --verbose --p_dropout_base=0.4

python main.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --Lt=1 --Lv=100 --num_inner_updates=10 --kl_reweight=.0001 --num_epochs=0 --num_epochs_save=2 --resume_epoch=4 --cross_validate --verbose --p_dropout_base=0.4 --test

run me to train cross validation:
python main.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --Lt=1 --num_inner_updates=10 --Lv=10 --kl_reweight=.0001 --num_epochs=3000 --num_epochs_save=1000 --cross_validate --resume_epoch=0 --verbose --p_dropout_base=0.4

run me for cross validation results:
python main.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --Lt=1 --Lv=100 --num_inner_updates=10 --kl_reweight=.0001 --num_epochs=0 --num_epochs_save=1000 --resume_epoch=3000 --cross_validate --verbose --p_dropout_base=0.4 --test


DON'T TOUCH ME YET
run me for test results:
python main.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=5 --Lt=10 --Lv=100 --num_inner_updates=10 --kl_reweight=.0001 --num_epochs=0 --resume_epoch=10000 --verbose --p_dropout_base=0.4 --test

run me for training a model on all the training data (not CV):
python main.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --Lt=1 --Lv=10 --kl_reweight=.0001 --num_epochs=3000 --resume_epoch=0 --verbose --p_dropout_base=0.3 --train
'''

import torch
import numpy as np
import random
import itertools
import pickle
import copy

import os
import sys

import argparse
from utils import load_chem_dataset
from utils import load_chem_dataset_testing
from FC_net import FCNet

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def parse_args():
    """Set up the initial variables for running PLATIPUS.

    Retrieves argument values from the terminal command line to create an argparse.Namespace object for
        initialization.

    Args:
        N/A

    Returns:
        args: Namespace object with the following attributes:
            datasource:         A string identifying the datasource to be used, with default datasource set to drp_chem.
            k_shot:             An integer representing the number of training samples per class, with default set to 1.
            n_way:              An integer representing the number of classes per task, with default set to 1.
            resume_epoch:       An integer representing the epoch to resume learning or perform testing, 0 by default.
            train:              A train_flag attribute. Including it in the command line will set the train_flag to
                                    True by default.
            test:               A train_flag attribute. Including it in the command line will set the train_flag to
                                    False.
            inner_lr:           A float representing the learning rate for task-specific parameters, corresponding to
                                    the 'alpha' learning rate in meta-training section of PLATIPUS.
                                    It is set to 1e-3 by default.
            pred_lr:            A float representing the learning rate for task-specific parameters during prediction,
                                    corresponding to the 'alpha' learning rate in meta-testing section of PLATIPUS.
                                    It is set to 1e-1 by default.
            num_inner_updates:  An integer representing the number of gradient updates for task-specific parameters,
                                    with default set to 5.
            meta_lr:            A float representing the learning rate of meta-parameters, corresponding to the 'beta'
                                    learning rate in meta-training section of PLATIPUS.
                                    It is set to 1e-3 by default.
            meta_batch_size:    An integer representing the number of tasks sampled per outer loop.
                                    It is set to 25 by default.
            num_epochs:         An integer representing the number of outer loops used to train, defaulted to 50000.
            num_epochs_save:    An integer representing the number of outer loops to train before one saving,
                                    with default set to 1000.
            TODO: kl_reweight, Lt, Lv
            p_dropout_base:     A float representing the dropout rate for the base network, with default set to 0.0
            cross_validate:     A boolean. Including it in the command will run the model with cross-validation.
            TODO: verbose: A boolean. Including it in the command line will ...
    """

    parser = argparse.ArgumentParser(description='Setup variables for PLATIPUS.')
    parser.add_argument('--datasource', type=str, default='drp_chem',
                        help='datasource to be used, defaults to drp_chem')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of training samples per class')
    parser.add_argument('--n_way', type=int, default=1,
                        help='Number of classes per task, this is 2 for the chemistry data')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--inner_lr', type=float, default=1e-3, help='Learning rate for task-specific parameters')
    parser.add_argument('--pred_lr', type=float, default=1e-1,
                        help='Learning rate for task-specific parameters during prediction (rather than training)')
    parser.add_argument('--num_inner_updates', type=int, default=5,
                        help='Number of gradient updates for task-specific parameters')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta-parameters')
    parser.add_argument('--meta_batch_size', type=int, default=25, help='Number of tasks sampled per outer loop')
    parser.add_argument('--num_epochs', type=int, default=50000, help='How many outer loops are used to train')
    parser.add_argument('--num_epochs_save', type=int, default=1000, help='How often should we save')

    parser.add_argument('--kl_reweight', type=float, default=1,
                        help='Reweight factor of the KL divergence between variational posterior q and prior p')

    parser.add_argument('--Lt', type=int, default=1,
                        help='Number of ensemble networks to train task specific parameters')
    parser.add_argument('--Lv', type=int, default=1, help='Number of ensemble networks to validate meta-parameters')

    parser.add_argument('--p_dropout_base', type=float, default=0., help='Dropout rate for the base network')
    parser.add_argument('--cross_validate', action='store_true')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    return args


def initialize():
    """Initializes a dictionary of parameters corresponding to the arguments

    The purpose of this function is trying to use a parameters dictionary without blowing up the number of
        parameters in each function

    Args:
        N/A

    Returns:
        params: A dictionary of the parameters for the model with the following keys:
            device:                             A torch.device object representing the device on which a torch.Tensor
                                                    is/will be allocated.
            gpu_id:                             An integer representing which GPU it will be using.
            train_flag:                         A boolean representing if it will be training the model or not.
            cross_validate:                     A boolean representing if it will use cross-validation or not.
            num_training_samples_per_class:     An integer representing the number of training samples per class.
            num_total_samples_per_class:        An integer representing the number of total samples per class.
            num_classes_per_task:               An integer representing the number of classes per task.
            datasource:                         A string representing the data used for PLATIPUS.
            inner_lr:                           A float representing the learning rate for task-specific parameters,
                                                    corresponding to the 'alpha' learning rate in meta-training section
                                                    of PLATIPUS.
            pred_lr:                            A float representing the learning rate for task-specific parameters
                                                    during prediction, corresponding to the 'alpha' learning rate in
                                                    meta-testing section of PLATIPUS.
            meta_lr:                            A float representing the learning rate of meta-parameters, corresponding
                                                    to the 'beta' learning rate in meta-training section of PLATIPUS.
            num_tasks_per_epoch:                An integer representing the number of tasks used per epoch.
            num_tasks_save_loss:                An integer representing the number of tasks used before one saving of
                                                    the values of the losses.
            num_epochs:                         An integer representing the number of outer loops used to train.
            num_epochs_save:                    An integer representing the number of outer loops to train before
                                                    one saving of the model parameters.
            num_inner_updates:                  An integer representing the number of gradient updates for task-specific
                                                    parameters.
            p_dropout_base:                     A float representing the dropout rate for the base network.
            TODO: L, K
            training_batches:                   A dictionary representing the training batches used to train PLATIPUS.
                                                    Key is amine left out, and value has hierarchy of:
                                                    batches -> x_t, y_t, x_v, y_v -> meta_batch_size number of amines
                                                    -> k_shot number of reactions -> number of features of each reaction
            validation_batches:                 A dictionary representing the validation batches used for
                                                    cross-validation in PLATIPUS. Key is amine which the data is for,
                                                    value has the following hierarchy: x_s, y_s, x_q, y_q
                                                    -> k_shot number of reactions -> number of features of each reaction
            testing_batches:                    A dictionary representing the testing batches held out for scientific
                                                    research purposes. Key is amine which the data is for, value has
                                                    the following hierarchy: x_s, y_s, x_q, y_q
                                                    -> k_shot number of reactions -> number of features of each reaction
            counts:                             A dictionary with 'total' and each available amines as keys and lists of
                                                    length 2 as values in the format of:
                                                    [# of failed reactions, # of successful reactions]
            net:                                A FCNet object representing the neural network model used for PLATIPUS.
            loss_fn:                            A CrossEntropyLoss object representing the loss function for the model.
            sm_loss:                            A Softmax object representing the softmax layer to handle losses.
            w_shape:                            A OrderedDict object representing the weight shape and number of weights
                                                    in the model, with format
                                                    {weight_name : (number_of_outputs, number_of_inputs)} for weights,
                                                    {weight_name : number_of_outputs} for biases.
            TODO: num_weights, KL_reweight
            dst_folder:                         A string representing the path to save models.
            TODO: graph_folder, active_learning_graph_folder
            resume_epoch:                       An integer representing the epoch id to resume learning or perform
                                                    testing.
            TODO: Theta:                        A dictionary representing the meta-parameters for PLATIPUS, with
                                                    weights/biases as keys and their corresponding torch.Tensor object
                                                    as values (requires_grad is set to True for all tensors).
            TODO: op_Theta:                     A torch.optim object representing the optimizer used for meta-parameter
                                                    theta. Currently using Adam as suggested in the PLATIPUS paper.
        """

    args = parse_args()
    params = {}

    # Set up training using either GPU or CPU
    gpu_id = 0
    device = torch.device('cuda:{0:d}'.format(gpu_id) if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    print()

    if device.type == 'cuda' and args.verbose:
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    params['device'] = device
    params['gpu_id'] = gpu_id

    # Set up PLATIPUS using user inputs
    params['train_flag'] = args.train_flag
    params['cross_validate'] = args.cross_validate
    # Set up the value of k in k-shot learning
    print(f'{args.k_shot}-shot')
    params['num_training_samples_per_class'] = args.k_shot

    # Total number of samples per class, need some extra for the outer loop update as well
    if params['train_flag']:
        params['num_total_samples_per_class'] = params['num_training_samples_per_class'] + 15
    else:
        params['num_total_samples_per_class'] = params['num_training_samples_per_class'] + 20

    # n-way: 2 for the chemistry data
    print(f"{args.n_way}-way")
    params['num_classes_per_task'] = args.n_way

    # Initialize the datasource and learning rate
    print(f'Dataset = {args.datasource}')
    params['datasource'] = args.datasource
    print(f'Inner learning rate = {args.inner_lr}')
    params['inner_lr'] = args.inner_lr
    params['pred_lr'] = args.pred_lr

    # Set up the meta learning rate
    print(f'Meta learning rate = {args.meta_lr}')
    params['meta_lr'] = args.meta_lr

    # Reducing this to 25 like in Finn et al.
    # Tells us how many tasks are per epoch and after how many tasks we should save the values of the losses
    params['num_tasks_per_epoch'] = args.meta_batch_size
    params['num_tasks_save_loss'] = args.meta_batch_size
    params['num_epochs'] = args.num_epochs

    # How often should we save?
    params['num_epochs_save'] = args.num_epochs_save

    # How many gradient updates we run on the inner loop
    print(f'Number of inner updates = {args.num_inner_updates}')
    params['num_inner_updates'] = args.num_inner_updates

    # Dropout rate for the neural network
    params['p_dropout_base'] = args.p_dropout_base

    # L as how many models we sample in the inner update and K as how many models we sample in validation
    print(f'L = {args.Lt}, K = {args.Lv}')
    params['L'] = args.Lt
    params['K'] = args.Lv

    if params['datasource'] == 'drp_chem':

        # Set number of training samples and number of total samples per class
        # These two values are hard-coded, corresponding to the values hard-coded in load_chem_dataset below
        params['num_total_samples_per_class'] = 40
        params['num_training_samples_per_class'] = 20

        if params['train_flag']:

            if params['cross_validate']:

                training_batches, validation_batches, testing_batches, counts = load_chem_dataset(k_shot=20,
                                                                                                  meta_batch_size=args.meta_batch_size,
                                                                                                  num_batches=250,
                                                                                                  verbose=args.verbose)
                params['training_batches'] = training_batches
                params['validation_batches'] = validation_batches
                params['testing_batches'] = testing_batches
                params['counts'] = counts

                # Save for reproducibility
                with open(os.path.join("./data", "train_dump.pkl"), "wb") as f:
                    pickle.dump(training_batches, f)
                with open(os.path.join("./data", "val_dump.pkl"), "wb") as f:
                    pickle.dump(validation_batches, f)
                with open(os.path.join("./data", "test_dump.pkl"), "wb") as f:
                    pickle.dump(testing_batches, f)
                with open(os.path.join("./data", "counts_dump.pkl"), "wb") as f:
                    pickle.dump(counts, f)

            else:
                training_batches, testing_batches, counts = load_chem_dataset_testing(k_shot=20,
                                                                                      meta_batch_size=args.meta_batch_size,
                                                                                      num_batches=250,
                                                                                      verbose=args.verbose)
                params['training_batches'] = training_batches
                params['testing_batches'] = testing_batches
                params['counts'] = counts

                # Save for reproducibility
                with open(os.path.join("./data", "train_dump_nocv.pkl"), "wb") as f:
                    pickle.dump({'training_data': training_batches}, f)
                with open(os.path.join("./data", "test_dump_nocv.pkl"), "wb") as f:
                    pickle.dump(testing_batches, f)
                with open(os.path.join("./data", "counts_dump_nocv.pkl"), "wb") as f:
                    pickle.dump(counts, f)

        # Make sure we don't overwrite our batches if we are validating and testing
        else:
            if params['cross_validate']:
                with open(os.path.join("./data", "train_dump.pkl"), "rb") as f:
                    params['training_batches'] = pickle.load(f)
                with open(os.path.join("./data", "val_dump.pkl"), "rb") as f:
                    params['validation_batches'] = pickle.load(f)
                with open(os.path.join("./data", "test_dump.pkl"), "rb") as f:
                    params['testing_batches'] = pickle.load(f)
                with open(os.path.join("./data", "counts_dump.pkl"), "rb") as f:
                    params['counts'] = pickle.load(f)

            else:
                with open(os.path.join("./data", "train_dump_nocv.pkl"), "rb") as f:
                    params['training_batches'] = pickle.load(f)['training_data']
                with open(os.path.join("./data", "test_dump_nocv.pkl"), "rb") as f:
                    params['testing_batches'] = pickle.load(f)
                with open(os.path.join("./data", "counts_dump_nocv.pkl"), "rb") as f:
                    params['counts'] = pickle.load(f)

        net = FCNet(
            dim_input=51,
            dim_output=params['num_classes_per_task'],
            num_hidden_units=(200, 100, 100),
            device=device
        )
        params['net'] = net
        # Try weighting positive reactions more heavily
        # Majority class label count/ class label count for each weight
        # See https://discuss.pytorch.org/t/what-is-the-weight-values-mean-in-torch-nn-crossentropyloss/11455/9
        # Do this again just in case we are loading weights
        counts = params['counts']
        weights = [counts['total'][0] / counts['total'][0], counts['total'][0] / counts['total'][1]]
        class_weights = torch.tensor(weights, device=device)
        params['loss_fn'] = torch.nn.CrossEntropyLoss(class_weights)
        # Set up a softmax layer to handle losses later on 
        params['sm_loss'] = torch.nn.Softmax(dim=2)
    else:
        sys.exit('Unknown dataset')

    params['w_shape'] = net.get_weight_shape()
    # Inefficient to call this twice, but I was getting an annoying linting error
    print(f'Number of parameters of base model = {net.get_num_weights()}')
    params['num_weights'] = net.get_num_weights()

    # Weight on the KL loss
    print(f'KL reweight = {args.kl_reweight}')
    params['KL_reweight'] = args.kl_reweight

    # Set up the path to save models
    dst_folder_root = '.'
    dst_folder = '{0:s}/PLATIPUS_few_shot/PLATIPUS_{1:s}_{2:d}way_{3:d}shot'.format(
        dst_folder_root,
        params['datasource'],
        params['num_classes_per_task'],
        params['num_training_samples_per_class']
    )
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print('No folder for storage found')
        print(f'Make folder to store meta-parameters at')
    else:
        print('Found existing folder. Meta-parameters will be stored at')
    print(dst_folder)
    params['dst_folder'] = dst_folder

    # Set up the path to save graphs
    graph_folder = '{0:s}/graphs'.format(params['dst_folder'])
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
        print('No folder for graph storage found')
        print(f'Make folder to store graphs at')
    else:
        print('Found existing folder. Graphs will be stored at')
    print(graph_folder)
    params['graph_folder'] = graph_folder

    # Set up sub-folder under graphs to save active learning cross validation graphs
    active_learning_graph_folder = '{0:s}/active_learning_cv_graphs'.format(params['graph_folder'])
    if not os.path.exists(active_learning_graph_folder):
        os.makedirs(active_learning_graph_folder)
        print('No folder for active learning cross-validation graph storage found')
        print(f'Make folder to store cross-validation graphs at')
    else:
        print('Found existing folder. Active learning cross-validation graphs will be stored at')
    print(active_learning_graph_folder)
    params['active_learning_graph_folder'] = active_learning_graph_folder

    # In the case that we are loading a model, resume epoch will not be zero 
    params['resume_epoch'] = args.resume_epoch
    if params['resume_epoch'] == 0:
        # Initialize meta-parameters
        # Theta is capital theta in the PLATIPUS paper, it holds everything we need
        # 'mean' is the mean model parameters 
        # 'logSigma' and 'logSigma_q' are the variance of the base and variational distributions
        # the two 'gamma' vectors are the learning rate vectors
        Theta = {}
        Theta['mean'] = {}
        Theta['logSigma'] = {}
        Theta['logSigma_q'] = {}
        Theta['gamma_q'] = {}
        Theta['gamma_p'] = {}
        for key in params['w_shape'].keys():
            if 'b' in key:
                Theta['mean'][key] = torch.zeros(params['w_shape'][key], device=device, requires_grad=True)
            else:
                Theta['mean'][key] = torch.empty(params['w_shape'][key], device=device)
                # Could also opt for Kaiming Normal here
                torch.nn.init.xavier_normal_(tensor=Theta['mean'][key], gain=1.)
                Theta['mean'][key].requires_grad_()

            # Subtract 4 to get us into appropriate range for log variances
            Theta['logSigma'][key] = torch.rand(params['w_shape'][key], device=device) - 4
            Theta['logSigma'][key].requires_grad_()

            Theta['logSigma_q'][key] = torch.rand(params['w_shape'][key], device=device) - 4
            Theta['logSigma_q'][key].requires_grad_()

            Theta['gamma_q'][key] = torch.tensor(1e-2, device=device, requires_grad=True)
            Theta['gamma_q'][key].requires_grad_()
            Theta['gamma_p'][key] = torch.tensor(1e-2, device=device, requires_grad=True)
            Theta['gamma_p'][key].requires_grad_()
    else:
        # Cross validation will load a bunch of models elsewhere when testing
        # A little confusing, but if we are training we want to initialize the first model
        if not params['cross_validate'] or params['train_flag']:
            # Here we are loading a previously trained model
            print('Restore previous Theta...')
            print('Resume epoch {0:d}'.format(params['resume_epoch']))
            checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt') \
                .format(params['datasource'],
                        params['num_classes_per_task'],
                        params['num_training_samples_per_class'],
                        params['resume_epoch'])
            checkpoint_file = os.path.join(dst_folder, checkpoint_filename)
            print('Start to load weights from')
            print('{0:s}'.format(checkpoint_file))
            if torch.cuda.is_available():
                saved_checkpoint = torch.load(
                    checkpoint_file,
                    map_location=lambda storage,
                                        loc: storage.cuda(gpu_id)
                )
            else:
                saved_checkpoint = torch.load(
                    checkpoint_file,
                    map_location=lambda storage,
                                        loc: storage
                )

            Theta = saved_checkpoint['Theta']

    if not params['cross_validate'] or params['train_flag']:
        params['Theta'] = Theta

    if not params['cross_validate'] or params['train_flag']:
        # Now we need to set up the optimizer for Theta, PyTorch makes this very easy for us, phew.
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
            lr=params['meta_lr']
        )

        if params['resume_epoch'] > 0:
            op_Theta.load_state_dict(saved_checkpoint['op_Theta'])
            # Set the meta learning rates appropriately
            op_Theta.param_groups[0]['lr'] = params['meta_lr']
            op_Theta.param_groups[1]['lr'] = params['meta_lr']

        params['op_Theta'] = op_Theta

    print()
    return params


def reinitialize_model_params(params):
    """Reinitialize model meta-parameters for cross validation in PLATIPUS

    Args:
        params: A dictionary of parameters used for this model. See documentation in initialize() for details.

    Returns:
        N/A
    """
    Theta = {}
    Theta['mean'] = {}
    Theta['logSigma'] = {}
    Theta['logSigma_q'] = {}
    Theta['gamma_q'] = {}
    Theta['gamma_p'] = {}
    for key in params['w_shape'].keys():
        if 'b' in key:
            Theta['mean'][key] = torch.zeros(params['w_shape'][key], device=params['device'], requires_grad=True)
        else:
            Theta['mean'][key] = torch.empty(params['w_shape'][key], device=params['device'])
            # Could also opt for Kaiming Normal here
            torch.nn.init.xavier_normal_(tensor=Theta['mean'][key], gain=1.)
            Theta['mean'][key].requires_grad_()

        # Subtract 4 to get us into appropriate range for log variances
        Theta['logSigma'][key] = torch.rand(params['w_shape'][key], device=params['device']) - 4
        Theta['logSigma'][key].requires_grad_()

        Theta['logSigma_q'][key] = torch.rand(params['w_shape'][key], device=params['device']) - 4
        Theta['logSigma_q'][key].requires_grad_()

        Theta['gamma_q'][key] = torch.tensor(1e-2, device=params['device'], requires_grad=True)
        Theta['gamma_q'][key].requires_grad_()
        Theta['gamma_p'][key] = torch.tensor(1e-2, device=params['device'], requires_grad=True)
        Theta['gamma_p'][key].requires_grad_()

    params['Theta'] = Theta

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
        lr=params['meta_lr']
    )
    params['op_Theta'] = op_Theta


def main():
    """Main driver code

    The main function to conduct PLATIPUS meta-training for each available amine.

    Args:
        N/A

    Returns:
        N/A
    """

    # Do the massive initialization and get back a dictionary instead of using global variables
    params = initialize()

    if params['train_flag']:
        if params['datasource'] == 'drp_chem':
            if params['cross_validate']:
                # TODO: This is going to be insanely nasty, basically reinitialize for each amine
                for amine in params['training_batches']:
                    print("Starting training for amine", amine)
                    # Change the path to save models
                    dst_folder_root = '.'
                    dst_folder = '{0:s}/PLATIPUS_few_shot/PLATIPUS_{1:s}_{2:d}way_{3:d}shot_{4:s}'.format(
                        dst_folder_root,
                        params['datasource'],
                        params['num_classes_per_task'],
                        params['num_training_samples_per_class'],
                        amine
                    )
                    if not os.path.exists(dst_folder):
                        os.makedirs(dst_folder)
                        print('No folder for storage found')
                        print(f'Make folder to store meta-parameters at')
                    else:
                        print('Found existing folder. Meta-parameters will be stored at')
                    print(dst_folder)
                    params['dst_folder'] = dst_folder

                    # Adjust the loss function for each amine
                    amine_counts = params['counts'][amine]
                    if amine_counts[0] >= amine_counts[1]:
                        weights = [amine_counts[0] / amine_counts[0], amine_counts[0] / amine_counts[1]]
                    else:
                        weights = [amine_counts[1] / amine_counts[0], amine_counts[1] / amine_counts[1]]

                    print('Using the following weights for loss function:', weights)
                    class_weights = torch.tensor(weights, device=params['device'])
                    params['loss_fn'] = torch.nn.CrossEntropyLoss(class_weights)

                    # Train the model then reinitialize a new one
                    meta_train(params, amine)
                    reinitialize_model_params(params)
            else:
                meta_train(params)

    elif params['resume_epoch'] > 0:
        if params['datasource'] == 'drp_chem' and params['cross_validate']:
            # I am saving this dictionary in case things go wrong
            # It will get added to in the active learning code
            stats_dict = {}
            stats_dict['accuracies'] = []
            stats_dict['confusion_matrices'] = []
            stats_dict['precisions'] = []
            stats_dict['recalls'] = []
            stats_dict['balanced_classification_rates'] = []
            stats_dict['accuracies_MAML'] = []
            stats_dict['confusion_matrices_MAML'] = []
            stats_dict['precisions_MAML'] = []
            stats_dict['recalls_MAML'] = []
            stats_dict['balanced_classification_rates_MAML'] = []
            params['cv_statistics'] = stats_dict
            # Test performance of each individual cross validation model 
            for amine in params['validation_batches']:
                print("Starting validation for amine", amine)
                # Change the path to save models
                dst_folder_root = '.'
                dst_folder = '{0:s}/PLATIPUS_few_shot/PLATIPUS_{1:s}_{2:d}way_{3:d}shot_{4:s}'.format(
                    dst_folder_root,
                    params['datasource'],
                    params['num_classes_per_task'],
                    params['num_training_samples_per_class'],
                    amine
                )
                params['dst_folder'] = dst_folder

                # Here we are loading a previously trained model
                print('Restore previous Theta...')
                print('Resume epoch {0:d}'.format(params['resume_epoch']))
                checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt') \
                    .format(params['datasource'],
                            params['num_classes_per_task'],
                            params['num_training_samples_per_class'],
                            params['resume_epoch'])
                checkpoint_file = os.path.join(dst_folder, checkpoint_filename)
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

                Theta = saved_checkpoint['Theta']
                params['Theta'] = Theta

                # Adjust the loss function for each amine
                amine_counts = params['counts'][amine]
                if amine_counts[0] >= amine_counts[1]:
                    weights = [amine_counts[0] / amine_counts[0], amine_counts[0] / amine_counts[1]]
                else:
                    weights = [amine_counts[1] / amine_counts[0], amine_counts[1] / amine_counts[1]]
                print('Using the following weights for loss function:', weights)
                class_weights = torch.tensor(weights, device=params['device'])
                params['loss_fn'] = torch.nn.CrossEntropyLoss(class_weights)

                # Run forward pass on the validation data
                validation_batches = params['validation_batches']
                val_batch = validation_batches[amine]
                x_t, y_t, x_v, y_v = torch.from_numpy(val_batch[0]).float().to(params['device']), torch.from_numpy(
                    val_batch[1]).long().to(params['device']), \
                                     torch.from_numpy(val_batch[2]).float().to(params['device']), torch.from_numpy(
                    val_batch[3]).long().to(params['device'])

                accuracies = []
                corrects = []
                probability_pred = []
                sm_loss = params['sm_loss']
                preds = get_task_prediction(x_t, y_t, x_v, params)
                # print('raw task predictions', preds)
                y_pred_v = sm_loss(torch.stack(preds))
                # print('after applying softmax', y_pred_v)
                y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)
                # print('after calling torch mean', y_pred)

                prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
                # print('print training labels', y_t)
                print('print labels predicted', labels_pred)
                print('print true labels', y_v)
                # print('print probability of prediction', prob_pred)
                correct = (labels_pred == y_v)
                corrects.extend(correct.detach().cpu().numpy())

                # print('length of validation set', len(y_v))
                accuracy = torch.sum(correct, dim=0).item() / len(y_v)
                accuracies.append(accuracy)

                probability_pred.extend(prob_pred.detach().cpu().numpy())

                print('accuracy for model is', accuracies)
                print('probabilities for predictions are', probability_pred)
                test_model_actively(params, amine)

            # Save this dictionary in case we need it later
            with open(os.path.join("./data", "cv_statistics.pkl"), "wb") as f:
                pickle.dump(params['cv_statistics'], f)

            with open(os.path.join("./data", "cv_statistics.pkl"), "rb") as f:
                params['cv_statistics'] = pickle.load(f)
            # Now plot the big graph
            stats_dict = params['cv_statistics']

            # Do some deletion so the average graph has no nan's
            for key in stats_dict.keys():
                del stats_dict[key][10]
                del stats_dict[key][11]

            for key in ['precisions', 'precisions_MAML']:
                for stat_list in stats_dict[key]:
                    print(stat_list[0])
                    if np.isnan(stat_list[0]):
                        stat_list[0] = 0

            # for key in stats_dict.keys():
            #     for stat_list in stats_dict[key]:
            #         del stat_list[0]        

            min_length = len(min(stats_dict['accuracies'], key=len))
            print(f'Minimum number of points we have to work with is {min_length}')
            average_accs = []
            average_precisions = []
            average_recalls = []
            average_bcrs = []

            MAML_average_accs = []
            MAML_average_precisions = []
            MAML_average_recalls = []
            MAML_average_bcrs = []

            num_examples = []

            print('Minimum length is', min_length)
            for i in range(min_length):
                if i == 0:
                    num_examples.append(0)
                else:
                    num_examples.append(20 + i)
                # Go by amine, this code is bulky but I like having it here for debugging
                total = 0
                for acc_list in stats_dict['accuracies']:
                    total += acc_list[i]
                average_accs.append(total / len(stats_dict['accuracies']))

                total = 0
                for prec_list in stats_dict['precisions']:
                    total += prec_list[i]
                average_precisions.append(total / len(stats_dict['precisions']))

                total = 0
                for rec_list in stats_dict['recalls']:
                    total += rec_list[i]
                average_recalls.append(total / len(stats_dict['recalls']))

                total = 0
                for bcr_list in stats_dict['balanced_classification_rates']:
                    total += bcr_list[i]
                average_bcrs.append(total / len(stats_dict['balanced_classification_rates']))

                total = 0
                for acc_list_maml in stats_dict['accuracies_MAML']:
                    total += acc_list_maml[i]
                MAML_average_accs.append(total / len(stats_dict['accuracies_MAML']))

                total = 0
                for prec_list_maml in stats_dict['precisions_MAML']:
                    total += prec_list_maml[i]
                MAML_average_precisions.append(total / len(stats_dict['precisions_MAML']))

                total = 0
                for rec_list_maml in stats_dict['recalls_MAML']:
                    total += rec_list_maml[i]
                MAML_average_recalls.append(total / len(stats_dict['recalls_MAML']))

                total = 0
                for bcr_list_maml in stats_dict['balanced_classification_rates_MAML']:
                    total += bcr_list_maml[i]
                MAML_average_bcrs.append(total / len(stats_dict['balanced_classification_rates_MAML']))

            fig = plt.figure(figsize=(16, 12))
            plt.subplot(2, 2, 1)
            plt.ylabel('Accuracy')
            plt.title(f'Averaged learning curve')
            plt.plot(num_examples, average_accs, 'ro-', label='PLATIPUS')
            plt.plot(num_examples, MAML_average_accs, 'bo-', label='MAML')
            plt.legend()

            print(average_precisions)
            print(MAML_average_precisions)
            plt.subplot(2, 2, 2)
            plt.ylabel('Precision')
            plt.title(f'Averaged precision curve')
            plt.plot(num_examples, average_precisions, 'ro-', label='PLATIPUS')
            plt.plot(num_examples, MAML_average_precisions, 'bo-', label='MAML')
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.ylabel('Recall')
            plt.title(f'Averaged recall curve')
            plt.plot(num_examples, average_recalls, 'ro-', label='PLATIPUS')
            plt.plot(num_examples, MAML_average_recalls, 'bo-', label='MAML')
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.ylabel('Balanced classification rate')
            plt.title(f'Averaged BCR curve')
            plt.plot(num_examples, average_bcrs, 'ro-', label='PLATIPUS')
            plt.plot(num_examples, MAML_average_bcrs, 'bo-', label='MAML')
            plt.legend()

            fig.text(0.5, 0.04, "Number of samples given", ha="center", va="center")

            # Save the average metrics graph to ./graphs folder
            graph_dst = '{0:s}/average_metrics.png'.format(params['graph_folder'])
            # Remove duplicate graphs in case we can't directly overwrite the files
            if os.path.isfile(graph_dst):
                os.remove(graph_dst)
            plt.savefig(graph_dst)
            # plt.show()

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
                preds = get_task_prediction(x_t, y_t, x_v, params)
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


def meta_train(params, amine=None):
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
                    meta_loss_avg_save.append(meta_loss_avg_print / num_meta_updates_count)
                    kl_loss_avg_save.append(kl_loss_avg_print / num_meta_updates_count)
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
            torch.save(checkpoint, os.path.join(dst_folder, checkpoint_filename))
        print()


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
        maml_checkpoint_filename = os.path.join(maml_folder, maml_filename.format(i))
        while (os.path.exists(maml_checkpoint_filename)):
            i = i + num_epochs_save
            maml_checkpoint_filename = os.path.join(maml_folder, maml_filename.format(i))

        # This is overshooting, just have it here as a sanity check
        print('loading from', maml_checkpoint_filename)

        if torch.cuda.is_available():
            maml_checkpoint = torch.load(
                os.path.join(maml_folder, maml_filename.format(i - num_epochs_save)),
                map_location=lambda storage,
                                    loc: storage.cuda(gpu_id)
            )
        else:
            maml_checkpoint = torch.load(
                os.path.join(maml_folder, maml_filename.format(i - num_epochs_save)),
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

        num_examples = []

        # Set up the number of active learning iterations
        # Starting from 1 so that we can compare PLATIPUS/MAML with other models such as SVM and KNN that
        #  have valid results from the first validation point.
        # For testing, overwrite with iters = 1
        iters = len(x_t) + len(x_v) - 1

        # CODE FOR ZERO POINT
        print('Getting the model baseline before training on zero points')
        preds = get_naive_prediction(all_data, params)
        y_pred_v = sm_loss(torch.stack(preds))
        y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)

        prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
        correct = (labels_pred == all_labels)
        corrects.extend(correct.detach().cpu().numpy())
        accuracy = torch.sum(correct, dim=0).item() / len(all_labels)
        accuracies.append(accuracy)

        probability_pred.extend(prob_pred.detach().cpu().numpy())

        cm = confusion_matrix(all_labels.detach().cpu().numpy(), labels_pred.detach().cpu().numpy())
        print(cm)
        confusion_matrices.append(cm)
        print('accuracy for model is', accuracy)

        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        recall = cm[1][1] / (cm[1][1] + cm[1][0])
        true_negative = cm[0][0] / (cm[0][0] + cm[0][1])
        bcr = 0.5 * (recall + true_negative)
        print('precision for model is', precision)
        print('recall for model is', recall)
        print('balanced classification rate for model is', bcr)

        precisions.append(precision)
        recalls.append(recall)
        balanced_classification_rates.append(bcr)
        num_examples.append(0)

        # Randomly pick a point to start active learning with
        rand_index = np.random.choice(iters+1)

        # Reset the training set and validation set
        x_t, x_v = all_data[rand_index].view(-1, 51), torch.cat([all_data[0:rand_index], all_data[rand_index + 1:]])
        y_t, y_v = all_labels[rand_index].view(1), torch.cat([all_labels[0:rand_index], all_labels[rand_index + 1:]])

        for _ in range(iters):
            print(f'Doing active learning with {len(x_t)} examples')
            num_examples.append(len(x_t))
            preds = get_task_prediction(x_t, y_t, all_data, params)
            y_pred_v = sm_loss(torch.stack(preds))
            y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)

            prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
            correct = (labels_pred == all_labels)
            corrects.extend(correct.detach().cpu().numpy())
            accuracy = torch.sum(correct, dim=0).item() / len(all_labels)
            accuracies.append(accuracy)

            probability_pred.extend(prob_pred.detach().cpu().numpy())

            # print(all_labels)
            # print(labels_pred)  

            # Now add the most uncertain point to the training data 
            preds_update = get_task_prediction(x_t, y_t, x_v, params)

            y_pred_v_update = sm_loss(torch.stack(preds_update))
            y_pred_update = torch.mean(input=y_pred_v_update, dim=0, keepdim=False)

            prob_pred_update, labels_pred_update = torch.max(input=y_pred_update, dim=1)

            print(y_v)
            print(labels_pred_update)
            print(len(prob_pred_update))

            value, index = prob_pred_update.min(0)
            print(f'Minimum confidence {value}')
            # Add to the training data 
            x_t = torch.cat((x_t, x_v[index].view(1, 51)))
            y_t = torch.cat((y_t, y_v[index].view(1)))
            # Remove from pool, there is probably a less clunky way to do this
            x_v = torch.cat([x_v[0:index], x_v[index + 1:]])
            y_v = torch.cat([y_v[0:index], y_v[index + 1:]])
            print('length of x_v is now', len(x_v))

            cm = confusion_matrix(all_labels.detach().cpu().numpy(), labels_pred.detach().cpu().numpy())
            print(cm)
            confusion_matrices.append(cm)
            print('accuracy for model is', accuracy)

            precision = cm[1][1] / (cm[1][1] + cm[0][1])
            recall = cm[1][1] / (cm[1][1] + cm[1][0])
            true_negative = cm[0][0] / (cm[0][0] + cm[0][1])
            bcr = 0.5 * (recall + true_negative)
            print('precision for model is', precision)
            print('recall for model is', recall)
            print('balanced classification rate for model is', bcr)

            precisions.append(precision)
            recalls.append(recall)
            balanced_classification_rates.append(bcr)

        # Now do it again but for the MAML model
        accuracies_MAML = []
        corrects_MAML = []
        confusion_matrices_MAML = []
        precisions_MAML = []
        recalls_MAML = []
        balanced_classification_rates_MAML = []
        num_examples = []
        corrects_MAML = []

        sm_loss_maml = torch.nn.Softmax(dim=1)

        print('Getting the MAML model baseline before training on zero points')
        preds = get_naive_task_prediction_maml(all_data, Theta_maml, params)
        y_pred = sm_loss_maml(preds)
        print(y_pred)

        _, labels_pred = torch.max(input=y_pred, dim=1)
        print(labels_pred)
        correct = (labels_pred == all_labels)
        corrects_MAML.extend(correct.detach().cpu().numpy())
        accuracy = torch.sum(correct, dim=0).item() / len(all_labels)
        accuracies_MAML.append(accuracy)

        cm = confusion_matrix(all_labels.detach().cpu().numpy(), labels_pred.detach().cpu().numpy())
        print(cm)
        confusion_matrices_MAML.append(cm)
        print('accuracy for model is', accuracy)

        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        recall = cm[1][1] / (cm[1][1] + cm[1][0])
        true_negative = cm[0][0] / (cm[0][0] + cm[0][1])
        bcr = 0.5 * (recall + true_negative)
        print('precision for model is', precision)
        print('recall for model is', recall)
        print('balanced classification rate for model is', bcr)

        precisions_MAML.append(precision)
        recalls_MAML.append(recall)
        balanced_classification_rates_MAML.append(bcr)
        num_examples.append(0)

        # Reset the training and validation data for MAML
        x_t, x_v = all_data[rand_index].view(1, 51), torch.cat([all_data[0:rand_index], all_data[rand_index + 1:]])
        y_t, y_v = all_labels[rand_index].view(1), torch.cat([all_labels[0:rand_index], all_labels[rand_index + 1:]])

        for i in range(iters):
            print(f'Doing MAML learning with {len(x_t)} examples')
            num_examples.append(len(x_t))
            preds = get_task_prediction_maml(x_t=x_t, y_t=y_t, x_v=all_data, meta_params=Theta_maml, params=params)
            y_pred = sm_loss_maml(preds)

            _, labels_pred = torch.max(input=y_pred, dim=1)
            correct = (labels_pred == all_labels)
            corrects_MAML.extend(correct.detach().cpu().numpy())
            accuracy = torch.sum(correct, dim=0).item() / len(all_labels)
            accuracies_MAML.append(accuracy)

            # print(all_labels)
            # print(labels_pred)  

            # Now add a random point since MAML cannot reason about uncertainty
            index = np.random.choice(len(x_v))
            # Add to the training data 
            x_t = torch.cat((x_t, x_v[index].view(1, 51)))
            y_t = torch.cat((y_t, y_v[index].view(1)))
            # Remove from pool, there is probably a less clunky way to do this
            x_v = torch.cat([x_v[0:index], x_v[index + 1:]])
            y_v = torch.cat([y_v[0:index], y_v[index + 1:]])
            print('length of x_v is now', len(x_v))

            cm = confusion_matrix(all_labels.detach().cpu().numpy(), labels_pred.detach().cpu().numpy())
            print(cm)
            confusion_matrices_MAML.append(cm)
            print('accuracy for model is', accuracy)

            precision = cm[1][1] / (cm[1][1] + cm[0][1])
            recall = cm[1][1] / (cm[1][1] + cm[1][0])
            true_negative = cm[0][0] / (cm[0][0] + cm[0][1])
            bcr = 0.5 * (recall + true_negative)
            print('precision for model is', precision)
            print('recall for model is', recall)
            print('balanced classification rate for model is', bcr)

            precisions_MAML.append(precision)
            recalls_MAML.append(recall)
            balanced_classification_rates_MAML.append(bcr)

        fig = plt.figure(figsize=(16, 12))
        plt.subplot(2, 2, 1)
        plt.ylabel('Accuracy')
        plt.title(f'Learning curve for {amine}')
        plt.plot(num_examples, accuracies, 'ro-', label='PLATIPUS')
        plt.plot(num_examples, accuracies_MAML, 'bo-', label='MAML')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.ylabel('Precision')
        plt.title(f'Precision curve for {amine}')
        plt.plot(num_examples, precisions, 'ro-', label='PLATIPUS')
        plt.plot(num_examples, precisions_MAML, 'bo-', label='MAML')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.ylabel('Recall')
        plt.title(f'Recall curve for {amine}')
        plt.plot(num_examples, recalls, 'ro-', label='PLATIPUS')
        plt.plot(num_examples, recalls_MAML, 'bo-', label='MAML')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.ylabel('Balanced classification rate')
        plt.title(f'BCR curve for {amine}')
        plt.plot(num_examples, balanced_classification_rates, 'ro-', label='PLATIPUS')
        plt.plot(num_examples, balanced_classification_rates_MAML, 'bo-', label='MAML')
        plt.legend()

        fig.text(0.5, 0.04, "Number of samples given", ha="center", va="center")

        # Save the active training graph to ./active_learning_cv_graphs folder
        active_graph_dst = '{0:s}/cv_metrics_{1:s}.png'.format(params['active_learning_graph_folder'], amine)
        # Remove duplicate graphs in case we can't directly overwrite the files
        if os.path.isfile(active_graph_dst):
            os.remove(active_graph_dst)
        plt.savefig(active_graph_dst)
        # plt.show()

        params['cv_statistics']['accuracies'].append(accuracies)
        params['cv_statistics']['confusion_matrices'].append(confusion_matrices)
        params['cv_statistics']['precisions'].append(precisions)
        params['cv_statistics']['recalls'].append(recalls)
        params['cv_statistics']['balanced_classification_rates'].append(balanced_classification_rates)
        params['cv_statistics']['accuracies_MAML'].append(accuracies_MAML)
        params['cv_statistics']['confusion_matrices_MAML'].append(confusion_matrices_MAML)
        params['cv_statistics']['precisions_MAML'].append(precisions_MAML)
        params['cv_statistics']['recalls_MAML'].append(recalls_MAML)
        params['cv_statistics']['balanced_classification_rates_MAML'].append(balanced_classification_rates_MAML)

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
            preds = get_task_prediction(x_t, y_t, all_data, params)
            y_pred_v = sm_loss(torch.stack(preds))
            y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)

            prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
            correct = (labels_pred == all_labels)
            corrects.extend(correct.detach().cpu().numpy())
            accuracy = torch.sum(correct, dim=0).item() / len(all_labels)
            accuracies.append(accuracy)

            probability_pred.extend(prob_pred.detach().cpu().numpy())

            # print(all_labels)
            # print(labels_pred)  

            # Now add the most uncertain point to the training data 
            preds_update = get_task_prediction(x_t, y_t, x_v, params)

            y_pred_v_update = sm_loss(torch.stack(preds_update))
            y_pred_update = torch.mean(input=y_pred_v_update, dim=0, keepdim=False)

            prob_pred_update, labels_pred_update = torch.max(input=y_pred_update, dim=1)

            print(y_v)
            print(labels_pred_update)
            print(len(prob_pred_update))

            value, index = prob_pred_update.min(0)
            print(f'Minimum confidence {value}')
            # Add to the training data 
            x_t = torch.cat((x_t, x_v[index].view(1, 51)))
            y_t = torch.cat((y_t, y_v[index].view(1)))
            # Remove from pool, there is probably a less clunky way to do this
            x_v = torch.cat([x_v[0:index], x_v[index + 1:]])
            y_v = torch.cat([y_v[0:index], y_v[index + 1:]])
            print('length of x_v is now', len(x_v))

            cm = confusion_matrix(all_labels.detach().cpu().numpy(), labels_pred.detach().cpu().numpy())
            print(cm)
            confusion_matrices.append(cm)
            print('accuracy for model is', accuracy)
            precision = cm[1][1] / (cm[1][1] + cm[0][1])
            recall = cm[1][1] / (cm[1][1] + cm[1][0])
            true_negative = cm[0][0] / (cm[0][0] + cm[0][1])
            bcr = 0.5 * (recall + true_negative)
            print('precision for model is', precision)
            print('recall for model is', recall)
            print('balanced classification rate for model is', bcr)

            precisions.append(precision)
            recalls.append(recall)
            balanced_classification_rates.append(bcr)

        fig = plt.figure(figsize=(16,12))
        plt.subplot(2, 2, 1)
        plt.ylabel('Accuracy')
        plt.title(f'Learning curve for {amine}')
        plt.plot(num_examples, accuracies, 'bo-', label='PLATIPUS')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.ylabel('Precision')
        plt.title(f'Precision curve for {amine}')
        plt.plot(num_examples, precisions, 'ro-', label='PLATIPUS')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.ylabel('Recall')
        plt.title(f'Recall curve for {amine}')
        plt.plot(num_examples, recalls, 'go-', label='PLATIPUS')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.ylabel('Balanced classification rate')
        plt.title(f'BCR curve for {amine}')
        plt.plot(num_examples, balanced_classification_rates, 'mo-', label='PLATIPUS')
        plt.legend()

        fig.text(0.5, 0.04, "Number of samples given", ha="center", va="center")


def get_training_loss(x_t, y_t, x_v, y_v, params):
    """Get the training loss

    Determines the KL and Meta Objective loss on a set of training and validation data
    Steps correspond to steps in the PLATIPUS TRAINING algorithm in Finn et al

    Args:
        x_t:        A numpy array (3D) representing the training data of one batch.
                        The dimension is meta_batch_size by k_shot by number of features of our data input.
        y_t:        A numpy array (3D) representing the training labels of one batch.
                        The dimension is meta_batch_size by k_shot by n_way.
        x_v:        A numpy array (3D) representing the validation data of one batch.
                        The dimension is meta_batch_size by k_shot by number of features of our data input.
        y_v:        A numpy array (3D) representing the validation labels of one batch.
                        The dimension is meta_batch_size by k_shot by n_way.
        params:     A dictionary of parameters used for this model. See documentation in initialize() for details.


    return:
        loss_query: An integer representing query level loss for the training dataset
        KL_q_p:     An integer representing the KL divergence between the training result
                    and ideal result distribution
    """

    # Start by unpacking the parameters we need
    Theta = params['Theta']
    net = params['net']
    p_dropout_base = params['p_dropout_base']
    loss_fn = params['loss_fn']
    L = params['L']
    num_inner_updates = params['num_inner_updates']
    inner_lr = params['inner_lr']
    num_weights = params['num_weights']

    # Initialize the variational distribution
    q = initialise_dict_of_dict(Theta['mean'].keys())

    # List of sampled models 
    phi = []

    # step 6 - Compute loss on query set
    y_pred_query = net.forward(x=x_v, w=Theta['mean'], p_dropout=p_dropout_base)
    loss_query = loss_fn(y_pred_query, y_v)
    loss_query_grads = torch.autograd.grad(
        outputs=loss_query,
        inputs=Theta['mean'].values(),
        create_graph=True
    )
    loss_query_gradients = dict(zip(Theta['mean'].keys(), loss_query_grads))

    # step 7 - Update parameters of the variational distribution
    for key in Theta['mean'].keys():
        q['mean'][key] = Theta['mean'][key] - Theta['gamma_q'][key] * loss_query_gradients[key]
        q['logSigma'][key] = Theta['logSigma_q'][key]

    # step 8 - Update L sampled models on the training data x_t using gradient descient
    for _ in range(L):
        # Generate a set of weights using the meta_parameters, equivalent to sampling a model 
        w = generate_weights(meta_params=Theta, params=params)
        y_pred_t = net.forward(x=x_t, w=w, p_dropout=p_dropout_base)
        loss_vfe = loss_fn(y_pred_t, y_t)

        loss_vfe_grads = torch.autograd.grad(
            outputs=loss_vfe,
            inputs=w.values(),
            create_graph=True
        )
        loss_vfe_gradients = dict(zip(w.keys(), loss_vfe_grads))

        # step 9 - Compute updated parameters phi_i using the gradients
        phi_i = {}
        for key in w.keys():
            phi_i[key] = w[key] - inner_lr * loss_vfe_gradients[key]
        phi.append(phi_i)

    # Repeat step 9 so we do num_inner_updates number of gradient descent steps
    for _ in range(num_inner_updates - 1):
        for phi_i in phi:
            y_pred_t = net.forward(x=x_t, w=phi_i, p_dropout=p_dropout_base)
            loss_vfe = loss_fn(y_pred_t, y_t)

            loss_vfe_grads = torch.autograd.grad(
                outputs=loss_vfe,
                inputs=phi_i.values(),
                retain_graph=True
            )
            loss_vfe_gradients = dict(zip(phi_i.keys(), loss_vfe_grads))
            for key in phi_i.keys():
                phi_i[key] = phi_i[key] - inner_lr * loss_vfe_gradients[key]

    # step 10 - Set up probability distribution given the training data
    p = initialise_dict_of_dict(key_list=Theta['mean'][key])
    y_pred_train = net.forward(x=x_t, w=Theta['mean'], p_dropout=p_dropout_base)
    loss_train = loss_fn(y_pred_train, y_t)
    loss_train_grads = torch.autograd.grad(
        outputs=loss_train,
        inputs=Theta['mean'].values(),
        create_graph=True
    )
    loss_train_gradients = dict(zip(Theta['mean'].keys(), loss_train_grads))
    for key in Theta['mean'].keys():
        p['mean'][key] = Theta['mean'][key] - Theta['gamma_p'][key] * loss_train_gradients[key]
        p['logSigma'][key] = Theta['logSigma'][key]

    # step 11 - Compute Meta Objective and KL loss
    loss_query = 0
    # Note: We can have multiple models here by adjusting the --Lt flag, but it will 
    # drastically slow down the training (linear scaling)
    for phi_i in phi:
        y_pred_query = net.forward(x=x_v, w=phi_i)
        loss_query += loss_fn(y_pred_query, y_v)
    loss_query /= L
    KL_q_p = 0
    for key in q['mean'].keys():
        # I am so glad somebody has a formula for this... You rock Cuong
        KL_q_p += torch.sum(torch.exp(2 * (q['logSigma'][key] - p['logSigma'][key])) \
                            + (p['mean'][key] - q['mean'][key]) ** 2 / torch.exp(2 * p['logSigma'][key])) \
                  + torch.sum(2 * (p['logSigma'][key] - q['logSigma'][key]))
    KL_q_p = (KL_q_p - num_weights) / 2
    return loss_query, KL_q_p


def get_task_prediction(x_t, y_t, x_v, params):
    """Get the task prediction for PLATIPUS model

    Update the PLATIPUS model on some training data then obtain its output
    on some testing data
    Steps correspond to steps in the PLATIPUS TESTING algorithm in Finn et al

    Args:
        x_t:        A numpy array (3D) representing the training data of one batch.
                        The dimension is meta_batch_size by k_shot by number of features of our data input.
        y_t:        A numpy array (3D) representing the training labels of one batch.
                        The dimension is meta_batch_size by k_shot by n_way.
        x_v:        A numpy array (3D) representing the validation data of one batch.
                        The dimension is meta_batch_size by k_shot by number of features of our data input.
        params:     A dictionary of parameters used for this model. See documentation in initialize() for details.

    Returns:
        y_pred_v: A numpy array (3D) representing the predicted labels
        given our the validation or testing data of one batch.
    """

    # As usual, begin by unpacking the parameters we need
    Theta = params['Theta']
    net = params['net']
    p_dropout_base = params['p_dropout_base']
    # Do not weight loss function during testing? 
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = params['loss_fn']
    K = params['K']
    num_inner_updates = params['num_inner_updates']
    pred_lr = params['pred_lr']

    # step 1 - Set up the prior distribution over weights (given the training data)
    p = initialise_dict_of_dict(key_list=Theta['mean'].keys())
    y_pred_train = net.forward(x=x_t, w=Theta['mean'], p_dropout=p_dropout_base)
    loss_train = loss_fn(y_pred_train, y_t)
    loss_train_grads = torch.autograd.grad(
        outputs=loss_train,
        inputs=Theta['mean'].values(),
        create_graph=True
    )
    loss_train_gradients = dict(zip(Theta['mean'].keys(), loss_train_grads))
    for key in Theta['mean'].keys():
        p['mean'][key] = Theta['mean'][key] - Theta['gamma_p'][key] * loss_train_gradients[key]
        p['logSigma'][key] = Theta['logSigma'][key]

    # step 2 - Sample K models and update determine the gradient of the loss function
    phi = []
    for _ in range(K):
        w = generate_weights(meta_params=p, params=params)
        y_pred_t = net.forward(x=x_t, w=w)
        loss_train = loss_fn(y_pred_t, y_t)
        loss_train_grads = torch.autograd.grad(
            outputs=loss_train,
            inputs=w.values()
        )
        loss_train_gradients = dict(zip(w.keys(), loss_train_grads))

        # step 3 - Compute adapted parameters using gradient descent
        phi_i = {}
        for key in w.keys():
            phi_i[key] = w[key] - pred_lr * loss_train_gradients[key]
        phi.append(phi_i)

    # Repeat step 3 as many times as specified
    for _ in range(num_inner_updates - 1):
        for phi_i in phi:
            y_pred_t = net.forward(x=x_t, w=phi_i)
            loss_train = loss_fn(y_pred_t, y_t)
            loss_train_grads = torch.autograd.grad(
                outputs=loss_train,
                inputs=phi_i.values()
            )
            loss_train_gradients = dict(zip(w.keys(), loss_train_grads))

            for key in w.keys():
                phi_i[key] = phi_i[key] - pred_lr * loss_train_gradients[key]

    y_pred_v = []
    # Now get the model predictions on the validation/test data x_v by calling the forward method
    for phi_i in phi:
        w = generate_weights(meta_params=p, params=params)
        y_pred_t = net.forward(x=x_t, w=w)
        y_pred_temp = net.forward(x=x_v, w=phi_i)
        y_pred_v.append(y_pred_temp)
    return y_pred_v


def get_naive_prediction(x_vals, params):
    """Get the naive task prediction for PLATIPUS model

    Get naive predictions from the model without doing any update steps
    This is used to get the zero point for the model for drp_chem

    Args:
        x_vals: A numpy array representing the data we want to find the prediction for
        params: A dictionary of parameters used for this model. See documentation in initialize() for details.

    return: A numpy array (3D) representing the predicted labels
    """
    # As usual, begin by unpacking the parameters we need
    Theta = params['Theta']
    net = params['net']
    K = params['K']

    y_pred_v = []
    # Now get the model predictions on the validation/test data x_v by calling the forward method
    for _ in range(K):
        # Generate some random weights
        w = generate_weights(meta_params=Theta, params=params)
        y_pred_temp = net.forward(x=x_vals, w=w)
        y_pred_v.append(y_pred_temp)
    return y_pred_v


def get_naive_task_prediction_maml(x_vals, meta_params, params):
    """Get the naive task prediction for MAML model

    Super simple function to get MAML prediction with no updates

    Args:
        x_vals:     A numpy array representing the data we want to find the prediction for
        meta_params:A dictionary of dictionaries used for the meta learning model.
        params:     A dictionary of parameters used for this model. See documentation in initialize() for details.

    return: A numpy array (3D) representing the predicted labels
    """
    net = params['net']
    y_pred_v = net.forward(x=x_vals, w=meta_params)
    return y_pred_v


def get_task_prediction_maml(x_t, y_t, x_v, meta_params, params):
    """Get the prediction of label with a MAML model

    Run a forward pass to obtain predictions given a MAML model
    It is required that a MAML model has already been trained
    Also make sure the MAML model has the same architecture as the PLATIPUS model
    or elese the call to net.forward() will give you problems

    Args:
        x_t:        A numpy array (3D) representing the training data of one batch.
                        The dimension is meta_batch_size by k_shot by number of features of our data input.
        y_t:        A numpy array (3D) representing the training labels of one batch.
                        The dimension is meta_batch_size by k_shot by n_way.
        x_v:        A numpy array (3D) representing the validation data of one batch.
                        The dimension is meta_batch_size by k_shot by number of features of our data input.
        meta_params:A dictionary of dictionaries used for the meta learning model.
        params:     A dictionary of parameters used for this model. See documentation in initialize() for details.

    Returns:
        y_pred_v: A numpy array (3D) representing the predicted labels
        given our the validation or testing data of one batch.
    """

    # Get the program parameters from params
    net = params['net']
    loss_fn = params['loss_fn']
    inner_lr = params['inner_lr']
    num_inner_updates = params['num_inner_updates']
    p_dropout_base = params['p_dropout_base']

    # The MAML model weights
    q = {}

    y_pred_t = net.forward(x=x_t, w=meta_params)
    loss_vfe = loss_fn(y_pred_t, y_t)

    grads = torch.autograd.grad(
        outputs=loss_vfe,
        inputs=meta_params.values(),
        create_graph=True
    )
    gradients = dict(zip(meta_params.keys(), grads))

    for key in meta_params.keys():
        q[key] = meta_params[key] - inner_lr * gradients[key]

    # Similar to PLATIPUS, we can perform more gradient updates if we wish to
    for _ in range(num_inner_updates - 1):
        loss_vfe = 0
        y_pred_t = net.forward(x=x_t, w=q, p_dropout=p_dropout_base)
        loss_vfe = loss_fn(y_pred_t, y_t)
        grads = torch.autograd.grad(
            outputs=loss_vfe,
            inputs=q.values(),
            retain_graph=True
        )
        gradients = dict(zip(q.keys(), grads))

        for key in q.keys():
            q[key] = q[key] - inner_lr * gradients[key]

    # Finally call the forward function
    y_pred_v = net.forward(x=x_v, w=q)
    return y_pred_v


def generate_weights(meta_params, params):
    """Generate a set of weights

    Use the PLATIPUS means and variances to actually generate a set of weights
    i.e. a plausible model

    Args:
        meta_params:A dictionary within dictionary.
                    The parameters for meta learning
        params:     The program parameters

    return: A dictionary with indices generated using the epsilon reparameterization trick
    """

    device = params['device']
    w = {}
    for key in meta_params['mean'].keys():
        eps_sampled = torch.randn(meta_params['mean'][key].shape, device=device)
        # Use the epsilon reparameterization trick in VI
        w[key] = meta_params['mean'][key] + eps_sampled * torch.exp(meta_params['logSigma'][key])
    return w


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


if __name__ == "__main__":
    main()
