'''
python maml.py --datasource=sine_line --train --num_inner_updates=5 --k_shot=5 --n_way=1 --inner_lr=1e-3 --meta_lr=1e-3  --num_epochs=10000 --resume_epoch=0 

python maml.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --num_epochs=3000 --cross_validate --resume_epoch=0 --verbose --p_dropout_base=0.4

'''
import torch

import numpy as np
from sklearn.metrics import confusion_matrix

import os
import sys

from models.meta.FC_net import FCNet

import pickle

import argparse


def parse_args():
    """Set up the initial variables for running MAML.

    Retrieves argument values from the terminal command line to create an argparse.Namespace object for initialization.

    Args:
        N/A

    Returns:
        args: Namespace object with the following attributes:
            datasource:         A string identifying the datasource to be used, with default datasource set to drp_chem.
            k_shot:             An integer representing the number of training samples per class, with default set to 5.
            n_way:              An integer representing the number of classes per task, with default set to 1.
            resume_epoch:       An integer representing the epoch id to resume learning or perform testing, with
                                    default set to 0.
            train:              A train_flag attribute. Including it in the command line will set the train_flag to
                                    True by default.
            test:               A train_flag attribute. Including it in the command line will set the train_flag
                                    to False.
            inner_lr:           A float representing the learning rate for task-specific parameters, corresponding to
                                    the 'alpha' learning rate in MAML paper. It is set to 1e-3 by default.
            num_inner_updates:  An integer representing the number of gradient updates for task-specific parameters,
                                    with default set to 5.
            meta_lr:            A float representing the learning rate of meta-parameters, corresponding to the 'beta'
                                    learning rate in MAML paper. It is set to 1e-3 by default.
            meta_batch_size:    An integer representing the number of tasks sampled per outer loop, defaulted to 25.
            num_epochs:         An integer representing the number of outer loops used to train, defaulted to 1000.
            num_epochs_save:    An integer representing the number of outer loops to train before one saving,
                                    defaulted to 1000.
            num_val_tasks:      An integer representing the number of validation tasks, with default set to 100.
            uncertainty:        A uncertainty_flag attribute. Including it in the command line will set the uncertainty_
                                    flag to True. (default)
            no_uncertainty:     A uncertainty_flag attribute. Including it in the command line will set the uncertainty_
                                    flag to False.
            p_dropout_base:     A float representing the dropout rate for the base network, with default set to 0.0
            cross_validate:     A boolean. Including it in the command will run the model with cross-validation.
            verbose:            A boolean. Including it in the command line will print out the memory usage of
                                    the current device.
    """
    parser = argparse.ArgumentParser(description='Setup variables for MAML.')

    parser.add_argument('--datasource', type=str, default='drp_chem',
                        help='datasource to be used, default is drp_chem')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of training samples per class or k-shot')
    parser.add_argument('--n_way', type=int, default=1,
                        help='Number of classes per task, this is 2 for the chemistry data')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Epoch id to resume learning or perform testing')

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--inner_lr', type=float, default=1e-3,
                        help='Learning rate for task-specific parameters')
    parser.add_argument('--num_inner_updates', type=int, default=5,
                        help='Number of gradient updates for task-specific parameters')
    parser.add_argument('--meta_lr', type=float, default=1e-3,
                        help='Learning rate of meta-parameters')
    parser.add_argument('--meta_batch_size', type=int, default=25,
                        help='Number of tasks sampled per outer loop')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='How many outer loops are used to train')
    parser.add_argument('--num_epochs_save', type=int,
                        default=1000, help='How often should we save')

    parser.add_argument('--num_val_tasks', type=int,
                        default=100, help='Number of validation tasks')
    parser.add_argument(
        '--uncertainty', dest='uncertainty_flag', action='store_true')
    parser.add_argument('--no_uncertainty',
                        dest='uncertainty_flag', action='store_false')
    parser.set_defaults(uncertainty_flag=True)

    parser.add_argument('--p_dropout_base', type=float,
                        default=0., help='Dropout rate for the base network')
    parser.add_argument('--cross_validate', action='store_true')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    return args


def initialize():
    """Initializes a dictionary of parameters corresponding to the arguments

    Args:
        N/A

    Returns:
        params: A dictionary of the parameters for the model with the following keys:
            device:                         A torch.device object representing the device on which a torch.Tensor
                                                is/will be allocated.
            gpu_id:                         An integer representing which GPU it will be using.
            train_flag:                     A boolean representing if it will be training the model or not.
            cross_validate:                 A boolean representing if it will use cross-validation or not.
            num_training_samples_per_class: An integer representing the number of training samples per class.
            num_total_samples_per_class:    An integer representing the number of total samples per class.
            num_classes_per_task:           An integer representing the number of classes per task.
            datasource:                     A string representing the data used for MAML.
            inner_lr:                       A float representing the learning rate for task-specific parameters,
                                                corresponding to the 'alpha' learning rate in MAML paper.
            meta_lr:                        A float representing the learning rate of meta-parameters, corresponding to
                                                the 'beta' learning rate in MAML paper.
            num_tasks_per_epoch:            An integer representing the number of tasks used per epoch.
            num_tasks_save_loss:            An integer representing the number of tasks used before one saving of
                                                the values of the losses.
            num_epochs:                     An integer representing the number of outer loops used to train.
            num_epochs_save:                An integer representing the number of outer loops to train before one saving
                                                of the current model parameters.
            num_inner_updates:              An integer representing the number of gradient updates for task-specific
                                                parameters.
            p_dropout_base:                 A float representing the dropout rate for the base network.
            training_batches:               A dictionary representing the training batches used to train PLATIPUS.
                                                Key is amine left out, and value has hierarchy of:
                                                batches -> x_t, y_t, x_v, y_v -> meta_batch_size number of amines
                                                -> k_shot number of reactions -> number of features of each reaction
            validation_batches:             A dictionary representing the validation batches used for cross-validation
                                                in PLATIPUS. Key is amine which the data is for, value has the following
                                                hierarchy: x_s, y_s, x_q, y_q -> k_shot number of reactions
                                                -> number of features of each reaction
            testing_batches:                A dictionary representing the testing batches held out for scientific
                                                research purposes. Key is amine which the data is for, value has the
                                                following hierarchy: x_s, y_s, x_q, y_q -> k_shot number of reactions
                                                -> number of features of each reaction
            counts:                         A dictionary with 'total' and each available amines as keys and lists of
                                                length 2 as values, in the format of:
                                                [# of failed reactions, # of successful reactions]
            net:                            A FCNet object representing the neural network model used for MAML.
            loss_fn:                        A CrossEntropyLoss object representing the loss function for the model.
            sm_loss:                        A Softmax object representing the softmax layer to handle losses later on.
            w_shape:                        A OrderedDict object representing the weight shape and number of weights in
                                                the model, with format:
                                                {weight_name : (number_of_outputs, number_of_inputs)} for weights, and
                                                {weight_name : number_of_outputs} for biases.
            num_val_tasks:                  An integer representing the number of validation tasks.
            dst_folder:                     A string representing the path to save models.
            resume_epoch:                   An integer representing the epoch id to resume learning or perform testing.
            theta:                          A dictionary representing the meta-parameters for MAML, with weights/biases
                                                as keys and their corresponding torch.Tensor object as values.
                                                requires_grad is set to True for all tensors.
            op_theta:                       A torch.optim object representing the optimizer used for meta-parameter
                                                theta. Currently using Adam as suggested in the MAML paper.
            uncertainty_flag:               A boolean representing if the uncertainty flag is turned on or not.
    """

    args = parse_args()
    params = vars(args)
    #params = {}

    # Set up training using either GPU or CPU
    gpu_id = 0
    device = torch.device('cuda:{0:d}'.format(
        gpu_id) if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    print()

    if device.type == 'cuda' and args.verbose:
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(
            torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    params['device'] = device
    params['gpu_id'] = gpu_id

    # Set up MAML using user inputs
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

    # n-way: 1 for the sine-line data and 2 for the chemistry data
    print(f'{args.n_way}-way')
    params['num_classes_per_task'] = args.n_way

    # Initialize the datasource and learning rate
    print(f'Dataset = {args.datasource}')
    params['datasource'] = args.datasource
    print(f'Inner learning rate = {args.inner_lr}')
    params['inner_lr'] = args.inner_lr

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

    if params['datasource'] == 'drp_chem':

        # Use pickle to load the training, validation, hold-out testing batches, and reaction counts used for PLATIPUS
        # Since we're only running MAML to compare with PLATIPUS, thus assume this data already exists
        if params['cross_validate']:
            with open(os.path.join("./data", "train_dump.pkl"), "rb") as f:
                params['training_batches'] = pickle.load(f)
            with open(os.path.join("./data", "val_dump.pkl"), "rb") as f:
                params['validation_batches'] = pickle.load(f)
            with open(os.path.join("./data", "test_dump.pkl"), "rb") as f:
                params['testing_batches'] = pickle.load(f)
            with open(os.path.join("./data", "counts_dump.pkl"), "rb") as f:
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
        weights = [counts['total'][0] / counts['total']
                   [0], counts['total'][0] / counts['total'][1]]
        class_weights = torch.tensor(weights, device=device)
        params['loss_fn'] = torch.nn.CrossEntropyLoss(class_weights)
        # Set up a softmax layer to handle losses later on
        params['sm_loss'] = torch.nn.Softmax(dim=2)
    else:
        sys.exit('Unknown dataset')

    # Weight shape and number of weights in the model
    params['w_shape'] = net.get_weight_shape()
    print(f'Number of parameters of base model = {net.get_num_weights()}')

    # Used in the validation step
    params['num_val_tasks'] = args.num_val_tasks

    # Set up the path to save models
    dst_folder_root = '.'
    dst_folder = '{0:s}/MAML_{1:s}_{2:d}way_{3:d}shot'.format(
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

    # In the case that we are loading a model, resume epoch will not be zero
    params['resume_epoch'] = args.resume_epoch

    if params['resume_epoch'] == 0:
        # Initialise meta-parameters for MAML
        theta = {}
        for key in params['w_shape'].keys():
            if 'b' in key:
                theta[key] = torch.zeros(
                    params['w_shape'][key], device=device, requires_grad=True)
            else:
                theta[key] = torch.empty(params['w_shape'][key], device=device)
                torch.nn.init.xavier_normal_(theta[key], gain=1.)
                theta[key].requires_grad_()
    else:
        # Here we are loading a previously trained model
        checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt').format(
            params['datasource'],
            params['num_classes_per_task'],
            params['num_training_samples_per_class'],
            params['resume_epoch']
        )
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

        theta = saved_checkpoint['theta']

    params['theta'] = theta

    # Now we need to set up the optimizer for theta, this is a lot simpler than it is for PLATIPUS
    op_theta = torch.optim.Adam(params=theta.values(), lr=params['meta_lr'])
    if params['resume_epoch'] > 0:
        op_theta.load_state_dict(saved_checkpoint['op_theta'])
        # Set the learning rate appropriately
        op_theta.param_groups[0]['lr'] = params['meta_lr']

    params['op_theta'] = op_theta
    params['uncertainty_flag'] = args.uncertainty_flag
    print()
    return params


def reinitialize_model_params(params):
    """Reinitialize model parameters for cross validation

    Args:
        params: A dictionary of parameters used for this model.
                See documentation in initialize() for details.

    Returns:
        N/A
    """

    # Initialise meta-parameters for MAML
    theta = {}
    for key in params['w_shape'].keys():
        if 'b' in key:
            theta[key] = torch.zeros(
                params['w_shape'][key], device=params['device'], requires_grad=True)
        else:
            theta[key] = torch.empty(
                params['w_shape'][key], device=params['device'])
            torch.nn.init.xavier_normal_(theta[key], gain=1.)
            theta[key].requires_grad_()

    params['theta'] = theta

    op_theta = torch.optim.Adam(params=theta.values(), lr=params['meta_lr'])
    params['op_theta'] = op_theta


def main():
    """Main driver code

    The main function to conduct meta-training for each available amine.

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
                    dst_folder = '{0:s}/MAML_few_shot/MAML_{1:s}_{2:d}way_{3:d}shot_{4:s}'.format(
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
                        print(
                            'Found existing folder. Meta-parameters will be stored at')
                    print(dst_folder)
                    params['dst_folder'] = dst_folder

                    # Adjust the loss function for each amine
                    amine_counts = params['counts'][amine]
                    weights = [amine_counts[0] / amine_counts[0],
                               amine_counts[0] / amine_counts[1]]
                    print('Using the following weights for loss function:', weights)
                    class_weights = torch.tensor(
                        weights, device=params['device'])
                    params['loss_fn'] = torch.nn.CrossEntropyLoss(
                        class_weights)

                    # Train the model then reinitialize a new one
                    meta_train(params, amine)
                    reinitialize_model_params(params)

    else:
        sys.exit('Unknown action')


def load_previous_model_maml(dst_folder_root, params, amine=None):
    """Load the MAML model previously trained.

    Args:
        dst_folder_root:        A string representing the root directory to get the checkpoint from.
        params:                 A dictionary of parameters used for this model.
                                    See documentation in initialize() for details.
        amine:                  A string representing the amine that our model metrics are for. Default to be None.

    Returns:
        maml_checkpoint:        The checkpoint of the previously trained MAML model.
    """

    # Load MAML data for comparison
    # This requires that we already trained a MAML model, see maml.py
    device = params['device']
    gpu_id = params['gpu_id']
    num_training_samples_per_class = params['num_training_samples_per_class']
    num_classes_per_task = params['num_classes_per_task']

    # Assume root directory is current directory
    num_epochs_save = params['num_epochs_save']

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
    # By default that value is 1000
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
    return maml_checkpoint


def meta_train(params, amine=None):
    """The meta-training function for MAML

    Args:
        params: A dictionary of parameters used for this model. See documentation in initialize() for details.
        amine:  A string representing the specific amine that the model will be trained on.

    Returns:
        N/A
    """

    # Start by unpacking the variables that we need
    datasource = params['datasource']
    num_total_samples_per_class = params['num_total_samples_per_class']
    device = params['device']
    num_classes_per_task = params['num_classes_per_task']
    num_training_samples_per_class = params['num_training_samples_per_class']
    num_tasks_save_loss = params['num_tasks_save_loss']

    # Epoch variables
    num_epochs = params['num_epochs']
    resume_epoch = params['resume_epoch']
    num_tasks_per_epoch = params['num_tasks_per_epoch']

    # Note we have lowercase theta here vs with PLATIPUS
    theta = params['theta']
    op_theta = params['op_theta']

    # How often should we do a printout?
    num_meta_updates_print = 1
    # How often should we save?
    num_epochs_save = params['num_epochs_save']

    for epoch in range(resume_epoch, resume_epoch + num_epochs):
        print(f"Starting epoch {epoch}")

        if datasource == 'drp_chem':
            training_batches = params['training_batches']
            if params['cross_validate']:
                b_num = np.random.choice(len(training_batches[amine]))
                batch = training_batches[amine][b_num]
            else:
                b_num = np.random.choice(len(training_batches))
                batch = training_batches[b_num]  # TODO: this seems wrong
            x_train, y_train, x_val, y_val = torch.from_numpy(batch[0]).float().to(params['device']), torch.from_numpy(
                batch[1]).long().to(params['device']), \
                torch.from_numpy(batch[2]).float().to(params['device']), torch.from_numpy(
                batch[3]).long().to(params['device'])

        # variables used to store information of each epoch for monitoring purpose
        meta_loss_saved = []  # meta loss to save
        val_accuracies = []
        train_accuracies = []

        task_count = 0  # a counter to decide when a minibatch of task is completed to perform meta update
        meta_loss = 0  # accumulate the loss of many ensambling networks to descent gradient for meta update
        num_meta_updates_count = 0

        meta_loss_avg_print = 0  # compute loss average to print

        meta_loss_avg_save = []  # meta loss to save

        while (task_count < num_tasks_per_epoch):
            if datasource == 'drp_chem':
                x_t, y_t, x_v, y_v = x_train[task_count], y_train[task_count], x_val[task_count], y_val[task_count]
            else:
                sys.exit('Unknown dataset')

            loss_NLL = get_task_prediction(x_t, y_t, x_v, params, y_v)

            if torch.isnan(loss_NLL).item():
                sys.exit('NaN error')

            # accumulate meta loss
            meta_loss = meta_loss + loss_NLL

            task_count = task_count + 1

            if task_count % num_tasks_per_epoch == 0:
                meta_loss = meta_loss / num_tasks_per_epoch

                # accumulate into different variables for printing purpose
                meta_loss_avg_print += meta_loss.item()

                op_theta.zero_grad()
                meta_loss.backward()

                # Clip gradients to prevent exploding gradient problem
                torch.nn.utils.clip_grad_norm_(
                    parameters=theta.values(),
                    max_norm=3
                )

                op_theta.step()

                # Printing losses
                num_meta_updates_count += 1
                if (num_meta_updates_count % num_meta_updates_print == 0):
                    meta_loss_avg_save.append(
                        meta_loss_avg_print / num_meta_updates_count)
                    print('{0:d}, {1:2.4f}'.format(
                        task_count,
                        meta_loss_avg_save[-1]
                    ))

                    num_meta_updates_count = 0
                    meta_loss_avg_print = 0

                if (task_count % num_tasks_save_loss == 0):
                    meta_loss_saved.append(np.mean(meta_loss_avg_save))

                    meta_loss_avg_save = []

                # Reset meta loss
                meta_loss = 0

            if (task_count >= num_tasks_per_epoch):
                break

        if ((epoch + 1) % num_epochs_save == 0):
            checkpoint = {
                'theta': theta,
                'meta_loss': meta_loss_saved,
                'val_accuracy': val_accuracies,
                'train_accuracy': train_accuracies,
                'op_theta': op_theta.state_dict()
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


def get_task_prediction(x_t, y_t, x_v, params, y_v=None):
    """Get the predictions on input data

    Args:
        x_t:        A numpy array (3D) representing the training data of one batch.
                        The dimension is meta_batch_size by k_shot by number of features of our data input.
        y_t:        A numpy array (3D) representing the training labels of one batch.
                        The dimension is meta_batch_size by k_shot by n_way.
        x_v:        A numpy array (3D) representing the validation data of one batch.
                        The dimension is meta_batch_size by k_shot by number of features of our data input.
        params:     A dictionary of parameters used for this model. See documentation in initialize() for details.
        y_v:        A numpy array (3D) representing the validation labels of one batch.
                        The dimension is meta_batch_size by k_shot by n_way. It is optional, with default set to None.

    Returns:
        y_pred_v:   A numpy array (3D) representing the predicted labels given our the testing data of one batch.
                        The dimension is meta_batch_size by k_shot by n_way.
        loss_NLL:   A torch.Tensor object representing the loss given our predicted labels and validation labels.
    """

    # Unpack the variables that we need
    theta = params['theta']
    net = params['net']
    p_dropout_base = params['p_dropout_base']
    loss_fn = params['loss_fn']

    num_inner_updates = params['num_inner_updates']
    inner_lr = params['inner_lr']

    # Holds the updated parameters
    q = {}

    # Compute loss on the training data
    y_pred_t = net.forward(x=x_t, w=theta, p_dropout=p_dropout_base)
    loss_NLL = loss_fn(y_pred_t, y_t)

    grads = torch.autograd.grad(
        outputs=loss_NLL,
        inputs=theta.values(),
        create_graph=True
    )
    gradients = dict(zip(theta.keys(), grads))

    # Obtain the weights for the updated model
    for key in theta.keys():
        q[key] = theta[key] - inner_lr * gradients[key]

    # This code gets run if we want to do more than one gradient update
    for _ in range(num_inner_updates - 1):
        loss_NLL = 0
        y_pred_t = net.forward(x=x_t, w=q, p_dropout=p_dropout_base)
        loss_NLL = loss_fn(y_pred_t, y_t)
        grads = torch.autograd.grad(
            outputs=loss_NLL,
            inputs=q.values(),
            retain_graph=True
        )
        gradients = dict(zip(q.keys(), grads))

        for key in q.keys():
            q[key] = q[key] - inner_lr * gradients[key]

    # Now predict on the validation or test data
    y_pred_v = net.forward(x=x_v, w=q, p_dropout=0)

    # Then we were operating on testing data, return our predictions
    if y_v is None:
        return y_pred_v
    # We were operating on validation data, return our loss
    else:
        loss_NLL = loss_fn(y_pred_v, y_v)
        return loss_NLL


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


def zero_point_maml(preds, sm_loss, all_labels):
    """Evalute MAML model performance w/o any active learning.

    Args:
        preds:          A list representing the labels predicted given our all our data points in the pool.
        sm_loss:        A Softmax object representing the softmax layer to handle losses.
        all_labels:     A torch.Tensor object representing all the labels of our reactions.

    Returns:
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
    """

    y_pred = sm_loss(preds)
    print(y_pred)

    _, labels_pred = torch.max(input=y_pred, dim=1)
    print(labels_pred)

    correct = (labels_pred == all_labels)

    accuracy = torch.sum(correct, dim=0).item() / len(all_labels)

    cm = confusion_matrix(all_labels.detach().cpu().numpy(),
                          labels_pred.detach().cpu().numpy())

    # To prevent nan value for precision, we set it to 1 and send out a warning message
    if cm[1][1] + cm[0][1] != 0:
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
    else:
        precision = 1.0
        print('WARNING: zero division during precision calculation')

    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    true_negative = cm[0][0] / (cm[0][0] + cm[0][1])
    bcr = 0.5 * (recall + true_negative)

    return correct, cm, accuracy, precision, recall, bcr


def active_learning_maml(preds, sm_loss, all_labels, x_t, y_t, x_v, y_v):
    """Update active learning pool and evalute MAML model performance.

    Args:
        preds:          A list representing the labels predicted given our all our data points in the pool.
        sm_loss:        A Softmax object representing the softmax layer to handle losses.
        all_labels:     A torch.Tensor object representing all the labels of our reactions.
        x_t:            A numpy array (3D) representing the data of the points used for active learning.
                            The dimension is meta_batch_size by k_shot by number of features of our data input.
        y_t:            A numpy array (3D) representing the labels of the points used for active learning.
                            The dimension is meta_batch_size by k_shot by n_way.
        x_v:            A numpy array (3D) representing the data of the available points in the active learning pool.
                            The dimension is meta_batch_size by k_shot by number of features of our data input.
        y_v:            A numpy array (3D) representing the labels of the available points in the active learning pool.
                            The dimension is meta_batch_size by k_shot by n_way.

    Returns:
        x_t:            A numpy array (3D) representing the training data of one batch.
                        The dimension is meta_batch_size by k_shot by number of features of our data input.
        y_t:            A numpy array (3D) representing the training labels of one batch.
                        The dimension is meta_batch_size by k_shot by n_way.
        x_v:            A numpy array (3D) representing the validation data of one batch.
                        The dimension is meta_batch_size by k_shot by number of features of our data input.
        y_v:            A numpy array (3D) representing the labels of the available points in the active learning pool.
                            The dimension is meta_batch_size by k_shot by n_way.
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
        bcr:            A float representing the balanced classification rate of the model. It's the average of
                            the recall rate and the true negative rate.
    """

    y_pred = sm_loss(preds)

    _, labels_pred = torch.max(input=y_pred, dim=1)
    correct = (labels_pred == all_labels)
    accuracy = torch.sum(correct, dim=0).item() / len(all_labels)

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

    cm = confusion_matrix(all_labels.detach().cpu().numpy(),
                          labels_pred.detach().cpu().numpy())

    # To prevent nan value for precision, we set it to 1 and send out a warning message
    if cm[1][1] + cm[0][1] != 0:
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
    else:
        precision = 1.0
        print('WARNING: zero division during precision calculation')

    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    true_negative = cm[0][0] / (cm[0][0] + cm[0][1])
    bcr = 0.5 * (recall + true_negative)

    return x_t, y_t, x_v, y_v, correct, cm, accuracy, precision, recall, bcr


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


if __name__ == "__main__":
    main()
