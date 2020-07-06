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

import argparse
import torch
import os
import sys

from .core import main
from .FC_net import FCNet
from utils.utils import *
from .platipus import *
#from models.meta.maml import *


def parse_args(args):
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
            kl_reweight:        An integer representing the re-weight factor of the KL divergence between
                                    variational posterior q and prior p. The default weight is set to 1.
            Lt:                 An integer representing the number of ensemble networks to train task specific
                                    parameters. The default is set to 1.
            Lv:                 An integer representing the number of ensemble networks to validate meta-parameters.
                                    The default is set to 1.
            p_dropout_base:     A float representing the dropout rate for the base network, with default set to 0.0
            cross_validate:     A boolean. Including it in the command will run the model with cross-validation.
            verbose:            A boolean. Including it in the command line will output additional information to the
                                    terminal for functions with verbose feature.
    """

    parser = argparse.ArgumentParser(
        description='Setup variables for PLATIPUS.')
    parser.add_argument('--datasource', type=str, default='drp_chem',
                        help='datasource to be used, defaults to drp_chem')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='Number of training samples per class')
    parser.add_argument('--n_way', type=int, default=1,
                        help='Number of classes per task, this is 2 for the chemistry data')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Epoch id to resume learning or perform testing')

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--inner_lr', type=float, default=1e-3,
                        help='Learning rate for task-specific parameters')
    parser.add_argument('--pred_lr', type=float, default=1e-1,
                        help='Learning rate for task-specific parameters during prediction (rather than training)')
    parser.add_argument('--num_inner_updates', type=int, default=5,
                        help='Number of gradient updates for task-specific parameters')
    parser.add_argument('--meta_lr', type=float, default=1e-3,
                        help='Learning rate of meta-parameters')
    parser.add_argument('--meta_batch_size', type=int, default=25,
                        help='Number of tasks sampled per outer loop')
    parser.add_argument('--num_epochs', type=int, default=50000,
                        help='How many outer loops are used to train')
    parser.add_argument('--num_epochs_save', type=int,
                        default=1000, help='How often should we save')

    parser.add_argument('--kl_reweight', type=float, default=1,
                        help='Reweight factor of the KL divergence between variational posterior q and prior p')

    parser.add_argument('--Lt', type=int, default=1,
                        help='Number of ensemble networks to train task specific parameters')
    parser.add_argument('--Lv', type=int, default=1,
                        help='Number of ensemble networks to validate meta-parameters')

    parser.add_argument('--p_dropout_base', type=float,
                        default=0., help='Dropout rate for the base network')
    parser.add_argument('--cross_validate', action='store_true')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args(args)
    return args


def initialize(models_list, args):
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
            L:                                  An integer representing the number of ensemble networks to train
                                                    task specific parameters.
            K:                                  An integer representing the number of ensemble networks to validate
                                                    meta-parameters.
            verbose:                            A boolean. representing whether it will output additional information to
                                                    the terminal for functions with verbose feature.
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
            num_weights:                        An integer representing the number of weights in the net. It is used to
                                                    avoid an linting error, even though it's inefficient.
            KL_reweight:                        An integer representing the re-weight factor of the KL divergence
                                                    between variational posterior q and prior p.
            dst_folder:                         A string representing the path to save models.
            graph_folder:                       A string representing the path to save all graphs. It will be under
                                                    dst_folder.
            active_learning_graph_folder:       A string representing the path to save the graphs in the active learning
                                                    section. It will be under graph_folder.
            resume_epoch:                       An integer representing the epoch id to resume learning or perform
                                                    testing.
            Theta:                              A dictionary representing the meta-parameters (big theta) for PLATIPUS,
                                                    with the mean of the model parameter, the variance of model
                                                    parameter, the learned diagonal covariance of posterior q, and two
                                                    learning rates as keys and tcorresponding torch.Tensor as values.
                                                    requires_grad is set to True for all tensors.
            op_Theta:                           A torch.optim object representing the optimizer used for meta-parameter
                                                    Theta. Currently using Adam as suggested in the PLATIPUS paper.
        """

    #args = parse_args()
    params = {}

    # Set up training using either GPU or CPU
    gpu_id = args.get('gpu_id', 0)
    device = torch.device('cuda:{0:d}'.format(
        gpu_id) if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    print()

    if device.type == 'cuda' and args.get('verbose', False):
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(
            torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    params['device'] = device
    params['gpu_id'] = gpu_id

    # Set up PLATIPUS using user inputs
    params['train_flag'] = args['train_flag']
    params['cross_validate'] = args['cross_validate']
    # Set up the value of k in k-shot learning

    print(f'{args["k_shot"]}-shot')
    params['num_training_samples_per_class'] = args['k_shot']

    # Total number of samples per class, need some extra for the outer loop update as well
    if params['train_flag']:
        params['num_total_samples_per_class'] = params['num_training_samples_per_class'] + 15
    else:
        params['num_total_samples_per_class'] = params['num_training_samples_per_class'] + 20

    # n-way: 2 for the chemistry data
    print(f"{args['n_way']}-way")
    params['num_classes_per_task'] = args['n_way']

    # Initialize the datasource and learning rate
    print(f'Dataset = {args["datasource"]}')
    params['datasource'] = args['datasource']
    params['test_data'] = args.get('test_data', False)

    print(f'Inner learning rate = {args["inner_lr"]}')
    params['inner_lr'] = args['inner_lr']
    params['pred_lr'] = args['pred_lr']

    # Set up the meta learning rate
    print(f'Meta learning rate = {args["meta_lr"]}')
    params['meta_lr'] = args['meta_lr']

    # Reducing this to 25 like in Finn et al.
    # Tells us how many tasks are per epoch and after how many tasks we should save the values of the losses
    params['num_tasks_per_epoch'] = args['meta_batch_size']
    params['num_tasks_save_loss'] = args['meta_batch_size']
    params['num_epochs'] = args['num_epochs']

    # How often should we save?
    params['num_epochs_save'] = args['num_epochs_save']

    # How many gradient updates we run on the inner loop
    print(f'Number of inner updates = {args["num_inner_updates"]}')
    params['num_inner_updates'] = args['num_inner_updates']

    # Dropout rate for the neural network
    params['p_dropout_base'] = args['p_dropout_base']

    # L as how many models we sample in the inner update and K as how many models we sample in validation
    print(f'L = {args["Lt"]}, K = {args["Lv"]}')
    params['L'] = args['Lt']
    params['K'] = args['Lv']
    params['verbose'] = args['verbose']

    # Set up the stats dictionary for later use
    stats_dict = create_stats_dict(models_list)
    params['cv_statistics'] = stats_dict

    # Save it in case we are running other models before PLATIPUS
    with open(os.path.join("./data", "cv_statistics.pkl"), "wb") as f:
        pickle.dump(params['cv_statistics'], f)

    if params['datasource'] == 'drp_chem':

        # Set number of training samples and number of total samples per class
        # These two values are hard-coded, corresponding to the values hard-coded in load_chem_dataset below
        params['num_total_samples_per_class'] = 40
        params['num_training_samples_per_class'] = 20

        if params['train_flag']:

            if params['cross_validate']:

                training_batches, validation_batches, testing_batches, counts = load_chem_dataset(k_shot=20,
                                                                                                  cross_validation=params[
                                                                                                      'cross_validate'],
                                                                                                  meta_batch_size=args[
                                                                                                      'meta_batch_size'],
                                                                                                  num_batches=250,
                                                                                                  verbose=args['verbose'],
                                                                                                  test=params.get('test_data', False))
                params['training_batches'] = training_batches
                params['validation_batches'] = validation_batches
                params['testing_batches'] = testing_batches
                params['counts'] = counts

                # Save for reproducibility
                write_pickle("./data/train_dump.pkl", training_batches)
                write_pickle("./data/val_dump.pkl", validation_batches)
                write_pickle("./data/test_dump.pkl", testing_batches)
                write_pickle("./data/counts_dump.pkl", counts)

            else:
                training_batches, testing_batches, counts = load_chem_dataset(k_shot=20,
                                                                              cross_validation=params['cross_validate'],
                                                                              meta_batch_size=args['meta_batch_size'],
                                                                              num_batches=250,
                                                                              verbose=args['verbose'],
                                                                              test=params.get('test_data', False))
                params['training_batches'] = training_batches
                params['testing_batches'] = testing_batches
                params['counts'] = counts

                # Save for reproducibility
                write_pickle("./data/train_dump_nocv.pkl",
                             {'training_data': training_batches})
                write_pickle("./data/test_dump_nocv.pkl", testing_batches)
                write_pickle("./data/counts_dump_nocv.pkl", counts)

        # Make sure we don't overwrite our batches if we are validating and testing
        else:
            if params['cross_validate']:
                params['training_batches'] = read_pickle(
                    "./data/train_dump.pkl")
                params['validation_batches'] = read_pickle(
                    "./data/val_dump.pkl")
                params['testing_batches'] = read_pickle("./data/test_dump.pkl")
                params['counts'] = read_pickle("./data/counts_dump.pkl")
            else:
                params['training_batches'] = read_pickle(
                    "./data/train_dump_nocv.pkl")['training_data']
                params['testing_batches'] = read_pickle(
                    "./data/test_dump_nocv.pkl")
                params['counts'] = read_pickle("./data/counts_dump_nocv.pkl")

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

    params['w_shape'] = net.get_weight_shape()
    # Inefficient to call this twice, but I was getting an annoying linting error
    print(f'Number of parameters of base model = {net.get_num_weights()}')
    params['num_weights'] = net.get_num_weights()

    # Weight on the KL loss
    print(f'KL reweight = {args["kl_reweight"]}')
    params['KL_reweight'] = args['kl_reweight']

    # Set up the path to save models
    params['dst_folder'] = save_model("PLATIPUS", params)

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
    active_learning_graph_folder = '{0:s}/active_learning_cv_graphs'.format(
        params['graph_folder'])
    if not os.path.exists(active_learning_graph_folder):
        os.makedirs(active_learning_graph_folder)
        print('No folder for active learning cross-validation graph storage found')
        print(f'Make folder to store cross-validation graphs at')
    else:
        print('Found existing folder. Active learning cross-validation graphs will be stored at')
    print(active_learning_graph_folder)
    params['active_learning_graph_folder'] = active_learning_graph_folder

    # In the case that we are loading a model, resume epoch will not be zero
    params['resume_epoch'] = args['resume_epoch']
    if params['resume_epoch'] == 0:
        # Initialize meta-parameters
        # Theta is capital theta in the PLATIPUS paper, it holds everything we need
        # 'mean' is the mean model parameters
        # 'logSigma' and 'logSigma_q' are the variance of the base and variational distributions
        # the two 'gamma' vectors are the learning rate vectors
        Theta = initialzie_theta_platipus(params)
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
            checkpoint_file = os.path.join(
                params["dst_folder"], checkpoint_filename)
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
        op_Theta = set_optim_platipus(Theta, params["meta_lr"])

        if params['resume_epoch'] > 0:
            op_Theta.load_state_dict(saved_checkpoint['op_Theta'])
            # Set the meta learning rates appropriately
            op_Theta.param_groups[0]['lr'] = params['meta_lr']
            op_Theta.param_groups[1]['lr'] = params['meta_lr']

        params['op_Theta'] = op_Theta

    print()
    return params


if __name__ == "__main__":
    models_list = ["PLATIPUS", "MAML", "SVM", "KNN", "RandomForest"]
    args = vars(parse_args(sys.argv[1:]))
    params = initialize(models_list, args)
    main(params)
