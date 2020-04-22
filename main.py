'''

python main.py --datasource=sine_line --k_shot=5 --n_way=1 --inner_lr=1e-3 --meta_lr=1e-3 --Lt=1 --Lv=1 --kl_reweight=1 --num_epochs=25000 --resume_epoch=0 --verbose

python main.py --datasource=sine_line --k_shot=5 --n_way=1 --inner_lr=1e-3 --meta_lr=1e-3 --Lt=1 --Lv=10 --kl_reweight=1 --num_epochs=1000 --resume_epoch=25000 --test --num_val_tasks=0

'''

import torch
import numpy as np
import random
import itertools

import os
import sys

import argparse
from utils import get_task_sine_line_data
# from utils import load_chem_dataset
from data_generator import DataGenerator
from FC_net import FCNet

import matplotlib
from matplotlib import pyplot as plt


# Set up the initial variables for running PLATIPUS
def parse_args():
    parser = argparse.ArgumentParser(description='Setup variables for PLATIPUS.')
    parser.add_argument('--datasource', type=str, default='sine_line', help='datasource to be used, sine_line or drp_chem')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of training samples per class')
    parser.add_argument('--n_way', type=int, default=1, help='Number of classes per task, this is 2 for the chemistry data')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--inner_lr', type=float, default=1e-3, help='Learning rate for task-specific parameters')
    parser.add_argument('--num_inner_updates', type=int, default=5, help='Number of gradient updates for task-specific parameters')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta-parameters')
    parser.add_argument('--meta_batch_size', type=int, default=25, help= 'Number of tasks sampled per outer loop')
    parser.add_argument('--num_epochs', type=int, default=50000, help='How many outer loops are used to train')

    parser.add_argument('--kl_reweight', type=float, default=1, help='Reweight factor of the KL divergence between variational posterior q and prior p')

    parser.add_argument('--Lt', type=int, default=1, help='Number of ensemble networks to train task specific parameters')
    parser.add_argument('--Lv', type=int, default=1, help='Number of ensemble networks to validate meta-parameters')
    
    parser.add_argument('--num_val_tasks', type=int, default=100, help='Number of validation tasks')
    parser.add_argument('--uncertainty', dest='uncertainty_flag', action='store_true')
    parser.add_argument('--no_uncertainty', dest='uncertainty_flag', action='store_false')
    parser.set_defaults(uncertainty_flag=True)

    parser.add_argument('--p_dropout_base', type=float, default=0., help='Dropout rate for the base network')
    parser.add_argument('--datasubset', type=str, default='sine', help='sine or line')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    return args 

# Do all of the initialization that we need for the rest of the code to run...
# It had a bunch of global variables before, so trying to use a parameters dictionary to circumvent that
# without blowing up the number of parameters in each function
def initialize():

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
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    params['device'] = device
    params['gpu_id'] = gpu_id    

    # Set up PLATIPUS using user inputs
    params['train_flag'] = args.train_flag
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

    # How many gradient updates we run on the inner loop
    print(f'Number of inner updates = {args.num_inner_updates}')
    params['num_inner_updates'] = args.num_inner_updates

     # Dropout rate for the neural network
    params['p_dropout_base'] = args.p_dropout_base
    # Sine or line if we are dealing with the sine-line task
    params['datasubset'] = args.datasubset

    # I think of L as how many models we sample in the inner update and K as how many models we sample in validation 
    print(f'L = {args.Lt}, K = {args.Lv}')
    params['L'] = args.Lt
    params['K'] = args.Lv

    if params['datasource'] == 'sine_line':
        net = FCNet(
            dim_input=1,
            dim_output=params['num_classes_per_task'],
            num_hidden_units=(40, 40),
            device=device
        )
        # bernoulli probability to pick sine or (1-p_sine) for line
        params['p_sine'] = 0.5
        params['net'] = net 
        params['loss_fn'] = torch.nn.MSELoss()

    elif params['datasource'] == 'drp_chem':

        training_batches, testing_batches, dataset_dimension = load_chem_dataset(k_shot=20, num_batches=250, verbose=args.verbose)
        params['training_batches'] = training_batches
        params['testing_batches'] = testing_batches

        net = FCNet(
            dim_input=dataset_dimension,
            dim_output=params['num_classes_per_task'],
            num_hidden_units=(400, 300, 200),
            device=device
        )
        params['net'] = net 
        params['loss_fn'] = torch.nn.CrossEntropyLoss()
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

    # Used in the validation step 
    params['num_val_tasks'] = args.num_val_tasks

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
        # Here we are loading a previously trained model
        print('Restore previous Theta...')
        print('Resume epoch {0:d}'.format(params['resume_epoch']))
        checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt')\
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

    params['Theta'] = Theta

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
    params['uncertainty_flag'] = args.uncertainty_flag
    print()
    return params

# Main driver code
def main():

    # Do the massive initialization and get back a dictionary instead of using 
    # global variables
    params = initialize()

    if params['train_flag']:
        meta_train(params)
    elif params['resume_epoch'] > 0:
        if params['datasource'] == 'sine_line':
            cal_data = meta_validation(datasubset=params['datasubset'], num_val_tasks=params['num_val_tasks'], params=params)

            if params['num_val_tasks'] > 0:
                cal_data = np.array(cal_data)
                np.savetxt(fname='PLATIPUS_{0:s}_calibration.csv'.format(params['datasource']), X=cal_data, delimiter=',')
        else:
            # Will need to replace this with chemistry related testing code 
            # A lot of this is currently broken, need to use the params dict in here too
            if not uncertainty_flag:
                accs, all_task_names = meta_validation(
                    datasubset='test',
                    num_val_tasks=num_val_tasks,
                    return_uncertainty=uncertainty_flag,
                    params=params
                )
                with open(file='{0:s}_{1:d}_{2:d}_accuracies.csv'.format(datasource, num_classes_per_task, num_training_samples_per_class), mode='w') as result_file:
                    for acc, classes_in_task in zip(accs, all_task_names):
                        row_str = ''
                        for class_in_task in classes_in_task:
                            row_str = '{0}{1},'.format(row_str, class_in_task)
                        result_file.write('{0}{1}\n'.format(row_str, acc))
            else:
                corrects, probs = meta_validation(
                    datasubset='test',
                    num_val_tasks=num_val_tasks,
                    return_uncertainty=uncertainty_flag
                )
                with open(file='platipus_{0:s}_correct_prob.csv'.format(datasource), mode='w') as result_file:
                    for correct, prob in zip(corrects, probs):
                        result_file.write('{0}, {1}\n'.format(correct, prob))
                        # print(correct, prob)
    else:
        sys.exit('Unknown action')

# Run the actual meta training for PLATIPUS
# Lots of cool stuff happening in this function
def meta_train(params):
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
    
    # How often should we do a printout?
    num_meta_updates_print = 1
    # How often should we save?
    num_epochs_save = 1000

    if datasource == 'sine_line':
        data_generator = DataGenerator(
            num_samples=num_total_samples_per_class,
            device=device
        )
        # create dummy sampler
        all_class = [0]*100
        sampler = torch.utils.data.sampler.RandomSampler(data_source=all_class)
        train_loader = torch.utils.data.DataLoader(
            dataset=all_class,
            batch_size=num_classes_per_task,
            sampler=sampler,
            drop_last=True
        )
    else:
        all_class = all_class_train
        embedding = embedding_train
        sampler = torch.utils.data.sampler.RandomSampler(
            data_source=list(all_class.keys()),
            replacement=False
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=list(all_class.keys()),
            batch_size=num_classes_per_task,
            sampler=sampler,
            drop_last=True
        )
    #endregion

    for epoch in range(resume_epoch, resume_epoch + num_epochs):
        print(f"Starting epoch {epoch}")
        # variables used to store information of each epoch for monitoring purpose
        meta_loss_saved = [] # meta loss to save
        kl_loss_saved = []
        val_accuracies = []
        train_accuracies = []

        meta_loss = 0 # accumulate the loss of many ensambling networks to descent gradient for meta update
        num_meta_updates_count = 0

        meta_loss_avg_print = 0 # compute loss average to print

        kl_loss = 0
        kl_loss_avg_print = 0

        meta_loss_avg_save = [] # meta loss to save
        kl_loss_avg_save = []

        task_count = 0 # a counter to decide when a minibatch of task is completed to perform meta update

        while (task_count < num_tasks_per_epoch):
            for class_labels in train_loader:
                if datasource == 'sine_line':
                    p_sine = params['p_sine']
                    x_t, y_t, x_v, y_v = get_task_sine_line_data(
                        data_generator=data_generator,
                        p_sine=p_sine,
                        num_training_samples=num_training_samples_per_class,
                        noise_flag=True
                    )
                elif datasource == 'drp_chem':
                    pass
                else:
                    sys.exit('Unknown dataset')
                
                loss_i, KL_q_p = get_training_loss(x_t, y_t, x_v, y_v, params)

                if torch.isnan(loss_i).item():
                    sys.exit('NaN error')

                # accumulate meta loss
                meta_loss = meta_loss + loss_i + KL_q_p
                kl_loss = kl_loss + KL_q_p

                task_count = task_count + 1

                if task_count % num_tasks_per_epoch == 0:
                    meta_loss = meta_loss/num_tasks_per_epoch
                    kl_loss /= num_tasks_per_epoch

                    # accumulate into different variables for printing purpose
                    meta_loss_avg_print += meta_loss.item()
                    kl_loss_avg_print += kl_loss.item()

                    op_Theta.zero_grad()
                    meta_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters=Theta['mean'].values(),
                        max_norm=10
                    )
                    torch.nn.utils.clip_grad_norm_(
                        parameters=Theta['logSigma'].values(),
                        max_norm=10
                    )
                    op_Theta.step()

                    # Printing losses
                    num_meta_updates_count += 1
                    if (num_meta_updates_count % num_meta_updates_print == 0):
                        meta_loss_avg_save.append(meta_loss_avg_print/num_meta_updates_count)
                        kl_loss_avg_save.append(kl_loss_avg_print/num_meta_updates_count)
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

                        # if datasource != 'sine_line':
                        #     print('Saving loss...')
                        #     val_accs, _ = meta_validation(
                        #         datasubset='val',
                        #         num_val_tasks=num_val_tasks,
                        #         return_uncertainty=False)
                        #     val_acc = np.mean(val_accs)
                        #     val_ci95 = 1.96*np.std(val_accs)/np.sqrt(num_val_tasks)
                        #     print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
                        #     val_accuracies.append(val_acc)

                        #     train_accs, _ = meta_validation(
                        #         datasubset= 'train',
                        #         num_val_tasks=num_val_tasks,
                        #         return_uncertainty=False)
                        #     train_acc = np.mean(train_accs)
                        #     train_ci95 = 1.96*np.std(train_accs)/np.sqrt(num_val_tasks)
                        #     print('Train accuracy = {0:2.4f} +/- {1:2.4f}'.format(train_acc, train_ci95))
                        #     train_accuracies.append(train_acc)

                        #     print()
                    
                    # reset meta loss
                    meta_loss = 0
                    kl_loss = 0

                if (task_count >= num_tasks_per_epoch):
                    break

        if ((epoch + 1)% num_epochs_save == 0):
            checkpoint = {
                'Theta': Theta,
                'kl_loss': kl_loss_saved,
                'meta_loss': meta_loss_saved,
                'val_accuracy': val_accuracies,
                'train_accuracy': train_accuracies,
                'op_Theta': op_Theta.state_dict()
            }
            print('SAVING WEIGHTS...')
            checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt')\
                        .format(datasource,
                                num_classes_per_task,
                                num_training_samples_per_class,
                                epoch + 1)
            print(checkpoint_filename)
            dst_folder = params['dst_folder']
            torch.save(checkpoint, os.path.join(dst_folder, checkpoint_filename))
        print()


# Determines the KL and Meta Objectlive loss on a set of training
# and validation data 
# Steps correspond to steps in the PLATIPUS TRAINING algorithm in Finn et al
def get_training_loss(x_t, y_t, x_v, y_v, params):
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
        q['mean'][key] = Theta['mean'][key] - Theta['gamma_q'][key]*loss_query_gradients[key]
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
            phi_i[key] = w[key] - inner_lr*loss_vfe_gradients[key]
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
                phi_i[key] = phi_i[key] - inner_lr*loss_vfe_gradients[key]

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
        p['mean'][key] = Theta['mean'][key] - Theta['gamma_p'][key]*loss_train_gradients[key]
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
        KL_q_p += torch.sum(torch.exp(2*(q['logSigma'][key] - p['logSigma'][key])) \
                + (p['mean'][key] - q['mean'][key])**2/torch.exp(2*p['logSigma'][key]))\
                    + torch.sum(2*(p['logSigma'][key] - q['logSigma'][key]))
    KL_q_p = (KL_q_p - num_weights)/2
    return loss_query, KL_q_p


# Update the PLATIPUS model on some training data then obtain its output
# on some testing data
# Steps correspond to steps in the PLATIPUS TESTING algorithm in Finn et al
def get_task_prediction(x_t, y_t, x_v, params):
    # As usual, begin by unpacking the parameters we need
    Theta = params['Theta']
    net = params['net']
    p_dropout_base = params['p_dropout_base']
    loss_fn = params['loss_fn']
    K = params['K']
    num_inner_updates = params['num_inner_updates']
    inner_lr = params['inner_lr']

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
        p['mean'][key] = Theta['mean'][key] - Theta['gamma_p'][key]*loss_train_gradients[key]
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
            phi_i[key] = w[key] - inner_lr*loss_train_gradients[key]
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
                phi_i[key] = phi_i[key] - inner_lr*loss_train_gradients[key]

    y_pred_v = []
    # Now get the model predictions on the validation/test data x_v by calling the forward method
    for phi_i in phi:
        y_pred_temp = net.forward(x=x_v, w=phi_i)
        y_pred_v.append(y_pred_temp)
    return y_pred_v

# This method is used for testing, we call it after we have trained a model using the 
# methods above
def meta_validation(datasubset, num_val_tasks, params, return_uncertainty=False):
    # Start by unpacking any necessary parameters
    datasource = params['datasource']
    device = params['device']
    gpu_id = params['gpu_id']
    num_training_samples_per_class = params['num_training_samples_per_class']
    num_classes_per_task = params['num_classes_per_task']
    # Assume root directory is current directory
    dst_folder_root = '.'
    num_epochs_save = params['num_epochs_save']


    if datasource == 'sine_line':
        # Set up the x-axis for plotting
        x0 = torch.linspace(start=-5, end=5, steps=100, device=device).view(-1, 1)

        if num_val_tasks == 0:

            matplotlib.rcParams['xtick.labelsize'] = 16
            matplotlib.rcParams['ytick.labelsize'] = 16
            matplotlib.rcParams['axes.labelsize'] = 18

            num_stds = 2
            data_generator = DataGenerator(
                num_samples=num_training_samples_per_class,
                device=device
            )
            if datasubset == 'line':
                x_t, y_t, amp, phase = data_generator.generate_sinusoidal_data(noise_flag=True)
                y0 = amp*torch.sin(x0 + phase)
            else:
                x_t, y_t, slope, intercept = data_generator.generate_line_data(noise_flag=True)
                y0 = slope*x0 + intercept

            # These are the PLATIPUS predictions
            y_preds = get_task_prediction(x_t=x_t, y_t=y_t, x_v=x0, params=params)

            # Load MAML data for comparison
            # This requires that we already trained a MAML model, see maml.py
            maml_folder = '{0:s}/MAML_sine_line_1way_5shot'.format(dst_folder_root)
            maml_filename = 'sine_line_{0:d}way_{1:d}shot_{2:s}.pt'.format(num_classes_per_task, num_training_samples_per_class, '{0:d}')

            # Try to find the model that was training for the most epochs
            # If you get an error here, your MAML model was not saved at epochs of increment num_epochs_save
            # By default that value is 1000 and is set in initialize()
            i = num_epochs_save
            maml_checkpoint_filename = os.path.join(maml_folder, maml_filename.format(i))
            while(os.path.exists(maml_checkpoint_filename)):
                i = i + num_epochs_save
                maml_checkpoint_filename = os.path.join(maml_folder, maml_filename.format(i))

            # We overshoot by one increment of num_epochs_save, so need to subtract
            maml_checkpoint = torch.load(
                os.path.join(maml_folder, maml_filename.format(i - num_epochs_save)),
                map_location=lambda storage,
                loc: storage.cuda(gpu_id)
            )

            # In the MAML model it is lower-case theta
            # We obtain the MAML model's predictions here
            Theta_maml = maml_checkpoint['theta']
            y_pred_maml = get_task_prediction_maml(x_t=x_t, y_t=y_t, x_v=x0, meta_params=Theta_maml, params=params)

            # Now plot the results
            # For PLATIPUS plot a range using the standard deviation across the K models
            # Need to call torch.stack for this to work properly
            _, ax = plt.subplots(figsize=(5, 5))
            y_top = torch.squeeze(torch.mean(torch.stack(y_preds), dim=0) + num_stds*torch.std(torch.stack(y_preds), dim=0))
            y_bottom = torch.squeeze(torch.mean(torch.stack(y_preds), dim=0) - num_stds*torch.std(torch.stack(y_preds), dim=0))

            ax.fill_between(
                x=torch.squeeze(x0).cpu().numpy(),
                y1=y_bottom.cpu().detach().numpy(),
                y2=y_top.cpu().detach().numpy(),
                alpha=0.25,
                color='C3',
                zorder=0,
                label='PLATIPUS'
            )
            ax.plot(x0.cpu().numpy(), y0.cpu().numpy(), color='C7', linestyle='-', linewidth=3, zorder=1, label='Ground truth')
            ax.plot(x0.cpu().numpy(), y_pred_maml.cpu().detach().numpy(), color='C2', linestyle='--', linewidth=3, zorder=2, label='MAML')
            ax.scatter(x=x_t.cpu().numpy(), y=y_t.cpu().numpy(), color='C0', marker='^', s=300, zorder=3, label='Data')
            plt.xticks([-5, -2.5, 0, 2.5, 5])
            plt.legend(loc='best')
            plt.show()
            plt.savefig(fname='img/mixed_sine_temp.svg', format='svg')
            return 0
        else:
            from scipy.special import erf
            p_sine = params['p_sine']
            quantiles = np.arange(start=0., stop=1.1, step=0.1)
            cal_data = []

            data_generator = DataGenerator(num_samples=num_training_samples_per_class, device=device)
            for _ in range(num_val_tasks):
                binary_flag = np.random.binomial(n=1, p=p_sine)
                if (binary_flag == 0):
                    # generate sinusoidal data
                    x_t, y_t, amp, phase = data_generator.generate_sinusoidal_data(noise_flag=True)
                    y0 = amp*torch.sin(x0 + phase)
                else:
                    # generate line data
                    x_t, y_t, slope, intercept = data_generator.generate_line_data(noise_flag=True)
                    y0 = slope*x0 + intercept
                y0 = y0.view(1, -1).cpu().numpy() # row vector
                
                y_preds = torch.stack(get_task_prediction(x_t=x_t, y_t=y_t, x_v=x0, params=params)) # K x len(x0)

                y_preds_np = torch.squeeze(y_preds, dim=-1).detach().cpu().numpy()
                
                y_preds_quantile = np.quantile(a=y_preds_np, q=quantiles, axis=0, keepdims=False)

                # ground truth cdf
                std = data_generator.noise_std
                cal_temp = (1 + erf((y_preds_quantile - y0)/(np.sqrt(2)*std)))/2
                cal_temp_avg = np.mean(a=cal_temp, axis=1) # average for a task
                cal_data.append(cal_temp_avg)
            return cal_data
    else:
        accuracies = []
        corrects = []
        probability_pred = []

        total_validation_samples = (num_total_samples_per_class - num_training_samples_per_class)*num_classes_per_task
        
        if datasubset == 'train':
            all_class_data = all_class_train
            embedding_data = embedding_train
        elif datasubset == 'val':
            all_class_data = all_class_val
            embedding_data = embedding_val
        elif datasubset == 'test':
            all_class_data = all_class_test
            embedding_data = embedding_test
        else:
            sys.exit('Unknown datasubset for validation')
        
        all_class_names = list(all_class_data.keys())
        all_task_names = itertools.combinations(all_class_names, r=num_classes_per_task)

        if train_flag:
            all_task_names = list(all_task_names)
            random.shuffle(all_task_names)

        task_count = 0
        for class_labels in all_task_names:
            x_t, y_t, x_v, y_v = get_task_image_data(
                all_class_data,
                embedding_data,
                class_labels,
                num_total_samples_per_class,
                num_training_samples_per_class,
                device)
            
            y_pred_v = sm_loss(torch.stack(get_task_prediction(x_t, y_t, x_v)))
            y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)

            prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
            correct = (labels_pred == y_v)
            corrects.extend(correct.detach().cpu().numpy())

            accuracy = torch.sum(correct, dim=0).item()/total_validation_samples
            accuracies.append(accuracy)

            probability_pred.extend(prob_pred.detach().cpu().numpy())

            task_count += 1
            if not train_flag:
                print(task_count)
            if (task_count >= num_val_tasks):
                break
        if not return_uncertainty:
            return accuracies, all_task_names
        else:
            return corrects, probability_pred

# Run a forward pass to obtain predictions given a MAML model
# It is required that a MAML model has already been trained
# Also make sure the MAML model has the same architecture as the PLATIPUS model
# or elese the call to net.forward() will give you problems
def get_task_prediction_maml(x_t, y_t, x_v, meta_params, params):
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
        q[key] = meta_params[key] - inner_lr*gradients[key]

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
            q[key] = q[key] - inner_lr*gradients[key]
    
    # Finally call the forward function
    y_pred_v = net.forward(x=x_v, w=q)
    return y_pred_v

# Use the PLATIPUS means and variances to actually generate a set of weights
# ie a plausible model 
def generate_weights(meta_params, params):
    device = params['device']
    w = {}
    for key in meta_params['mean'].keys():
        eps_sampled = torch.randn(meta_params['mean'][key].shape, device=device)
        # Use the epsilon reparameterization trick in VI
        w[key] = meta_params['mean'][key] + eps_sampled*torch.exp(meta_params['logSigma'][key])
    return w

# Helps us create a data structure to store model weights during gradient updates
def initialise_dict_of_dict(key_list):
    q = dict.fromkeys(['mean', 'logSigma'])
    for para in q.keys():
        q[para] = {}
        for key in key_list:
            q[para][key] = 0
    return q

if __name__ == "__main__":
    main()