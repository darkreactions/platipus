'''
python maml.py --datasource=sine_line --train --num_inner_updates=5 --k_shot=5 --n_way=1 --inner_lr=1e-3 --meta_lr=1e-3  --num_epochs=10000 --resume_epoch=0 

python maml.py --datasource=drp_chem --k_shot=20 --n_way=2 --inner_lr=1e-3 --meta_lr=1e-3 --meta_batch_size=10 --num_epochs=3000 --cross_validate --resume_epoch=0 --verbose --p_dropout_base=0.4

'''
import torch

import numpy as np
import random
import itertools

import os
import sys

from FC_net import FCNet

import pickle


import argparse


# Set up the initial variables for running MAML
def parse_args():
    parser = argparse.ArgumentParser(description='Setup variables for MAML.')

    parser.add_argument('--datasource', type=str, default='drp_chem', help='datasource to be used, default is drp_chem')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of training samples per class or k-shot')
    parser.add_argument('--n_way', type=int, default=1, help='Number of classes per task, this is 2 for the chemistry data')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--inner_lr', type=float, default=1e-3, help='Learning rate for task-specific parameters')
    parser.add_argument('--num_inner_updates', type=int, default=5, help='Number of gradient updates for task-specific parameters')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta-parameters')
    parser.add_argument('--meta_batch_size', type=int, default=25, help='Number of tasks sampled per outer loop')
    parser.add_argument('--num_epochs', type=int, default=1000, help='How many outer loops are used to train')

    parser.add_argument('--num_val_tasks', type=int, default=100, help='Number of validation tasks')
    parser.add_argument('--uncertainty', dest='uncertainty_flag', action='store_true')
    parser.add_argument('--no_uncertainty', dest='uncertainty_flag', action='store_false')
    parser.set_defaults(uncertainty_flag=True)

    parser.add_argument('--p_dropout_base', type=float, default=0., help='Dropout rate for the base network')
    parser.add_argument('--cross_validate', action='store_true')


    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    return args 


# Do all of the initialization that we need for the rest of the code to run...
# Initializes a dictionary of parameters corresponding to the arguments various 
# functions will need 
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

    # How many gradient updates we run on the inner loop
    print(f'Number of inner updates = {args.num_inner_updates}')
    params['num_inner_updates'] = args.num_inner_updates

    # Dropout rate for the neural network
    params['p_dropout_base'] = args.p_dropout_base

    if params['datasource'] == 'drp_chem':
        
        # I am only running MAML to compare with PLATIPUS, thus assume this data already exists
        if params['cross_validate']:
            with open(os.path.join(".\\data","train_dump.pkl"), "rb") as f:
                params['training_batches'] = pickle.load(f)
            with open(os.path.join(".\\data","val_dump.pkl"), "rb") as f:
                params['validation_batches'] = pickle.load(f)
            with open(os.path.join(".\\data","test_dump.pkl"), "rb") as f:
                params['testing_batches'] = pickle.load(f)
            with open(os.path.join(".\\data","counts_dump.pkl"), "rb") as f:
                params['counts'] = pickle.load(f)

        net = FCNet(
            dim_input= 51,
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
        weights = [counts['total'][0]/counts['total'][0], counts['total'][0]/counts['total'][1]]
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
                theta[key] = torch.zeros(params['w_shape'][key], device=device, requires_grad=True)      
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

# Use this for cross validation, we need to set up a new loss function too
def reinitialize_model_params(params):
    # Initialise meta-parameters for MAML
    theta = {}
    for key in params['w_shape'].keys():
        if 'b' in key:
            theta[key] = torch.zeros(params['w_shape'][key], device=params['device'], requires_grad=True)      
        else:
            theta[key] = torch.empty(params['w_shape'][key], device=params['device'])
            torch.nn.init.xavier_normal_(theta[key], gain=1.)
            theta[key].requires_grad_()

    params['theta'] = theta

    op_theta = torch.optim.Adam(params=theta.values(), lr=params['meta_lr'])
    params['op_theta'] = op_theta


# Main driver code
def main():

    # Do the massive initialization and get back a dictionary instead of using 
    # global variables
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
                        print('Found existing folder. Meta-parameters will be stored at')
                    print(dst_folder)
                    params['dst_folder'] = dst_folder

                    # Adjust the loss function for each amine
                    amine_counts = params['counts'][amine]
                    weights = [amine_counts[0]/amine_counts[0], amine_counts[0]/amine_counts[1]]
                    print('Using the following weights for loss function:', weights)
                    class_weights = torch.tensor(weights, device=params['device'])
                    params['loss_fn'] = torch.nn.CrossEntropyLoss(class_weights)

                    # Train the model then reinitialize a new one
                    meta_train(params, amine)
                    reinitialize_model_params(params)


    elif params['resume_epoch'] > 0:

        if not uncertainty_flag:
            accs, all_task_names = meta_validation(
                datasubset=test_set,
                num_val_tasks=num_val_tasks,
                return_uncertainty=uncertainty_flag
            )
            with open(file='maml_{0:s}_{1:d}_{2:d}_accuracies.csv'.format(datasource, num_classes_per_task, num_training_samples_per_class), mode='w') as result_file:
                for acc, classes_in_task in zip(accs, all_task_names):
                    row_str = ''
                    for class_in_task in classes_in_task:
                        row_str = '{0}{1},'.format(row_str, class_in_task)
                    result_file.write('{0}{1}\n'.format(row_str, acc))
        else:
            corrects, probs = meta_validation(
                datasubset=test_set,
                num_val_tasks=num_val_tasks,
                return_uncertainty=uncertainty_flag
            )
            with open(file='maml_{0:s}_correct_prob.csv'.format(datasource), mode='w') as result_file:
                for correct, prob in zip(corrects, probs):
                    result_file.write('{0}, {1}\n'.format(correct, prob))
                    # print(correct, prob)
    else:
        sys.exit('Unknown action')

# Run the actual meta training for MAML
def meta_train(params, amine=None):
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
    num_epochs_save = 1000


    for epoch in range(resume_epoch, resume_epoch + num_epochs):
        print(f"Starting epoch {epoch}")

        if datasource == 'drp_chem':
            training_batches = params['training_batches']
            if params['cross_validate']:
                b_num = np.random.choice(len(training_batches[amine]))
                batch = training_batches[amine][b_num]
            else:
                b_num = np.random.choice(len(training_batches))
                batch = training_batches[b_num]
            x_train, y_train, x_val, y_val = torch.from_numpy(batch[0]).float().to(params['device']), torch.from_numpy(batch[1]).long().to(params['device']), \
                torch.from_numpy(batch[2]).float().to(params['device']), torch.from_numpy(batch[3]).long().to(params['device'])

        # variables used to store information of each epoch for monitoring purpose
        meta_loss_saved = [] # meta loss to save
        val_accuracies = []
        train_accuracies = []

        task_count = 0 # a counter to decide when a minibatch of task is completed to perform meta update
        meta_loss = 0 # accumulate the loss of many ensambling networks to descent gradient for meta update
        num_meta_updates_count = 0

        meta_loss_avg_print = 0 # compute loss average to print

        meta_loss_avg_save = [] # meta loss to save

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
                meta_loss = meta_loss/num_tasks_per_epoch

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
                    meta_loss_avg_save.append(meta_loss_avg_print/num_meta_updates_count)
                    print('{0:d}, {1:2.4f}'.format(
                        task_count,
                        meta_loss_avg_save[-1]
                    ))

                    num_meta_updates_count = 0
                    meta_loss_avg_print = 0
                
                if (task_count % num_tasks_save_loss == 0):
                    meta_loss_saved.append(np.mean(meta_loss_avg_save))

                    meta_loss_avg_save = []

                    # print('Saving loss...')
                    # if datasource != 'sine_line':
                    #     val_accs, _ = meta_validation(
                    #         datasubset=val_set,
                    #         num_val_tasks=num_val_tasks,
                    #         return_uncertainty=False)
                    #     val_acc = np.mean(val_accs)
                    #     val_ci95 = 1.96*np.std(val_accs)/np.sqrt(num_val_tasks)
                    #     print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
                    #     val_accuracies.append(val_acc)

                    #     train_accs, _ = meta_validation(
                    #         datasubset=train_set,
                    #         num_val_tasks=num_val_tasks,
                    #         return_uncertainty=False)
                    #     train_acc = np.mean(train_accs)
                    #     train_ci95 = 1.96*np.std(train_accs)/np.sqrt(num_val_tasks)
                    #     print('Train accuracy = {0:2.4f} +/- {1:2.4f}\n'.format(train_acc, train_ci95))
                    #     train_accuracies.append(train_acc)
                
                # Reset meta loss
                meta_loss = 0

            if (task_count >= num_tasks_per_epoch):
                break

        if ((epoch + 1)% num_epochs_save == 0):
            checkpoint = {
                'theta': theta,
                'meta_loss': meta_loss_saved,
                'val_accuracy': val_accuracies,
                'train_accuracy': train_accuracies,
                'op_theta': op_theta.state_dict()
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


# Get the prediction on some input data
# This is used for both training and testing, hence why y_v is optional
def get_task_prediction(x_t, y_t, x_v, params, y_v=None):
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
        q[key] = theta[key] - inner_lr*gradients[key]

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
            q[key] = q[key] - inner_lr*gradients[key]
    
    # Now predict on the validation or test data
    y_pred_v = net.forward(x=x_v, w=q, p_dropout=0)
    
    # Then we were operating on testing data, return our predictions
    if y_v is None:
        return y_pred_v
    # We were operating on validation data, return our loss
    else:
        loss_NLL = loss_fn(y_pred_v, y_v)
        return loss_NLL
        

# This method is used for testing, we call it after we have trained a model using the 
# methods above
def meta_validation(datasubset, num_val_tasks, params, return_uncertainty=False):
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
    all_task_names = list(itertools.combinations(all_class_names, r=num_classes_per_task))

    if train_flag:
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

        y_pred_v = get_task_prediction(x_t, y_t, x_v, y_v=None)
        y_pred = sm_loss(y_pred_v)

        prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
        correct = (labels_pred == y_v)
        corrects.extend(correct.detach().cpu().numpy())

        accuracy = torch.sum(correct, dim=0).item()/total_validation_samples
        accuracies.append(accuracy)

        probability_pred.extend(prob_pred.detach().cpu().numpy())

        task_count += 1
        if not train_flag:
            print(task_count)
        if task_count >= num_val_tasks:
            break
    if not return_uncertainty:
        return accuracies, all_task_names
    else:
        return corrects, probability_pred

if __name__ == "__main__":
    main()
