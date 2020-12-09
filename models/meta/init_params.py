import torch
import logging
from pathlib import Path
from utils import create_stats_dict, load_chem_dataset, save_model, write_pickle

from models.meta.main import initialzie_theta_platipus, set_optim_platipus
from utils.dataset_class import DataSet, Setting
import pickle


def init_params(args):
    #with open('./data/full_frozen_dataset.pkl', 'rb') as f:
    #    dataset = pickle.load(f)

    params = {**args}
    gpu_id = params.get('gpu_id', 0)
    # Set up training using either GPU or CPU

    logging.info(f'{args["k_shot"]}-shot')

    # Total number of samples per class, need some extra for the outer loop update as well
    params['num_total_samples_per_class'] = 40
    if params['train_flag']:
        params['num_total_samples_per_class'] = params['k_shot'] + 15
    else:
        params['num_total_samples_per_class'] = params['k_shot'] + 20

    # n-way: 2 for the chemistry data
    logging.info(f"{params['n_way']}-way")

    # Initialize the datasource and learning rate
    logging.info(f'Dataset = {params["datasource"]}')
    logging.info(f'Inner learning rate = {params["inner_lr"]}')

    # Set up the meta learning rate
    print(f'Meta learning rate = {params["meta_lr"]}')

    # Reducing this to 25 like in Finn et al.
    # Tells us how many tasks are per epoch and after how many tasks we should save the values of the losses
    params['num_tasks_per_epoch'] = params['meta_batch_size']
    params['num_tasks_save_loss'] = params['meta_batch_size']

    # How many gradient updates we run on the inner loop
    logging.info(f'Number of inner updates = {args["num_inner_updates"]}')

    # L as how many models we sample in the inner update and K as how many models we sample in validation
    logging.info(f'L = {args["Lt"]}, K = {args["Lv"]}')

    # Set up the stats dictionary for later use
    stats_dict = create_stats_dict([params['model_name']])
    params['cv_statistics'] = stats_dict

    

    # Set number of training samples and number of total samples per class
    # These two values are hard-coded, corresponding to the values hard-coded in load_chem_dataset below
    params['num_total_samples_per_class'] = 40
    #params['num_training_samples_per_class'] = 20

    if params['cross_validate']:
        training_batches, validation_batches, testing_batches, counts \
            = load_chem_dataset(k_shot=args['k_shot'],
                                cross_validation=params['cross_validate'],
                                meta_batch_size=args['meta_batch_size'],
                                num_batches=250,
                                verbose=args['verbose'],
                                test=params.get('test_data', False))
        params['validation_batches'] = validation_batches
        # write_pickle(params['dst_folder'] /
        #             Path("val_dump.pkl"), validation_batches)
    else:
        training_batches, testing_batches, counts = \
            load_chem_dataset(k_shot=args['k_shot'],
                              cross_validation=params['cross_validate'],
                              meta_batch_size=args['meta_batch_size'],
                              num_batches=250,
                              verbose=args['verbose'],
                              test=params.get('test_data', False))
        params['validation_batches'] = {}
    params['training_batches'] = training_batches
    params['testing_batches'] = testing_batches
    params['counts'] = counts

    # Save for reproducibility
    # Set up the path to save models
    params['dst_folder'] = Path(save_model(params['model_name'], params))

    write_pickle(params['dst_folder'] /
                 Path("train_dump.pkl"), training_batches)
    write_pickle(params['dst_folder'] /
                 Path("test_dump.pkl"), testing_batches)
    write_pickle(params['dst_folder'] /
                 Path("counts_dump.pkl"), counts)

    # Weight on the KL loss
    logging.info(f'KL reweight = {params["kl_reweight"]}')

    # In the case that we are loading a model, resume epoch will not be zero
    if params['resume_epoch'] == 0:
        # Initialize meta-parameters
        # Theta is capital theta in the PLATIPUS paper, it holds everything we need
        # 'mean' is the mean model parameters
        # 'logSigma' and 'logSigma_q' are the variance of the base and variational distributions
        # the two 'gamma' vectors are the learning rate vectors
        #
        pass
    else:
        # Cross validation will load a bunch of models elsewhere when testing
        # A little confusing, but if we are training we want to initialize the first model
        if not params['cross_validate'] or params['train_flag']:
            Theta = initialzie_theta_platipus(params)
            # Here we are loading a previously trained model
            print('Restore previous Theta...')
            print('Resume epoch {0:d}'.format(params['resume_epoch']))
            checkpoint_filename = ('{0:s}_{1:d}way_{2:d}shot_{3:d}.pt') \
                .format(params['datasource'],
                        params['n_way'],
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
        # Now we need to set up the optimizer for Theta, PyTorch makes this very easy for us, phew.
        #

        if params['resume_epoch'] > 0:
            op_Theta = set_optim_platipus(Theta, params["meta_lr"])
            op_Theta.load_state_dict(saved_checkpoint['op_Theta'])
            # Set the meta learning rates appropriately
            op_Theta.param_groups[0]['lr'] = params['meta_lr']
            op_Theta.param_groups[1]['lr'] = params['meta_lr']

            params['op_Theta'] = op_Theta

    return params
