import torch
import sys

from utils import *

from sklearn.metrics import confusion_matrix
from core import test_model_actively


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
        params: A dictionary of updated parameters
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
    return params

        
def generate_weights_platipus(meta_params, params):
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
        eps_sampled = torch.randn(
            meta_params['mean'][key].shape, device=device)
        # Use the epsilon reparameterization trick in VI
        w[key] = meta_params['mean'][key] + eps_sampled * \
            torch.exp(meta_params['logSigma'][key])
    return w


def get_naive_prediction_platipus(x_vals, params):
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
        w = generate_weights_platipus(meta_params=Theta, params=params)
        y_pred_temp = net.forward(x=x_vals, w=w)
        y_pred_v.append(y_pred_temp)
    return y_pred_v


def get_task_prediction_platipus(x_t, y_t, x_v, params):
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
    y_pred_train = net.forward(
        x=x_t, w=Theta['mean'], p_dropout=p_dropout_base)
    loss_train = loss_fn(y_pred_train, y_t)
    loss_train_grads = torch.autograd.grad(
        outputs=loss_train,
        inputs=Theta['mean'].values(),
        create_graph=True
    )
    loss_train_gradients = dict(zip(Theta['mean'].keys(), loss_train_grads))
    for key in Theta['mean'].keys():
        p['mean'][key] = Theta['mean'][key] - \
            Theta['gamma_p'][key] * loss_train_gradients[key]
        p['logSigma'][key] = Theta['logSigma'][key]

    # step 2 - Sample K models and update determine the gradient of the loss function
    phi = []
    for _ in range(K):
        w = generate_weights_platipus(meta_params=p, params=params)
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
        w = generate_weights_platipus(meta_params=p, params=params)
        y_pred_t = net.forward(x=x_t, w=w)
        y_pred_temp = net.forward(x=x_v, w=phi_i)
        y_pred_v.append(y_pred_temp)
    return y_pred_v


def get_training_loss_platipus(x_t, y_t, x_v, y_v, params):
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
    y_pred_query = net.forward(
        x=x_v, w=Theta['mean'], p_dropout=p_dropout_base)
    loss_query = loss_fn(y_pred_query, y_v)
    loss_query_grads = torch.autograd.grad(
        outputs=loss_query,
        inputs=Theta['mean'].values(),
        create_graph=True
    )
    loss_query_gradients = dict(zip(Theta['mean'].keys(), loss_query_grads))

    # step 7 - Update parameters of the variational distribution
    for key in Theta['mean'].keys():
        q['mean'][key] = Theta['mean'][key] - \
            Theta['gamma_q'][key] * loss_query_gradients[key]
        q['logSigma'][key] = Theta['logSigma_q'][key]

    # step 8 - Update L sampled models on the training data x_t using gradient descient
    for _ in range(L):
        # Generate a set of weights using the meta_parameters, equivalent to sampling a model
        w = generate_weights_platipus(meta_params=Theta, params=params)
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
    y_pred_train = net.forward(
        x=x_t, w=Theta['mean'], p_dropout=p_dropout_base)
    loss_train = loss_fn(y_pred_train, y_t)
    loss_train_grads = torch.autograd.grad(
        outputs=loss_train,
        inputs=Theta['mean'].values(),
        create_graph=True
    )
    loss_train_gradients = dict(zip(Theta['mean'].keys(), loss_train_grads))
    for key in Theta['mean'].keys():
        p['mean'][key] = Theta['mean'][key] - \
            Theta['gamma_p'][key] * loss_train_gradients[key]
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
        KL_q_p += torch.sum(torch.exp(2 * (q['logSigma'][key] - p['logSigma'][key]))
                            + (p['mean'][key] - q['mean'][key]) ** 2 / torch.exp(2 * p['logSigma'][key])) \
            + torch.sum(2 * (p['logSigma'][key] - q['logSigma'][key]))
    KL_q_p = (KL_q_p - num_weights) / 2
    return loss_query, KL_q_p


def zero_point_platipus(preds, sm_loss, all_labels):
    """Evalute platipus model performance w/o any active learning.

    Args:
        preds:          A list representing the labels predicted given our all our data points in the pool.
        sm_loss:        A Softmax object representing the softmax layer to handle losses.
        all_labels:     A torch.Tensor object representing all the labels of our reactions.

    Returns:
        prob_pred:      A torch.Tensor object representing all the probabilities of the current predictions of all data
                            points w/o active learning.
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

    y_pred_v = sm_loss(torch.stack(preds))
    y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)

    prob_pred, labels_pred = torch.max(input=y_pred, dim=1)

    correct = (labels_pred == all_labels)

    # Evaluate the model zero-point accuracy
    accuracy = torch.sum(correct, dim=0).item() / len(all_labels)

    # Generate confusion matrix using actual labels and predicted labels
    cm = confusion_matrix(all_labels.detach().cpu().numpy(), labels_pred.detach().cpu().numpy())

    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    true_negative = cm[0][0] / (cm[0][0] + cm[0][1])
    bcr = 0.5 * (recall + true_negative)

    return prob_pred, correct, cm, accuracy, precision, recall, bcr


def active_learning_platipus(preds, sm_loss, all_labels, params, x_t, y_t, x_v, y_v):
    """Update active learning pool and evalute platipus model performance.

    Args:
        preds:          A list representing the labels predicted given our all our data points in the pool.
        sm_loss:        A Softmax object representing the softmax layer to handle losses.
        all_labels:     A torch.Tensor object representing all the labels of our reactions.
        params:         A dictionary of parameters used for this model.
                            See documentation in initialize() for details.
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
        prob_pred:      A torch.Tensor object representing all the probabilities of the current predictions of all data
                            points w/o active learning.
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

    y_pred_v = sm_loss(torch.stack(preds))
    y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)

    prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
    correct = (labels_pred == all_labels)

    accuracy = torch.sum(correct, dim=0).item() / len(all_labels)

    # Now add the most uncertain point to the training data
    preds_update = get_task_prediction_platipus(x_t, y_t, x_v, params)

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

    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    true_negative = cm[0][0] / (cm[0][0] + cm[0][1])
    bcr = 0.5 * (recall + true_negative)

    return x_t, y_t, x_v, y_v, prob_pred, correct, cm, accuracy, precision, recall, bcr


def forward_pass_validate_platipus(params, amine):
    """ The forward pass for validation in the PLATIPUS model

    Args:
        params:     A dictionary of parameters used for this model.
                    See documentation in initialize() for details.
        amine:      The amine that we are validating on
    """
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
    # print('print probability of prediction', prob_pred)
    correct = (labels_pred == y_v)
    corrects.extend(correct.detach().cpu().numpy())

    # print('length of validation set', len(y_v))
    accuracy = torch.sum(correct, dim=0).item() / len(y_v)
    accuracies.append(accuracy)

    probability_pred.extend(prob_pred.detach().cpu().numpy())

    print('accuracy for model is', accuracies)
    print('probabilities for predictions are', probability_pred)
