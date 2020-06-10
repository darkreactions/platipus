import torch

from utils import *

from sklearn.metrics import confusion_matrix


def initialzie_theta(params):
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


def initialize_optimization_for_theta(Theta,meta_lr):
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


def reinitialize_model_params(params):
    """Reinitialize model meta-parameters for cross validation in PLATIPUS

    Args:
        params: A dictionary of parameters used for this model. See documentation in initialize() for details.

    Returns:
        N/A
    """
    Theta = initialzie_theta(params)
    params['Theta'] = Theta

    params['op_Theta'] = initialize_optimization_for_theta(Theta, params["meta_lr"])


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
