import torch
import numpy as np
import logging
import sys
import os
from pathlib import Path
from collections import defaultdict
from utils import (initialise_dict_of_dict, save_model,
                   update_cv_stats_dict, read_pickle, write_pickle)
from hpc_scripts.hpc_params import (common_params, local_meta_params,
                                    local_meta_train, local_meta_test)
from pathlib import Path
#from models.meta import main as platipus
from models.meta.init_params import init_params
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score)


class Platipus:
    def __init__(self, params, amine=None,
                 model_name='Platipus',
                 model_folder='./results',
                 ):
        self.params = params
        self.amine = amine
        for key in self.params:
            setattr(self, key, self.params[key])

        # self.set_optim()

        if amine:
            self.training_batches = params['training_batches'][amine]
            self.dst_folder = save_model(self.model_name, params, amine)

        self.initialize_loss_function()

    def initialize_loss_function(self):
        success = self.counts[self.amine][1]
        failures = self.counts[self.amine][0]

        if failures >= success:
            self.weights = [failures/failures, failures/success]
        else:
            self.weights = [success/failures, success/success]
        logging.debug(f'Weights for loss function: {self.weights}')

        class_weights = torch.tensor(self.weights, device=self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(class_weights)

    def initialize_Theta(self):
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
        for key in self.w_shape.keys():
            if 'b' in key:
                Theta['mean'][key] = torch.zeros(self.w_shape[key],
                                                 device=self.device,
                                                 requires_grad=True)
            else:
                Theta['mean'][key] = torch.empty(self.w_shape[key],
                                                 device=self.device)
                # Could also opt for Kaiming Normal here
                torch.nn.init.xavier_normal_(
                    tensor=Theta['mean'][key], gain=1.)
                Theta['mean'][key].requires_grad_()

            # Subtract 4 to get us into appropriate range for log variances
            Theta['logSigma'][key] = torch.rand(self.w_shape[key],
                                                device=self.device) - 4
            Theta['logSigma'][key].requires_grad_()

            Theta['logSigma_q'][key] = torch.rand(self.w_shape[key],
                                                  device=self.device) - 4
            Theta['logSigma_q'][key].requires_grad_()

            Theta['gamma_q'][key] = torch.tensor(1e-2,
                                                 device=self.device,
                                                 requires_grad=True)
            Theta['gamma_q'][key].requires_grad_()
            Theta['gamma_p'][key] = torch.tensor(1e-2,
                                                 device=self.device,
                                                 requires_grad=True)
            Theta['gamma_p'][key].requires_grad_()
        return Theta

    def set_optim(self):
        keys = ['mean', 'logSigma', 'logSigma_q', 'gamma_p', 'gamma_q']
        parameters = []
        for key in keys:
            parameters.append({'params': self.Theta[key].values()})
        self.op_Theta = torch.optim.Adam(parameters, lr=self.meta_lr)

    def meta_train(self):
        # for epoch in range(resume_epoch, resume_epoch + num_epochs):
        for epoch in range(self.num_epochs):
            logging.debug(f"Starting epoch {epoch}")

            b_num = np.random.choice(len(self.training_batches))
            batch = self.training_batches[b_num]

            x_train, y_train, x_val, y_val = \
                torch.from_numpy(batch[0]).float().to(self.device),\
                torch.from_numpy(batch[1]).long().to(self.device),\
                torch.from_numpy(batch[2]).float().to(self.device),\
                torch.from_numpy(batch[3]).long().to(self.device)

            # variables used to store information of each epoch for monitoring purpose
            meta_loss_saved = []  # meta loss to save
            kl_loss_saved = []
            val_accuracies = []
            train_accuracies = []

            meta_loss = 0  # accumulate the loss of many ensambling networks to descent gradient for meta update
            num_meta_updates_count = 0
            num_meta_updates_print = 1

            meta_loss_avg_print = 0  # compute loss average to print

            kl_loss = 0
            kl_loss_avg_print = 0

            meta_loss_avg_save = []  # meta loss to save
            kl_loss_avg_save = []

            task_count = 0  # a counter to decide when a minibatch of task is completed to perform meta update

            # Treat batches as tasks for the chemistry data
            while task_count < self.num_tasks_per_epoch:

                x_t, y_t, x_v, y_v = x_train[task_count],\
                    y_train[task_count],\
                    x_val[task_count],\
                    y_val[task_count]

                loss_i, KL_q_p = self.get_training_loss(x_t, y_t, x_v, y_v)
                KL_q_p = KL_q_p * self.kl_reweight

                if torch.isnan(loss_i).item():
                    sys.exit('NaN error')

                # accumulate meta loss
                meta_loss = meta_loss + loss_i + KL_q_p
                kl_loss = kl_loss + KL_q_p

                task_count = task_count + 1

                if task_count % self.num_tasks_per_epoch == 0:
                    meta_loss = meta_loss / self.num_tasks_per_epoch
                    kl_loss /= self.num_tasks_per_epoch

                    # accumulate into different variables for printing purpose
                    meta_loss_avg_print += meta_loss.item()
                    kl_loss_avg_print += kl_loss.item()

                    self.op_Theta.zero_grad()
                    meta_loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.Theta['mean'].values(),
                                                   max_norm=3)
                    torch.nn.utils.clip_grad_norm_(parameters=self.Theta['logSigma'].values(),
                                                   max_norm=3)
                    self.op_Theta.step()

                    # Printing losses
                    num_meta_updates_count += 1
                    if (num_meta_updates_count % num_meta_updates_print == 0):
                        meta_loss_avg_save.append(
                            meta_loss_avg_print / num_meta_updates_count)
                        kl_loss_avg_save.append(
                            kl_loss_avg_print / num_meta_updates_count)
                        logging.debug('{0:d}, {1:2.4f}, {2:2.4f}'.format(
                            task_count,
                            meta_loss_avg_save[-1],
                            kl_loss_avg_save[-1]
                        ))

                        num_meta_updates_count = 0
                        meta_loss_avg_print = 0
                        kl_loss_avg_print = 0

                    if (task_count % self.num_tasks_save_loss == 0):
                        meta_loss_saved.append(np.mean(meta_loss_avg_save))
                        kl_loss_saved.append(np.mean(kl_loss_avg_save))

                        meta_loss_avg_save = []
                        kl_loss_avg_save = []

                    # reset meta loss
                    meta_loss = 0
                    kl_loss = 0

            if ((epoch + 1) % self.num_epochs_save == 0):
                checkpoint = {
                    'Theta': self.Theta,
                    'kl_loss': kl_loss_saved,
                    'meta_loss': meta_loss_saved,
                    'val_accuracy': val_accuracies,
                    'train_accuracy': train_accuracies,
                    'op_Theta': self.op_Theta.state_dict()
                }
                logging.info('SAVING WEIGHTS...')

                checkpoint_filename = Path(
                    f"{self.datasource}_{self.k_shot}shot_{epoch+1}.pt")
                logging.info(checkpoint_filename)
                torch.save(checkpoint, self.dst_folder / checkpoint_filename)

        #self.Theta = self.initialize_Theta()

    def get_loss_gradients(self, x, y, w, lr, logSigma=None):
        """
            Calculates gradient loss. If theta=True then variational distribution is updated
        """
        if logSigma is not None:
            q = initialise_dict_of_dict(key_list=w.keys())
            y_pred = self.net.forward(x=x, w=w,
                                      p_dropout=self.p_dropout_base)
            loss = self.loss_fn(y_pred, y)
            loss_grads = torch.autograd.grad(outputs=loss,
                                             inputs=w.values(),
                                             create_graph=True)
            loss_gradients = dict(zip(w.keys(), loss_grads))
            # Assume w is self.Theta
            for key in w.keys():
                q['mean'][key] = w[key] - lr[key] * loss_gradients[key]
                q['logSigma'][key] = logSigma[key]
        else:
            y_pred = self.net.forward(x=x, w=w)
            loss = self.loss_fn(y_pred, y)
            loss_grads = torch.autograd.grad(outputs=loss,
                                             inputs=w.values())
            loss_gradients = dict(zip(w.keys(), loss_grads))
            q = {}
            for key in w.keys():
                q[key] = w[key] - lr * loss_gradients[key]

        return q

    def get_training_loss(self, x_t, y_t, x_v, y_v):
        phi = []
        # step 6 - Compute loss on query set
        # step 7 - Update parameters of the variational distribution
        q = self.get_loss_gradients(x_v, y_v, self.Theta['mean'],
                                    self.Theta['gamma_q'],
                                    logSigma=self.Theta['logSigma_q'])

        # step 8 - Update L sampled models on the training data x_t using gradient descient
        for _ in range(self.Lt):
            # Generate a set of weights using the meta_parameters, equivalent to sampling a model
            w = self.generate_weights(self.Theta)
            # step 9 - Compute updated parameters phi_i using the gradients
            phi_i = self.get_loss_gradients(x_t, y_t, w, self.inner_lr)
            phi.append(phi_i)

        # Repeat step 9 so we do num_inner_updates number of gradient descent steps
        for _ in range(self.num_inner_updates - 1):
            for phi_i in phi:
                phi_i = self.get_loss_gradients(x_t, y_t, w, self.inner_lr)

        # step 10 - Set up probability distribution given the training data
        p = self.get_loss_gradients(x_t, y_t, self.Theta['mean'],
                                    self.Theta['gamma_p'],
                                    logSigma=self.Theta['logSigma'])

        # step 11 - Compute Meta Objective and KL loss
        loss_query = 0
        # Note: We can have multiple models here by adjusting the --Lt flag, but it will
        # drastically slow down the training (linear scaling)
        for phi_i in phi:
            y_pred_query = self.net.forward(x=x_v, w=phi_i)
            loss_query += self.loss_fn(y_pred_query, y_v)
        loss_query /= self.Lt
        KL_q_p = 0
        for key in q['mean'].keys():
            # I am so glad somebody has a formula for this... You rock Cuong
            KL_q_p += torch.sum(torch.exp(2 * (q['logSigma'][key] - p['logSigma'][key]))
                                + (p['mean'][key] - q['mean'][key]) ** 2 / torch.exp(2 * p['logSigma'][key])) \
                + torch.sum(2 * (p['logSigma'][key] - q['logSigma'][key]))
        KL_q_p = (KL_q_p - self.num_weights) / 2
        return loss_query, KL_q_p

    def generate_weights(self, meta_params):
        w = {}
        for key in meta_params['mean'].keys():
            eps_sampled = torch.randn(meta_params['mean'][key].shape,
                                      device=self.device)
            # Use the epsilon reparameterization trick in VI
            w[key] = meta_params['mean'][key] + eps_sampled * \
                torch.exp(meta_params['logSigma'][key])
        return w

    def validate(self):
        logging.info(f'Starting validation for {self.amine}')
        # Run forward pass on the validation data
        logging.debug(f'Weights for loss function: {self.weights}')
        class_weights = torch.tensor(self.weights, device=self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(class_weights)

        val_batch = self.validation_batches[self.amine]
        x_t, y_t, x_v, y_v = torch.from_numpy(val_batch[0]).float().to(self.device),\
            torch.from_numpy(val_batch[1]).long().to(self.device),\
            torch.from_numpy(val_batch[2]).float().to(self.device),\
            torch.from_numpy(val_batch[3]).long().to(self.device)

        accuracies = []
        corrects = []
        probability_pred = []
        preds = self.predict(x_t, y_t, x_v)
        #logging.debug(f'Raw task predictions: {preds}')
        y_pred_v = self.sm_loss(torch.stack(preds))
        #logging.debug(f'after applying softmax: {y_pred_v}')
        y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)
        #logging.debug(f'after calling torch mean: {y_pred}')

        prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
        #logging.debug(f'Labels predicted: {labels_pred}')
        #logging.debug(f'True labels: {y_v}')
        logging.debug(f'probability of prediction: {prob_pred}')
        correct = (labels_pred == y_v)
        corrects.extend(correct.detach().cpu().numpy())

        # print('length of validation set', len(y_v))
        accuracy = torch.sum(correct, dim=0).item() / len(y_v)
        accuracies.append(accuracy)

        probability_pred.extend(prob_pred.detach().cpu().numpy())

        #print('accuracy for model is', accuracies)
        #logging.debug(f'Prediction prob: {probability_pred}')
        #print('probabilities for predictions are', probability_pred)
        self.test_model_actively()

        # Save this dictionary in case we need it later
        write_pickle(self.dst_folder / Path('cv_statistics.pkl'),
                     self.cv_statistics)

    def predict(self, x_t, y_t, x_v):
        # step 1 - Set up the prior distribution over weights (given the training data)
        p = self.get_loss_gradients(x_t, y_t, self.Theta['mean'],
                                    self.Theta['gamma_p'],
                                    logSigma=self.Theta['logSigma'])

        # step 2 - Sample K models and update determine the gradient of the loss function
        phi = []
        for _ in range(self.Lv):
            w = self.generate_weights(p)
            phi_i = self.get_loss_gradients(x_t, y_t, w, self.pred_lr)
            phi.append(phi_i)

        # Repeat step 3 as many times as specified
        for _ in range(self.num_inner_updates - 1):
            for phi_i in phi:
                phi_i = self.get_loss_gradients(x_t, y_t, phi_i, self.pred_lr)

        y_pred_v = []
        # Now get the model predictions on the validation/test data x_v by calling the forward method
        for phi_i in phi:
            w = self.generate_weights(p)
            y_pred_t = self.net.forward(x=x_t, w=w)
            y_pred_temp = self.net.forward(x=x_v, w=phi_i)
            y_pred_v.append(y_pred_temp)
        return y_pred_v

    def test_model_actively(self):
        # Create the stats dictionary to store performance metrics
        cv_stats_dict = {self.model_name: defaultdict(list)}

        val_batch = self.validation_batches[self.amine]

        # Initialize the training and the active learning pool for model
        x_t, y_t, x_v, y_v = torch.from_numpy(val_batch[0]).float().to(self.device),\
            torch.from_numpy(val_batch[1]).long().to(self.device),\
            torch.from_numpy(val_batch[2]).float().to(self.device),\
            torch.from_numpy(val_batch[3]).long().to(self.device)
        all_labels = torch.cat((y_t, y_v))
        all_data = torch.cat((x_t, x_v))

        # Pre-fill num_examples for zero-point evaluation
        num_examples = [0]

        # Set up the number of active learning iterations
        # Starting from 1 so that we can compare PLATIPUS/MAML with other models such as SVM and KNN that
        # have valid results from the first validation point.
        # For testing, overwrite with iters = 1

        # Hard coded to 40 or length of training+validation set whichever is lower
        iters = min([40, len(x_t) + len(x_v) - 1])

        # Randomly pick a point to start active learning with
        rand_index = np.random.choice(len(x_t) + len(x_v) - 1)

        """
        data_dict = read_pickle('./results/non_meta_data.pkl')
        pretrain_data_x = data_dict['test']['k_x'][amine]
        k_x = torch.from_numpy(pretrain_data_x).float().to(device)

        pretrain_data_y = data_dict['test']['k_y'][amine]
        k_y = torch.from_numpy(pretrain_data_y).long().to(device)

        logging.info(
            f'Pretraining model with {len(pretrain_data_x)} points')

        preds = get_task_prediction_platipus(k_x, k_y, all_data, params)
        """

        # Zero point prediction for PLATIPUS model
        logging.info(
            'Getting the PLATIPUS model baseline before training on zero points')
        preds = self.get_naive_prediction(all_data)

        # Evaluate zero point performance for PLATIPUS
        prob_pred, correct, cm, accuracy,\
            precision, recall, bcr = self.zero_point_prediction(preds,
                                                                self.sm_loss,
                                                                all_labels)

        # Display and update individual performance metric
        cv_stats_dict = update_cv_stats_dict(cv_stats_dict, self.model_name,
                                             correct, cm, accuracy,
                                             precision, recall,
                                             bcr, prob_pred=prob_pred,
                                             verbose=self.verbose)

        # Reset the training set and validation set
        x_t, x_v = all_data[rand_index].view(-1, 51),\
            torch.cat([all_data[0:rand_index],
                       all_data[rand_index + 1:]])
        y_t, y_v = all_labels[rand_index].view(1),\
            torch.cat([all_labels[0:rand_index],
                       all_labels[rand_index + 1:]])

        for _ in range(iters):
            #print(f'Doing active learning with {len(x_t)} examples')
            logging.debug(f'Doing active learning with {len(x_t)} examples')
            num_examples.append(len(x_t))
            preds = self.predict(x_t, y_t, all_data)

            # Update available datapoints in the pool and evaluate current model performance
            x_t, y_t, x_v, y_v, prob_pred, \
                correct, cm, accuracy, precision, \
                recall, bcr = self.active_learning(preds,
                                                   all_labels,
                                                   x_t, y_t, x_v, y_v)

            # Display and update individual performance metric
            cv_stats_dict = update_cv_stats_dict(cv_stats_dict, self.model_name,
                                                 correct, cm, accuracy,
                                                 precision, recall, bcr,
                                                 prob_pred=prob_pred,
                                                 verbose=self.verbose)
        logging.info(f'BCR values: {bcr}')
        self.cv_statistics = cv_stats_dict

    def zero_point_prediction(self, preds, sm_loss, all_labels):
        y_pred_v = sm_loss(torch.stack(preds))
        y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)
        prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
        correct = (labels_pred == all_labels)
        # Generate confusion matrix using actual labels and predicted labels
        y_true = all_labels.detach().cpu().numpy()
        y_pred = labels_pred.detach().cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        bcr = balanced_accuracy_score(y_true, y_pred)

        return prob_pred, correct, cm, accuracy, precision, recall, bcr

    def get_naive_prediction(self, x_vals):
        y_pred_v = []
        # Now get the model predictions on the validation/test data x_v by calling the forward method
        for _ in range(self.Lv):
            # Generate some random weights
            w = self.generate_weights(meta_params=self.Theta)
            y_pred_temp = self.net.forward(x=x_vals, w=w)
            y_pred_v.append(y_pred_temp)
        return y_pred_v

    def active_learning(self, preds, all_labels, x_t, y_t, x_v, y_v):

        y_pred_v = self.sm_loss(torch.stack(preds))
        y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)

        prob_pred, labels_pred = torch.max(input=y_pred, dim=1)
        correct = (labels_pred == all_labels)

        y_true = all_labels.detach().cpu().numpy()
        y_pred = labels_pred.detach().cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        bcr = balanced_accuracy_score(y_true, y_pred)

        # Now add the most uncertain point to the training data
        preds_update = self.predict(x_t, y_t, x_v)

        y_pred_v_update = self.sm_loss(torch.stack(preds_update))
        y_pred_update = torch.mean(input=y_pred_v_update, dim=0, keepdim=False)

        prob_pred_update, labels_pred_update = torch.max(input=y_pred_update,
                                                         dim=1)

        logging.debug(y_v)
        logging.debug(labels_pred_update)
        logging.debug(len(prob_pred_update))

        value, index = prob_pred_update.min(0)
        logging.debug(f'Minimum confidence {value}')
        # Add to the training data
        x_t = torch.cat((x_t, x_v[index].view(1, 51)))
        y_t = torch.cat((y_t, y_v[index].view(1)))
        # Remove from pool, there is probably a less clunky way to do this
        x_v = torch.cat([x_v[0:index], x_v[index + 1:]])
        y_v = torch.cat([y_v[0:index], y_v[index + 1:]])
        logging.debug(f'length of x_v is now {len(x_v)}')

        return x_t, y_t, x_v, y_v, prob_pred, correct, cm, accuracy, precision, recall, bcr


if __name__ == '__main__':
    params = {**common_params, **local_meta_params}

    train_params = {**params, **local_meta_train}
    # train_params = platipus.initialize(
    #    [train_params['model_name']], train_params)
    logging.basicConfig(filename=Path(save_model(params['model_name'], params))/Path('logfile.log'),
                        level=logging.DEBUG)
    train_params = init_params(train_params)

    for amine in train_params['training_batches']:
        platipus = Platipus(train_params, amine=amine,
                            model_name=train_params['model_name'])
        platipus.meta_train()
        platipus.validate()

    #test_params = {**params, **local_meta_test}

    # platipus.validate()
