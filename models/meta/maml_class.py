import torch
import numpy as np
import logging
import sys
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score)
from utils import (initialise_dict_of_dict, save_model,
                   update_cv_stats_dict, write_pickle)


class MAML:
    def __init__(self, params, amine=None,
                 model_name='Platipus',
                 model_folder='./results',
                 training=True,
                 epoch_al=False,
                 ):
        self.epoch_al = epoch_al
        # For SHAP
        self.predict_proba_calls = 0
        self.optimizer_fn = None
        self.rand_index = None
        self.activation_fn = torch.nn.functional.relu
        self.params = params
        self.amine = amine
        self.cv_statistics = {}
        for key in self.params:
            setattr(self, key, self.params[key])

        self.device = torch.device(
            f'cuda:{self.gpu_id}' if (torch.cuda.is_available() and self.gpu_id is not None) else "cpu")

        if self.device.type == 'cuda':
            logging.info(
                f'Using device: {torch.cuda.get_device_name(self.device)}')
            logging.info('Memory Usage:')
            logging.info(
                f'Allocated:{round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB ')
            logging.info(
                f'Cached: {round(torch.cuda.memory_cached(0) / 1024 ** 3, 1)} GB')

        if amine and training:
            self.training_batches = params['training_batches'][amine]
            self.dst_folder = save_model(self.model_name, params, amine)
            self.initialize_loss_function()

        self.net = FCNet(dim_input=51, dim_output=self.n_way,
                         num_hidden_units=self.num_hidden_units,
                         device=self.device, activation_fn=self.activation_fn)
        self.w_shape = self.net.get_weight_shape()
        self.num_weights = self.net.get_num_weights()
        self.sm_loss = torch.nn.Softmax(dim=1)

        self.Theta = self.initialize_Theta()
        self.set_optim()
        self.model_name_temp = self.model_name

    def predict(self, x, y, x_v, y_v=None):
        # Holds the updated parameters
        q = {}

        # Compute loss on the training data
        y_pred_t = net.forward(
            x=x, w=self.theta, p_dropout=self.p_dropout_base)
        loss_NLL = self.loss_fn(y_pred_t, y)

        grads = torch.autograd.grad(
            outputs=loss_NLL,
            inputs=self.theta.values(),
            create_graph=True
        )
        gradients = dict(zip(self.theta.keys(), grads))

        # Obtain the weights for the updated model
        for key in self.theta.keys():
            q[key] = self.theta[key] - self.inner_lr * gradients[key]

        # This code gets run if we want to do more than one gradient update
        for _ in range(self.num_inner_updates - 1):
            loss_NLL = 0
            y_pred_t = net.forward(x=x, w=q, p_dropout=self.p_dropout_base)
            loss_NLL = self.loss_fn(y_pred_t, y)
            grads = torch.autograd.grad(
                outputs=loss_NLL,
                inputs=q.values(),
                retain_graph=True
            )
            gradients = dict(zip(q.keys(), grads))

            for key in q.keys():
                q[key] = q[key] - self.inner_lr * gradients[key]

        # Now predict on the validation or test data
        y_pred_v = net.forward(x=x_v, w=q, p_dropout=0)

        # Then we were operating on testing data, return our predictions
        if y_v is None:
            return y_pred_v
        # We were operating on validation data, return our loss
        else:
            loss_NLL = self.loss_fn(y_pred_v, y_v)
            return loss_NLL

    def meta_train(self):
        # Start by unpacking the variables that we need
        for epoch in range(0, self.num_epochs):
            print(f"Starting epoch {epoch}")

            if self.cross_validate:
                b_num = np.random.choice(
                    len(self.training_batches[self.amine]))
                batch = training_batches[self.amine][b_num]
            else:
                b_num = np.random.choice(len(self.training_batches))
                batch = self.training_batches[b_num]  # TODO: this seems wrong
            x_train, y_train, x_val, y_val = torch.from_numpy(
                batch[0]).float().to(params['device']),
            torch.from_numpy(batch[1]).long().to(params['device']),
            torch.from_numpy(batch[2]).float().to(params['device']),
            torch.from_numpy(batch[3]).long().to(params['device'])

            # variables used to store information of each epoch for monitoring purpose
            meta_loss_saved = []  # meta loss to save
            val_accuracies = []
            train_accuracies = []

            task_count = 0  # a counter to decide when a minibatch of task is completed to perform meta update
            meta_loss = 0  # accumulate the loss of many ensambling networks to descent gradient for meta update
            num_meta_updates_count = 0

            meta_loss_avg_print = 0  # compute loss average to print

            meta_loss_avg_save = []  # meta loss to save
            num_meta_updates_print = 1
            while (task_count < self.num_tasks_per_epoch):
                x_t, y_t, x_v, y_v = x_train[task_count], y_train[task_count], x_val[task_count], y_val[task_count]
                loss_NLL = self.get_task_prediction(x_t, y_t, x_v, y_v)

                if torch.isnan(loss_NLL).item():
                    sys.exit('NaN error')

                # accumulate meta loss
                meta_loss = meta_loss + loss_NLL

                task_count = task_count + 1
                if task_count % self.num_tasks_per_epoch == 0:
                    meta_loss = meta_loss / self.num_tasks_per_epoch

                    # accumulate into different variables for printing purpose
                    meta_loss_avg_print += meta_loss.item()

                    self.op_theta.zero_grad()
                    meta_loss.backward()

                    # Clip gradients to prevent exploding gradient problem
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.theta.values(),
                        max_norm=3
                    )

                    self.op_theta.step()

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

                    if (task_count % self.meta_batch_size == 0):
                        meta_loss_saved.append(np.mean(meta_loss_avg_save))

                        meta_loss_avg_save = []

                    # Reset meta loss
                    meta_loss = 0

                if (task_count >= self.num_tasks_per_epoch):
                    break

            if ((epoch + 1) % self.num_epochs_save == 0):
                checkpoint = {
                    'theta': self.theta,
                    'meta_loss': meta_loss_saved,
                    'val_accuracy': val_accuracies,
                    'train_accuracy': train_accuracies,
                    'op_theta': self.op_theta.state_dict()
                }
                print('SAVING WEIGHTS...')
                checkpoint_filename = Path(
                    f"{self.model_name}_{self.k_shot}shot_{epoch+1}.pt")

                torch.save(checkpoint, self.dst_folder / checkpoint_filename)

    def setup_active_learning(self):
                class_weights = torch.tensor(self.weights, device=self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(class_weights)

        # Create the stats dictionary to store performance metrics
        self.cv_statistics.update({self.model_name_temp: defaultdict(list)})

        val_batch = self.validation_batches[self.amine]

        # Initialize the training and the active learning pool for model
        x_t, y_t, x_v, y_v = torch.from_numpy(val_batch[0]).float().to(self.device),\
            torch.from_numpy(val_batch[1]).long().to(self.device),\
            torch.from_numpy(val_batch[2]).float().to(self.device),\
            torch.from_numpy(val_batch[3]).long().to(self.device)
        all_labels = torch.cat((y_t, y_v))
        all_data = torch.cat((x_t, x_v))

        # Zero point prediction for PLATIPUS model
        logging.info(
            'Getting the PLATIPUS model baseline before training on zero points')

        # Evaluate zero point performance for PLATIPUS
        # self.zero_point_prediction(all_data, all_labels)

        # Randomly pick a k-points to start active learning with
        k_points = 10
        if self.rand_index is None:
            # Creating self.rand_index to save the random k points
            self.rand_index = np.random.choice(
                len(x_t) + len(x_v) - 1, k_points, replace=False)
        val_loc = [
            0 if i in self.rand_index else 1 for i in range(len(all_labels))]
        loc = [1 if i in self.rand_index else 0 for i in range(
            len(all_labels))]

        x_t = all_data[torch.BoolTensor(loc)]
        x_v = all_data[torch.BoolTensor(val_loc)]

        y_t = all_labels[torch.BoolTensor(loc)]
        y_v = all_labels[torch.BoolTensor(val_loc)]

        logging.info('Lenghts of training and validation vectors')
        logging.info(f'X = {x_t.shape}, {x_v.shape}')
        logging.info(f'Y = {y_t.shape}, {y_v.shape}')
        # Hard coded to 40 or length of training+validation set whichever is
        # lower
        iters = min([40, len(x_v) - 1])

        return iters, all_data, all_labels, x_t, y_t, x_v, y_v

    def zero_point_maml(self, preds, all_labels):
        y_pred = self.sm_loss(preds)
        _, labels_pred = torch.max(input=y_pred, dim=1)
        
        correct = (labels_pred == all_labels)
        y_true = all_labels.detach().cpu().numpy()
        y_pred = labels_pred.detach().cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        bcr = balanced_accuracy_score(y_true, y_pred)

        return correct, cm, accuracy, precision, recall, bcr

    def active_learning_maml(self, preds, all_labels, x_t, y_t, x_v, y_v):
        y_pred = self.sm_loss(preds)

        _, labels_pred = torch.max(input=y_pred, dim=1)
        correct = (labels_pred == all_labels)
        accuracy = torch.sum(correct, dim=0).item() / len(all_labels)

        # Now add a random point since MAML cannot reason about uncertainty
        index = np.random.choice(len(x_v))
        # Add to the training data
        x_t = torch.cat((x_t, x_v[index].view(1, 51)))
        y_t = torch.cat((y_t, y_v[index].view(1)))
        # Remove from pool, there is probably a less clunky way to do this
        x_v = torch.cat([x_v[0:index], x_v[index + 1:]])
        y_v = torch.cat([y_v[0:index], y_v[index + 1:]])
        print('length of x_v is now', len(x_v))
        y_true = all_labels.detach().cpu().numpy()
        y_pred = labels_pred.detach().cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        bcr = balanced_accuracy_score(y_true, y_pred)

        return x_t, y_t, x_v, y_v, correct, cm, accuracy, precision, recall, bcr

    def test_model_actively(self):

        iters, all_data, all_labels, x_t, y_t, x_v, y_v = self.setup_active_learning()
        # Zero point prediction for MAML model
        logging.info(
            'Getting the MAML model baseline before training on zero points')
        preds = self.net.forward(x=all_data, w=self.theta)

        # Evaluate zero point performance for MAML
        correct, cm, accuracy, precision, recall, bcr = self.zero_point_maml(
            preds, all_labels)

        # Display and update individual performance metric
        self.cv_statistics = update_cv_stats_dict(self.cv_statistics,
                                                  self.model_name_temp,
                                                  correct, cm, accuracy,
                                                  precision, recall, bcr,
                                                  verbose=self.verbose)

        for i in range(iters):
            logging.debug(
                f'Doing MAML learning with {len(x_t)} examples')
            preds = self.predict(x_t, y_t, all_data)

            # Update available datapoints in the pool and evaluate current model performance
            x_t, y_t, x_v, y_v, correct, \
                cm, accuracy, precision, \
                recall, bcr = \
                active_learning_maml(preds, 
                                        all_labels, x_t, y_t,
                                        x_v, y_v
                                        )

            # Display and update individual performance metric
            self.cv_statistics = update_cv_stats_dict(self.cv_statistics,
                                                      self.model_name_temp,
                                                      correct, cm, accuracy,
                                                      precision, recall, bcr,
                                                      verbose=self.verbose)



if __name__ == '__main__':
    from hpc_scripts.hpc_params import (common_params, local_meta_params,
                                    local_meta_train)
    from models.meta.init_params import init_params
    params = {**common_params, **local_meta_params}
    params['test_data'] = True

    train_params = {**params, **local_meta_train}
    # train_params = platipus.initialize(
    #    [train_params['model_name']], train_params)
    logging.basicConfig(filename=Path(save_model(params['model_name'], params))/Path('logfile.log'),
                        level=logging.DEBUG)
    train_params = init_params(train_params)
    train_params['activation_fn'] = torch.nn.functional.relu
    train_params['optimizer_fn'] = torch.optim.Adam
    for amine in train_params['training_batches']:
        logging.info(f'Starting process with amine: {amine}')
        maml = MAML(train_params, amine=amine,
                    model_name=train_params['model_name'], epoch_al=True)

        logging.info(f'Begin training with amine: {amine}')
        maml.meta_train()
        # run shap for 1st time
        logging.info(f'Begin active learning with amine: {amine}')
        maml.test_model_actively()
        # run shap for 2st time
        logging.info(f'Completed active learning with amine: {amine}')
