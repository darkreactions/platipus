from numpy.lib.arraysetops import isin
import torch
import numpy as np
import logging
import sys
from pathlib import Path
from collections import defaultdict

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler

from utils import (
    initialise_dict_of_dict,
    save_model,
    update_cv_stats_dict,
    write_pickle,
)
from hpc_scripts.hpc_params import common_params, local_meta_params, local_meta_train
from models.meta.init_params import init_params
from models.meta.FC_net import FCNet

from utils.dataset_class import *
import pickle


class Platipus:
    def __init__(
        self,
        params,
        amine=None,
        model_name="Platipus",
        model_folder="./results",
        training=True,
        epoch_al=False,
        set_id=0,
    ):
        
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=Path(f'./results/{params["model_name"]}_10_shot/testing')/Path('logfile.log'),
                            level=logging.DEBUG)
        
        # Number of features/inputs to NN
        self.ip_dim = 50
        
        self.set_id = set_id
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
            f"cuda:{self.gpu_id}"
            if (torch.cuda.is_available() and self.gpu_id is not None)
            else "cpu"
        )

        if self.device.type == "cuda":
            logging.info(f"Using device: {torch.cuda.get_device_name(self.device)}")
            logging.info("Memory Usage:")
            logging.info(
                f"Allocated:{round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB "
            )
            logging.info(
                f"Cached: {round(torch.cuda.memory_cached(0) / 1024 ** 3, 1)} GB"
            )

        if isinstance(amine, str) and training:
            self.training_batches = params["training_batches"][amine]
            self.dst_folder = save_model(self.model_name, params, amine)
            self.initialize_loss_function()
        elif isinstance(amine, list) and training:
            self.training_batches = params["training_batches"]
            self.dst_folder = save_model(self.model_name, params, amine)
            self.initialize_loss_function()

        self.net = FCNet(
            dim_input=self.ip_dim,
            dim_output=self.n_way,
            num_hidden_units=self.num_hidden_units,
            device=self.device,
            activation_fn=self.activation_fn,
        )
        self.w_shape = self.net.get_weight_shape()
        self.num_weights = self.net.get_num_weights()
        self.sm_loss = torch.nn.Softmax(dim=2)

        self.Theta = self.initialize_Theta()
        self.set_optim()
        self.model_name_temp = self.model_name

    # Initialization functions <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def initialize_loss_function(self):
        """
        success = self.counts[self.amine][1]
        failures = self.counts[self.amine][0]

        if failures >= success:
            self.weights = [failures / failures, failures / success]
        else:
            self.weights = [success / failures, success / success]
        """
        self.weights = [1.0, 8.0]
        logging.debug(f"Weights for loss function: {self.weights}")

        class_weights = torch.tensor(self.weights, device=self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(class_weights)

    def initialize_Theta(self):
        """This function is to initialize Theta

        Args:
            params: A dictionary of initialized parameters

        return: A dictionary of initialized meta parameters used for cross 
                validation
        """
        Theta = {}
        Theta["mean"] = {}
        Theta["logSigma"] = {}
        Theta["logSigma_q"] = {}
        Theta["gamma_q"] = {}
        Theta["gamma_p"] = {}
        for key in self.w_shape.keys():
            if "b" in key:
                Theta["mean"][key] = torch.zeros(
                    self.w_shape[key], device=self.device, requires_grad=True
                )
            else:
                Theta["mean"][key] = torch.empty(self.w_shape[key], device=self.device)
                # Could also opt for Kaiming Normal here
                torch.nn.init.xavier_normal_(tensor=Theta["mean"][key], gain=1.0)
                Theta["mean"][key].requires_grad_()

            # Subtract 4 to get us into appropriate range for log variances
            Theta["logSigma"][key] = (
                torch.rand(self.w_shape[key], device=self.device) - 4
            )
            Theta["logSigma"][key].requires_grad_()

            Theta["logSigma_q"][key] = (
                torch.rand(self.w_shape[key], device=self.device) - 4
            )
            Theta["logSigma_q"][key].requires_grad_()

            Theta["gamma_q"][key] = torch.tensor(
                1e-2, device=self.device, requires_grad=True
            )
            Theta["gamma_q"][key].requires_grad_()
            Theta["gamma_p"][key] = torch.tensor(
                1e-2, device=self.device, requires_grad=True
            )
            Theta["gamma_p"][key].requires_grad_()
        return Theta

    def set_optim(self):
        keys = ["mean", "logSigma", "logSigma_q", "gamma_p", "gamma_q"]
        parameters = []
        for key in keys:
            parameters.append({"params": self.Theta[key].values()})
        if not self.optimizer_fn:
            self.op_Theta = torch.optim.Adam(parameters, lr=self.meta_lr)
        else:
            self.op_Theta = self.optimizer_fn(parameters, lr=self.meta_lr)

    # Training Functions <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def meta_train(self):
        # for epoch in range(resume_epoch, resume_epoch + num_epochs):
        for epoch in range(self.num_epochs):
            # logging.debug(f"Starting epoch {epoch}")

            b_num = np.random.choice(len(self.training_batches))
            batch = self.training_batches[b_num]

            x_train, y_train, x_val, y_val = (
                torch.from_numpy(batch[0]).float().to(self.device),
                torch.from_numpy(batch[1]).long().to(self.device),
                torch.from_numpy(batch[2]).float().to(self.device),
                torch.from_numpy(batch[3]).long().to(self.device),
            )

            # variables used to store information of each epoch for monitoring
            # purpose
            meta_loss_saved = []  # meta loss to save
            kl_loss_saved = []
            val_accuracies = []
            train_accuracies = []

            meta_loss = 0  # accumulate the loss of many ensambling networks
            # to descent gradient for meta update
            num_meta_updates_count = 0
            num_meta_updates_print = 1

            meta_loss_avg_print = 0  # compute loss average to print

            kl_loss = 0
            kl_loss_avg_print = 0

            meta_loss_avg_save = []  # meta loss to save
            kl_loss_avg_save = []

            task_count = 0  # a counter to decide when a minibatch of task is
            # completed to perform meta update

            # Treat batches as tasks for the chemistry data
            while task_count < self.num_tasks_per_epoch:

                x_t, y_t, x_v, y_v = (
                    x_train[task_count],
                    y_train[task_count],
                    x_val[task_count],
                    y_val[task_count],
                )

                loss_i, KL_q_p = self.get_training_loss(x_t, y_t, x_v, y_v)
                KL_q_p = KL_q_p * self.kl_reweight

                if torch.isnan(loss_i).item():
                    sys.exit("NaN error")

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
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.Theta["mean"].values(), max_norm=3
                    )
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.Theta["logSigma"].values(), max_norm=3
                    )
                    self.op_Theta.step()

                    # Printing losses
                    num_meta_updates_count += 1
                    if num_meta_updates_count % num_meta_updates_print == 0:
                        meta_loss_avg_save.append(
                            meta_loss_avg_print / num_meta_updates_count
                        )
                        kl_loss_avg_save.append(
                            kl_loss_avg_print / num_meta_updates_count
                        )
                        # logging.debug('{0:d}, {1:2.4f}, {2:2.4f}'.format(
                        #    task_count,
                        #    meta_loss_avg_save[-1],
                        #    kl_loss_avg_save[-1]
                        # ))

                        num_meta_updates_count = 0
                        meta_loss_avg_print = 0
                        kl_loss_avg_print = 0

                    if task_count % self.num_tasks_save_loss == 0:
                        meta_loss_saved.append(np.mean(meta_loss_avg_save))
                        kl_loss_saved.append(np.mean(kl_loss_avg_save))

                        meta_loss_avg_save = []
                        kl_loss_avg_save = []

                    # reset meta loss
                    meta_loss = 0
                    kl_loss = 0

            if (epoch + 1) % self.num_epochs_save == 0:
                checkpoint = {
                    "Theta": self.Theta,
                    "kl_loss": kl_loss_saved,
                    "meta_loss": meta_loss_saved,
                    "val_accuracy": val_accuracies,
                    "train_accuracy": train_accuracies,
                    "op_Theta": self.op_Theta.state_dict(),
                }
                logging.info("SAVING WEIGHTS...")

                checkpoint_filename = Path(
                    f"{self.datasource}_{self.k_shot}shot_{epoch+1}.pt"
                )
                logging.info(checkpoint_filename)
                torch.save(checkpoint, self.dst_folder / checkpoint_filename)
                if self.epoch_al:
                    self.current_epoch = epoch + 1
                    self.model_name_temp = f"{self.model_name}_{epoch+1}"
                    self.test_model_actively()

        # self.Theta = self.initialize_Theta()

    def get_training_loss(self, x_t, y_t, x_v, y_v):
        # step 6 - Compute loss on query set
        # step 7 - Update parameters of the variational distribution
        q = self.get_loss_gradients(
            x_v,
            y_v,
            self.Theta["mean"],
            self.Theta["gamma_q"],
            logSigma=self.Theta["logSigma_q"],
        )

        loss_query = 0
        # step 8 - Update L sampled models on the training data x_t using
        # gradient descient
        for _ in range(self.Lt):
            # Generate a set of weights using the meta_parameters, equivalent
            # to sampling a model
            w = self.generate_weights(self.Theta)
            # step 9 - Compute updated parameters phi_i using the gradients
            phi_i = self.get_loss_gradients(x_t, y_t, w, self.inner_lr)
            # Repeat step 9 so we do num_inner_updates number of gradient
            # descent steps
            for _ in range(self.num_inner_updates - 1):
                phi_i = self.get_loss_gradients(x_t, y_t, phi_i, self.inner_lr)
            y_pred_query = self.net.forward(x=x_v, w=phi_i)
            loss_query += self.loss_fn(y_pred_query, y_v)

        loss_query /= self.Lt

        # step 10 - Set up probability distribution given the training data
        p = self.get_loss_gradients(
            x_t,
            y_t,
            self.Theta["mean"],
            self.Theta["gamma_p"],
            logSigma=self.Theta["logSigma"],
        )

        # step 11 - Compute Meta Objective and KL loss

        # Note: We can have multiple models here by adjusting the --Lt flag,
        # but it will drastically slow down the training (linear scaling)

        KL_q_p = 0
        for key in q["mean"].keys():
            # I am so glad somebody has a formula for this... You rock Cuong
            KL_q_p += torch.sum(
                torch.exp(2 * (q["logSigma"][key] - p["logSigma"][key]))
                + (p["mean"][key] - q["mean"][key]) ** 2
                / torch.exp(2 * p["logSigma"][key])
            ) + torch.sum(2 * (p["logSigma"][key] - q["logSigma"][key]))
        KL_q_p = (KL_q_p - self.num_weights) / 2
        return loss_query, KL_q_p

    # General model related functions <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def get_loss_gradients(self, x, y, w, lr, logSigma=None):
        """
            Calculates gradient loss. If logSigma is given then variational
            distribution is updated
        """
        if logSigma is not None:
            q = initialise_dict_of_dict(key_list=w.keys())
            y_pred = self.net.forward(x=x, w=w, p_dropout=self.p_dropout_base)
            loss = self.loss_fn(y_pred, y)
            loss_grads = torch.autograd.grad(
                outputs=loss, inputs=w.values(), create_graph=True
            )
            loss_gradients = dict(zip(w.keys(), loss_grads))
            # Assume w is self.Theta
            for key in w.keys():
                q["mean"][key] = w[key] - lr[key] * loss_gradients[key]
                q["logSigma"][key] = logSigma[key]
        else:
            y_pred = self.net.forward(x=x, w=w)
            loss = self.loss_fn(y_pred, y)
            loss_grads = torch.autograd.grad(outputs=loss, inputs=w.values())
            loss_gradients = dict(zip(w.keys(), loss_grads))
            q = {}
            for key in w.keys():
                q[key] = w[key] - lr * loss_gradients[key]

        return q

    def generate_weights(self, meta_params):
        w = {}
        for key in meta_params["mean"].keys():
            eps_sampled = torch.randn(
                meta_params["mean"][key].shape, device=self.device
            )
            # Use the epsilon reparameterization trick in VI
            w[key] = meta_params["mean"][key] + eps_sampled * torch.exp(
                meta_params["logSigma"][key]
            )
        return w

    def setup_weight_dist(self, x_t, y_t):
        # step 1 - Set up the prior distribution over weights
        # (given the training data)
        p = self.get_loss_gradients(
            x_t,
            y_t,
            self.Theta["mean"],
            self.Theta["gamma_p"],
            logSigma=self.Theta["logSigma"],
        )
        # Experimental saving new weights p as Theta
        # self.Theta.update(p)

        # step 2 - Sample Lv models and update determine

        phi = []
        for m in range(self.Lv):
            w = self.generate_weights(p)
            # the gradient of the loss function
            phi_i = self.get_loss_gradients(x_t, y_t, w, self.pred_lr)
            # Repeat step 3 as many times as specified
            for _ in range(self.num_inner_updates - 1):
                phi_i = self.get_loss_gradients(x_t, y_t, phi_i, self.pred_lr)

            phi.append(phi_i)
        return phi

    def predict(self, x_v, phi=None, proba=False):
        if isinstance(x_v, np.ndarray):
            x_v = torch.from_numpy(x_v).float().to(self.device)
        y_pred_v = []
        if phi is None:
            phi = []
            for _ in range(self.Lv):
                phi.append(self.generate_weights(self.Theta))

        for phi_i in phi:
            # Now get the model predictions on the validation/test data x_v
            #  by calling the forward method
            y_pred_temp = self.net.forward(x=x_v, w=phi_i)
            y_pred_v.append(y_pred_temp)

        y_pred_v = self.sm_loss(torch.stack(y_pred_v))
        y_pred = torch.mean(input=y_pred_v, dim=0, keepdim=False)

        prob_pred, labels_pred = torch.max(input=y_pred, dim=1)

        del y_pred_v
        del y_pred
        del phi
        del x_v

        if proba:
            return prob_pred, labels_pred
        else:
            del prob_pred
            return labels_pred

    def predict_proba(self, x_v):
        self.predict_proba_calls += 1
        print(
            f"Calling predict_proba {self.predict_proba_calls} with x_v = {x_v.shape}"
        )
        prob_pred, labels_pred = self.predict(x_v, proba=True)
        del labels_pred
        return prob_pred.detach().cpu().numpy()

    def phase3_active_learning(self, x_t, y_t, x_ss, debug=False):
        phi = self.setup_weight_dist(x_t, y_t)
        prob_pred_update, labels_pred_update = self.predict(x_ss, phi=phi, proba=True)
        value, index = prob_pred_update.min(0)
        if debug:
            print(prob_pred_update)
        return value, index

    # Active learning function <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def active_learning(self, all_data, all_labels, x_t, y_t, x_v, y_v):
        """
            Performs single step of the active learning process
        """
        phi = self.setup_weight_dist(x_t, y_t)
        prob_pred, labels_pred = self.predict(all_data, phi=phi, proba=True)
        correct = labels_pred == all_labels

        y_true = all_labels.detach().cpu().numpy()
        y_pred = labels_pred.detach().cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        bcr = balanced_accuracy_score(y_true, y_pred)

        # Now add the most uncertain point to the training data
        prob_pred_update, labels_pred_update = self.predict(x_v, phi=phi, proba=True)

        logging.debug(y_v)
        logging.debug(labels_pred_update)
        logging.debug(prob_pred_update)

        value, index = prob_pred_update.min(0)
        # prob_pred_update_np = prob_pred_update.detach().cpu().numpy()
        # val = np.nanmin(prob_pred_update_np)
        # idx = np.nanargmin(prob_pred_update_np)

        logging.debug(f"Minimum confidence {value}")
        # logging.debug(f'Minimum confidence np : {val} {idx}')
        # Add to the training data

        x_t = torch.cat((x_t, x_v[index].view(1, self.ip_dim)))
        y_t = torch.cat((y_t, y_v[index].view(1)))
        # x_t = x_v[index].view(1, 51)
        # y_t = y_v[index].view(1)

        # Remove from pool, there is probably a less clunky way to do this
        x_v = torch.cat([x_v[0:index], x_v[index + 1 :]])
        y_v = torch.cat([y_v[0:index], y_v[index + 1 :]])
        logging.debug(f"length of x_v is now {len(x_v)}")

        self.cv_statistics = update_cv_stats_dict(
            self.cv_statistics,
            self.model_name_temp,
            correct,
            cm,
            accuracy,
            precision,
            recall,
            bcr,
            prob_pred=prob_pred,
            verbose=self.verbose,
        )

        return x_t, y_t, x_v, y_v

    def setup_active_learning(self, set_id, amine):
        class_weights = torch.tensor(self.weights, device=self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(class_weights)

        # Create the stats dictionary to store performance metrics

        self.model_name_temp = f"{self.model_name}_{self.current_epoch}_set{set_id}_{amine}"

        self.cv_statistics.update({self.model_name_temp: defaultdict(list)})

        """
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
        #self.zero_point_prediction(all_data, all_labels)

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
        """

        # Adding Dataset class here
        # dataset = pickle.load(open("./data/full_frozen_dataset.pkl", "rb"))
        dataset = pickle.load(open("../data/phase3_dataset.pkl", "rb"))
        data = dataset.get_dataset("metaALHk", set_id, "random")[amine]

        # scaler = StandardScaler()
        # scaler.fit(data["x_vk"])

        # data["x_vk"] = scaler.transform(data["x_vk"])

        x_t_test = torch.from_numpy(data["x_vk"]).float().to(self.device)
        y_t_test = torch.from_numpy(data["y_vk"]).long().to(self.device)

        logging.info(f"Keys in data! {data.keys()}")
        # Frozen dataset is a little messed up because of a missing comma
        # ['x_t', 'y_t', 'x_vk', 'y_vk', 'x_vx', 'y_vxx_vrem', 'y_vrem', 'x_v', 'y_v']
        # x_h_meta, y_h_meta, x_vk, y_vk, x_vx, y_vx, x_vrem, y_vrem, x_v, y_v
        # data = dataset.get_dataset("metaALHk", set_id, "random")[self.amine]

        # data["x_vrem"] = scaler.transform(data["x_vrem"])
        # data["x_v"] = scaler.transform(data["x_v"])

        x_v_test = torch.from_numpy(data["x_vrem"]).float().to(self.device)
        y_v_test = torch.from_numpy(data["y_vrem"]).long().to(self.device)

        all_data_test = torch.from_numpy(data["x_v"]).float().to(self.device)
        all_labels_test = torch.from_numpy(data["y_v"]).long().to(self.device)

        logging.info("Lengths of training and validation vectors for test")
        logging.info(f"X = {x_t_test.shape}, {x_v_test.shape}")
        logging.info(f"Y = {y_t_test.shape}, {y_v_test.shape}")
        logging.info(f"Y = {all_data_test.shape}, {all_labels_test.shape}")

        # Hard coded to 40 or length of training+validation set whichever is
        # lower
        iters = min([40, len(x_v_test) - 1])

        return (
            iters,
            all_data_test,
            all_labels_test,
            x_t_test,
            y_t_test,
            x_v_test,
            y_v_test,
        )

    def test_model_actively(self):
        """
            Starts the active learning process
        """
        logging.info(f"Starting validation for {self.amine}")
        # Run forward pass on the validation data
        logging.debug(f"Weights for loss function: {self.weights}")

        if isinstance(self.amine, list):
            for amine in self.amine:
                for set_id in range(5):
                    (iters, all_data, all_labels, x_t, y_t, x_v, y_v,
                    ) = self.setup_active_learning(set_id, amine)

                    for i in range(iters):
                        logging.debug(
                            f"Doing active learning for {amine} with {len(x_t)} examples. Iteration: {i}"
                        )
                        # Update available datapoints in the pool and evaluate current
                        # model performance
                        x_t, y_t, x_v, y_v = self.active_learning(
                            all_data, all_labels, x_t, y_t, x_v, y_v
                        )

                # Save this dictionary in case we need it later
                write_pickle(self.dst_folder / Path(f"cv_statistics_{amine}.pkl"), self.cv_statistics)

        elif isinstance(self.amine, str):
            for set_id in range(5):
                (iters, all_data, all_labels, x_t, y_t, x_v, y_v,
                ) = self.setup_active_learning(set_id, self.amine)

                for i in range(iters):
                    logging.debug(
                        f"Doing active learning with {len(x_t)} example. Iteration: {i}"
                    )
                    # Update available datapoints in the pool and evaluate current
                    # model performance
                    x_t, y_t, x_v, y_v = self.active_learning(
                        all_data, all_labels, x_t, y_t, x_v, y_v
                    )

            # Save this dictionary in case we need it later
            write_pickle(self.dst_folder / Path("cv_statistics.pkl"), self.cv_statistics)

    # Utils <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def load_model(self, checkpoint_path):
        if torch.cuda.is_available() and self.gpu_id is not None:
            saved_checkpoint = torch.load(
                checkpoint_path,
                map_location=lambda storage, loc: storage.cuda(self.gpu_id),
            )
        else:
            saved_checkpoint = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
        self.Theta = saved_checkpoint["Theta"]
        self.op_Theta.load_state_dict(saved_checkpoint["op_Theta"])


if __name__ == "__main__":
    params = {**common_params, **local_meta_params}
    params["test_data"] = True

    train_params = {**params, **local_meta_train}
    # train_params = platipus.initialize(
    #    [train_params['model_name']], train_params)
    logging.basicConfig(
        filename=Path(save_model(params["model_name"], params)) / Path("logfile.log"),
        level=logging.DEBUG,
    )
    train_params = init_params(train_params)
    train_params["activation_fn"] = torch.nn.functional.relu
    train_params["optimizer_fn"] = torch.optim.Adam
    for amine in train_params["training_batches"]:
        logging.info(f"Starting process with amine: {amine}")
        platipus = Platipus(
            train_params,
            amine=amine,
            model_name=train_params["model_name"],
            epoch_al=True,
        )

        logging.info(f"Begin training with amine: {amine}")
        platipus.meta_train()
        # run shap for 1st time
        logging.info(f"Begin active learning with amine: {amine}")
        platipus.test_model_actively()
        # run shap for 2st time
        logging.info(f"Completed active learning with amine: {amine}")

