import torch
import numpy as np
import random
import itertools
import pickle
import copy

import os
import sys

from utils import *
from FC_net import FCNet

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
