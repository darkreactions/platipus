'''
Class containing the code for a fully connected neural network

Author: Cuong Nguyen
Modified By: Gareth Nicholas
'''

import torch
import numpy as np

import collections

class FCNet(torch.nn.Module):

    # Initialize our neural network, ideally device will not be CPU but it can be
    def __init__(self,
                dim_input=1,
                dim_output=1,
                num_hidden_units=(100, 100, 100),
                device=torch.device('cpu')
    ):

        super(FCNet, self).__init__()
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.num_hidden_units = num_hidden_units
        self.device = device

    # Dictionary is formatted as {weight_name : (number_of_outputs, number_of_inputs)}
    # For the bias we have {weight_name : number_of_outputs}
    def get_weight_shape(self):
        weight_shape = collections.OrderedDict()

        weight_shape['w1'] = (self.num_hidden_units[0], self.dim_input)
        weight_shape['b1'] = weight_shape['w1'][0]

        for i in range(len(self.num_hidden_units) - 1):
            weight_shape['w{0:d}'.format(i + 2)] = (self.num_hidden_units[i + 1], self.num_hidden_units[i])
            weight_shape['b{0:d}'.format(i + 2)] = weight_shape['w{0:d}'.format(i + 2)][0]

        weight_shape['w{0:d}'.format(len(self.num_hidden_units) + 1)] = (self.dim_output, self.num_hidden_units[len(self.num_hidden_units) - 1])
        weight_shape['b{0:d}'.format(len(self.num_hidden_units) + 1)] = self.dim_output

        return weight_shape
    
    # Initialize the weights of our neural network, this is called by maml not PLATIPUS
    # since PLATIPUS has its own fun initialization
    def initialise_weights(self):
        w = {}
        weight_shape = self.get_weight_shape()
        for key in weight_shape.keys():
            if 'b' in key:
                w[key] = torch.zeros(weight_shape[key], device=self.device, requires_grad=True)
            else:
                w[key] = torch.empty(weight_shape[key], device=self.device)
                # Need to select our initialization for the weights here
                # Here is a resource I found: https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

                # torch.nn.init.xavier_normal_(tensor=w[key], gain=1.)
                torch.nn.init.kaiming_normal_(tensor=w[key], mode='fan_out', nonlinearity='relu')
                # torch.nn.init.normal_(tensor=w[key], mean=0., std=1.)
                w[key].requires_grad_()
        return w

    # Run a forward pass of the model, in Pytorch we do not need to worry about the final Softmax 
    def forward(self, x, w, p_dropout=0):
        out = x

        for i in range(len(self.num_hidden_units) + 1):
            out = torch.nn.functional.linear(
                input=out,
                weight=w['w{0:d}'.format(i + 1)],
                bias=w['b{0:d}'.format(i + 1)]
            )

            if (i < len(self.num_hidden_units)):
                # We could change the activation function here if we wanted to! 
                # out = torch.tanh(out)
                out = torch.nn.functional.relu(out)
                if p_dropout > 0:
                    out = torch.nn.functional.dropout(out, p_dropout)
        return out

    # Get the number if weights in the net
    def get_num_weights(self):
        num_weights = 0
        # Note that get_weight_shape() is found in FC_net.py
        weight_shape = self.get_weight_shape()
        for key in weight_shape.keys():
            # Product of number of nodes in this layer and number of nodes in next layer
            # We are fully connected after all
            num_weights += np.prod(weight_shape[key], dtype=np.int32)
        return num_weights
