'''
Class containing the code for a fully connected neural network

Author: Cuong Nguyen
Modified By: Gareth Nicholas
'''

import collections

import numpy as np
import torch


class FCNet(torch.nn.Module):
    """
    Establishing a fully connected neural network
    """

    def __init__(self,
                 dim_input=1,
                 dim_output=1,
                 num_hidden_units=(100, 100, 100),
                 device=torch.device('cpu'),
                 activation_fn=torch.nn.functional.relu
                 ):
        """Initializes the FCNet

        Args:
            dim_input:          An integer. Dimension of the input matrix
            dim_output:         An integer. Dimension of the output matrix
            num_hidden_units:   A tuple. The number of hidden nodes in
                                the three layers defined in the net
            device:             The device we use to run,
                                ideally will not be CPU but it can be
        """
        super(FCNet, self).__init__()
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.num_hidden_units = num_hidden_units
        self.device = device
        self.activation_fn = activation_fn

    def get_weight_shape(self):
        """Get the shape of the weight we use in each layer

        Documenting the shape of the weights inputted and outputted
        from the hidden nodes in each layer.
        The shape of the first layer will later be used
        in the 'initialise weight' function.

        return: Generates an ordered dictionary in the format of:
        {weight_name : (number_of_outputs, number_of_inputs)}
        For the bias we have {weight_name: number_of_outputs}
        """
        weight_shape = collections.OrderedDict()

        weight_shape['w1'] = (self.num_hidden_units[0], self.dim_input)
        weight_shape['b1'] = weight_shape['w1'][0]

        for i in range(len(self.num_hidden_units) - 1):
            weight_shape['w{0:d}'.format(
                i + 2)] = (self.num_hidden_units[i + 1], self.num_hidden_units[i])
            weight_shape['b{0:d}'.format(
                i + 2)] = weight_shape['w{0:d}'.format(i + 2)][0]

        weight_shape['w{0:d}'.format(len(self.num_hidden_units) + 1)] = (
            self.dim_output, self.num_hidden_units[len(self.num_hidden_units) - 1])
        weight_shape['b{0:d}'.format(
            len(self.num_hidden_units) + 1)] = self.dim_output

        return weight_shape

    def initialise_weights(self):
        """Initializing the weights of the neural network

        Generates the weights being inputted into the neural network
        The initialization method can be varied.
        The default is kaiming initialization
        This function will be called by MAML model but not PLATIPUS model
        since PLATIPUS has its own initialization

        return: A dictionary of the same keys as
        the output of "get_weights_shape" function
        but with initialized weights as index for keys with 'w'
        and zeros as indices for keys with 'b', in the format of
        {weight_name: weight_value} for weights and {bias_name: bias_value}
        """
        w = {}
        weight_shape = self.get_weight_shape()
        for key in weight_shape.keys():
            if 'b' in key:
                w[key] = torch.zeros(
                    weight_shape[key], device=self.device, requires_grad=True)
            else:
                w[key] = torch.empty(weight_shape[key], device=self.device)
                # Need to select our initialization for the weights here
                # Here is a resource I found: https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

                # torch.nn.init.xavier_normal_(tensor=w[key], gain=1.)
                torch.nn.init.kaiming_normal_(
                    tensor=w[key], mode='fan_out', nonlinearity='relu')
                # torch.nn.init.normal_(tensor=w[key], mean=0., std=1.)
                w[key].requires_grad_()
        return w

    def forward(self, x, w, p_dropout=0):
        """The forward pass of the model

        A forward pass that applies a linear transformation to the
        incoming data with ReLU activation layers and dropout layers.
        In Pytorch we do not need to worry about the final Softmax

        Args:
            x:          The input tensor that we want to train on.
            w:          The weights for the network. Should be the previously defined
                        dictionary with initialized weights.
            p_dropout:  The probability of an element in the input tensor to be zeroed.
                        The probability can be zero.

        return: A tensor of the same shape with the input tensor
        that has gone through the forward loops
        """
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
                out = self.activation_fn(out)

                if p_dropout > 0:
                    out = torch.nn.functional.dropout(out, p_dropout)
        return out

    def get_num_weights(self):
        """Get the fully connected weight of the net

        return: An integer that is the product of
        all the weights presented in the fully connected layers
        """
        num_weights = 0
        # Note that get_weight_shape() is found in FC_net.py
        weight_shape = self.get_weight_shape()
        for key in weight_shape.keys():
            # Product of number of nodes in this layer and number of nodes in next layer
            # We are fully connected after all
            num_weights += np.prod(weight_shape[key], dtype=np.int32)
        return num_weights
