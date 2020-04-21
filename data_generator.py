'''
Class containing the code to generate sinusoid and line data

Author: Cuong Nguyen
Modified By: Gareth Nicholas
'''


import torch
import numpy as np
import os
import random

# Uniformly sample r_size number of values in the range [low, high]
def SampleUniform(r_size, low, high, device):
    return (low - high) * torch.rand(r_size, device=device) + high

class DataGenerator(object):
    def __init__(self, num_samples, device=torch.device('cpu'), config={}):
        self.num_samples = num_samples

        # The defaults from the original paper if new parameters are not specified in 
        # the config dictionary parameter
        self.input_range = config.get('input_range', [-5., 5.])
        self.amp_range = config.get('amp_range', [.1, 5.0])
        self.phase_range = config.get('phase_range', [0., np.pi])
        self.slope_range = config.get('slope_range', [-3., 3.])
        self.intercept_range = config.get('intercept_range', [-3., 3.])

        self.noise_std = 0.3
        self.device = device
    
    # Generate data from ONE sinusoid, number of points generated is self.num_samples
    def generate_sinusoidal_data(self, noise_flag=False):
        amp = SampleUniform(
            r_size=(1, 1),
            low=self.amp_range[0],
            high=self.amp_range[1],
            device=self.device
        )
        phase = SampleUniform(
            r_size=(1, 1),
            low=self.phase_range[0],
            high=self.phase_range[1],
            device=self.device
        )

        x_inputs = SampleUniform(
            r_size=(self.num_samples, 1),
            low=self.input_range[0],
            high=self.input_range[1],
            device=self.device
        )
        outputs = amp*torch.sin(x_inputs + phase)

        # Optional addition of noise
        if noise_flag:
            output_noises = torch.randn((self.num_samples, 1), device=self.device)
            outputs = outputs + self.noise_std*output_noises

        return x_inputs, outputs, amp, phase
    
    # Generate data from ONE line, number of points generated is self.num_samples
    def generate_line_data(self, noise_flag=False):
        slope = SampleUniform(
            r_size=(1, 1),
            low=self.slope_range[0],
            high=self.slope_range[1],
            device=self.device
        )
        intercept = SampleUniform(
            r_size=(1, 1),
            low=self.intercept_range[0],
            high=self.intercept_range[1],
            device=self.device
        )

        x_inputs = SampleUniform(
            r_size=(self.num_samples, 1),
            low=self.input_range[0],
            high=self.input_range[1],
            device=self.device
        )
        outputs = slope*x_inputs + intercept

        # Optional addition of noise
        if noise_flag:
            output_noises = torch.randn((self.num_samples, 1), device=self.device)
            outputs = outputs + (self.noise_std**2)*output_noises

        return x_inputs, outputs, slope, intercept