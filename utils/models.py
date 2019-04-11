"""
Define the fully convolutional neural net (FCNN) model to be trained.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# MODEL DEFINITIONS
# -----------------------------------------------------------------------------

class FCNN(nn.Module):

    def __init__(self,
                 n_channels: int = 512,
                 n_convolutional_layers: int = 12):

        super(FCNN, self).__init__()

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------

        self.n_channels = n_channels
        self.n_convolutional_layers = n_convolutional_layers

        # ---------------------------------------------------------------------
        # Define the model's layers
        # ---------------------------------------------------------------------

        # Define the input layer of the net that maps the number of channels
        # from 2 -> n_channels
        self.input_layer = nn.Conv1d(in_channels=2,
                                     out_channels=self.n_channels,
                                     kernel_size=1)

        # Define the output layer that maps the number of channels from
        # n_output_channels of the last convolutional layer -> 1
        self.output_layer = nn.Conv1d(in_channels=self.n_channels,
                                      out_channels=1,
                                      kernel_size=1)

        # Store convolutional layers in a ModuleList(). This is important to
        # ensure that everything works even when training with multiple GPUs.
        self.convolutional_layers = nn.ModuleList()

        # Create as many convolutional layers as requested
        for i in range(self.n_convolutional_layers):

            # Create a 1D convolutional layer where the dilation increases
            # exponentially with the index of the layer
            conv_layer = nn.Conv1d(in_channels=self.n_channels,
                                   out_channels=self.n_channels,
                                   kernel_size=2,
                                   dilation=2**i)
            
            self.convolutional_layers.append(conv_layer)

        # ---------------------------------------------------------------------
        # Initialize the weights of the model to sensible defaults
        # ---------------------------------------------------------------------

        # Initialize the weight and bias of the in- and output layer
        nn.init.kaiming_normal_(self.input_layer.weight)
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.constant_(self.input_layer.bias, 0.001)
        nn.init.constant_(self.output_layer.bias, 0.001)

        # Initialize all the convolutional layer in between
        for conv_layer in self.convolutional_layers:
            nn.init.kaiming_normal_(conv_layer.weight)
            nn.init.constant_(conv_layer.bias, 0.001)

    # -------------------------------------------------------------------------

    def forward(self, x):

        x = self.input_layer.forward(x)

        for conv_layer in self.convolutional_layers:
            x = conv_layer.forward(x)
            x = torch.relu(x)

        x = self.output_layer.forward(x)
        x = torch.sigmoid(x)

        return x


# -----------------------------------------------------------------------------
# MAIN CODE (= BASIC TESTING ZONE)
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Instantiate the default model
    print('Instantiating model...', end=' ', flush=True)
    model = FCNN()
    print('Done!', flush=True)

    # Compute the number of trainable parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([np.prod(p.size()) for p in parameters])
    print('Number of trainable parameters:', n_parameters, '\n')

    # Create some dummy input
    print('Creating random input...', end=' ', flush=True)
    data = torch.randn((1, 2, 8 * 2048))
    print('Done!', flush=True)
    print('Input shape:', data.shape, '\n')

    # Compute the forward pass through the model
    print('Computing forward pass...', end=' ', flush=True)
    output = model.forward(data)
    print('Done!', flush=True)
    print('Output shape:', output.shape)
