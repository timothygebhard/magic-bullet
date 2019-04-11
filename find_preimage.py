"""
For a given pre-trained network, find a pre-image for a given desired
output by optimizing the inputs to the network.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import h5py
import os
import time
import torch
import torch.nn as nn

from pathlib import Path

from utils.models import FCNN
from utils.datasets import InjectionDataset


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_arguments() -> argparse.Namespace:
    """
    Set up an ArgumentParser to get the command line arguments.

    Returns:
        A Namespace object containing all the command line arguments
        for the script.
    """

    # Set up parser
    parser = argparse.ArgumentParser(description='Find pre-image.')

    # Add arguments
    parser.add_argument('--constraint',
                        type=str,
                        metavar='NAME',
                        default='gw_like',
                        help='Type of constraint to enforce on the inputs'
                             'during the optimization. '
                             'Default: gw_like.')
    parser.add_argument('--epochs',
                        type=int,
                        metavar='N',
                        default=256,
                        help='Number of epochs for the optimization. '
                             'Default: 256.')
    parser.add_argument('--index',
                        type=int,
                        metavar='N',
                        default=0,
                        help='Index of the sample in the noise-only part of'
                             'the testing dataset. Default: 0.')

    # Parse and return the arguments (as a Namespace object)
    arguments = parser.parse_args()
    return arguments


def smooth_weights(weights: torch.Tensor,
                   kernel_size: int = 15) -> torch.Tensor:
    """
    Smooth a 1D weights tensor by passing it through a 1D convolutional
    layer with a fixed kernel of a given `kernel_size`.
    
    Args:
        weights: Weights tensor to be smoothed.
        kernel_size: Size of the kernel / rectangular window to be
            used for smoothing.

    Returns:
        A smoothed version of the original `weights`.
    """

    # Ensure window_size is an odd number
    if kernel_size % 2 == 0:
        kernel_size -= 1

    # Ensure weights has the right shape, that is, add a fake
    # batch and channel dimension if necessary
    while len(weights.shape) < 3:
        weights = weights.unsqueeze(0)

    # Define convolutional layer
    layer = nn.Conv1d(in_channels=1,
                      out_channels=1,
                      kernel_size=kernel_size,
                      padding=int(kernel_size / 2))

    # Fix values of the kernel
    nn.init.constant_(layer.weight, 1.0 / kernel_size)
    nn.init.constant_(layer.bias, 0)

    # Apply convolution to weights tensor to smooth it
    return layer.forward(weights)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()

    print('')
    print('FIND PRE-IMAGE THROUGH INPUT OPTIMIZATION')
    print('')

    # -------------------------------------------------------------------------
    # Get command line arguments and define shortcuts
    # -------------------------------------------------------------------------

    # Get arguments and define shortcut
    arguments = get_arguments()
    epochs = int(arguments.epochs)
    index = int(arguments.index)
    constraint = arguments.constraint

    # Fix PyTorch seed for reproducibility
    torch.manual_seed(index)

    # -------------------------------------------------------------------------
    # Create a new instance of the model and load weights from checkpoint file
    # -------------------------------------------------------------------------

    # Device that we will use (CUDA if it is available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a new instance of the model we have previously trained
    model = FCNN()

    # Define the checkpoint we want to load (and ensure it exists)
    checkpoint_file = './checkpoints/best.pth'
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f'{checkpoint_file} does not exist!')

    # Open the checkpoint file and load weights into the model
    print('Loading model checkpoint...', end=' ', flush=True)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Done!', flush=True)

    # Move model to the correct device (CPU or GPU)
    model.to(device)

    # Freeze all layers (we are optimizing the input, not the network!)
    for param in model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # Select noise (as starting input for the optimization) and create target
    # -------------------------------------------------------------------------

    print('Loading input data and creating target...', end=' ', flush=True)

    # Set up the dataset from which we are loading our background noise
    dataset = InjectionDataset(mode='testing', sample_type='noise')

    # Select the noise series according to the index passed to the script,
    # shorten it to only 3 seconds (at 2048 Hz = 6144 time steps) and add a
    # dummy batch dimension. This is the starting point for the optimization.
    noise, _ = dataset[index]
    noise = noise[..., :6144].unsqueeze(0).to(device).requires_grad_(True)

    # Depending on the constraints, we optimize different things: Either the
    # model inputs as a whole, or only the additive "signal" component.
    if constraint in ('gw_like', 'minimal_perturbation'):

        # Initialize a small random "signal" to be added onto the noise, and
        # initialize the input to the network as the noise plus that signal
        signal = torch.randn(noise.shape, requires_grad=True, device=device)
        inputs = noise + signal

    else:

        # Start with the pure, unaltered noise as the input to the model. In
        # this case, there is no explicit "signal" component.
        inputs = noise.clone().detach().to(device).requires_grad_(True)
        signal = None

    # Instantiate target outputs: Length 1 second plus 1 time step, everything
    # is zero except in a 102 / 2048 Hz = 0.05s interval around the center.
    targets = torch.zeros((1, 1, 2049), requires_grad=False, device=device)
    targets[..., 1024-102:1024+102] = 1.0
    targets = targets.float()

    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up an optimizer, a loss function and an LR scheduler
    # -------------------------------------------------------------------------

    # Set up an optimizer that optimizes either the "signal" or the inputs
    if constraint in ('gw_like', 'minimal_perturbation'):
        optimizer = torch.optim.Adam(params=[signal], lr=3e-1, amsgrad=True)
    else:
        optimizer = torch.optim.Adam(params=[inputs], lr=3e-1, amsgrad=True)

    # Instantiate different loss functions (depending on the constraints, we
    # will use a weighted sum of different losses to guide the optimization)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    # Set up a scheduler to reduce the learning rate during training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=epochs,
                                                           eta_min=1e-4)

    # -------------------------------------------------------------------------
    # Optimize the inputs for the given number of epochs
    # -------------------------------------------------------------------------

    # Initialize the variable for the network outputs
    outputs = None

    print('Optimizing inputs to produce desired outputs:', flush=True)
    for epoch in range(epochs):

        # ---------------------------------------------------------------------
        # Enforce possible constraints on the "signal" or the inputs
        # ---------------------------------------------------------------------

        # NOTE: Depending on the type of input for the model we want, we need
        # to guide the optimization in different ways in order to converge to
        # examples of the desired kind. This is valid, because the in the end
        # we only state that there *exist* examples of inputs with particular
        # characteristics; the way we found them, however, should not matter.

        # For "GW-like" signals, we want to make sure the "signal" is located
        # near the target coalescence time, so we suppress signal components
        # too far from this
        if constraint == 'gw_like' and epoch < 0.5 * epochs:
            weights = 0.01 * torch.ones(6144).float()
            weights[3072 - 512:3072 + 512] = 1
            weights = smooth_weights(weights).to(device)
            signal.data = signal.data * weights

        # Try to avoid a signal at the expected coalescence time
        if constraint == 'minimal_perturbation' and epoch < 0.8 * epochs:
            weights = torch.ones(6144).float()
            weights[3072 - 512:3072 + 512] = 0
            weights = smooth_weights(weights).to(device)
            signal.data = signal.data * weights

        # To achieve minimal amplitude, we explicitly force a large part of
        # the inputs to be zero, so that the optimization can focus on the
        # "signal" part, which should be as small as possible.
        if constraint == 'minimal_amplitude':
            inputs.data = torch.clamp(inputs.data, -0.08, 0.08)

        # Set all negative strain values to zero
        if constraint == 'positive_strain':
            inputs.data = torch.relu(inputs.data)

        # Set the region around the "coalescence" to zero
        if constraint == 'zero_coalescence':
            weights = torch.ones(6144).float()
            weights[3072 - 256:3072 + 256] = 0
            weights = smooth_weights(weights, kernel_size=7).to(device)
            inputs.data = inputs.data * weights

        # ---------------------------------------------------------------------
        # Compute the model inputs if we are only optimizing the "signal"
        # ---------------------------------------------------------------------
        
        if constraint in ('gw_like', 'minimal_perturbation'):
            inputs = noise + signal

        # ---------------------------------------------------------------------
        # Do a forward pass through the model and compute the loss
        # ---------------------------------------------------------------------

        # Compute the forward pass through the model
        outputs = model.forward(inputs)

        # Compute the loss depending on the constraint:
        if constraint in ('gw_like', 'positive_strain', 'zero_coalescence'):
            loss = (1.000 * bce_loss(outputs, targets) +
                    0.175 * mse_loss(inputs, noise))

        elif constraint == 'minimal_perturbation':
            loss = (1.000 * bce_loss(outputs, targets) +
                    5.000 * mse_loss(inputs, noise))

        elif constraint == 'minimal_amplitude':
            loss = (1.000 * bce_loss(outputs, targets))

        else:
            raise ValueError('Illegal value for constraint parameter!')

        # ---------------------------------------------------------------------
        # Take a step with the optimizer (this should only modify the signal)
        # ---------------------------------------------------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'Epoch: {epoch+1:3d}/{epochs}\tLoss: {loss.item():.5f}')

    # -------------------------------------------------------------------------
    # Save the result as an HDF file
    # -------------------------------------------------------------------------

    # Construct path at which we will save this file (as a PDF)
    preimages_dir = './results/preimages'
    Path(preimages_dir).mkdir(exist_ok=True, parents=True)
    preimage_path = os.path.join(preimages_dir, f'{constraint}__{index}.hdf')

    # Create the plot and save it as a PDF
    print('\nSaving result as HDF file...', end=' ', flush=True)
    with h5py.File(preimage_path, 'w') as hdf_file:

        # Store command line arguments
        hdf_file.attrs['constraint'] = constraint
        hdf_file.attrs['epochs'] = epochs
        hdf_file.attrs['index'] = index

        # Save the inputs, outputs and targets
        hdf_file.create_dataset(name='noise',
                                data=noise.data.cpu().numpy().squeeze())
        hdf_file.create_dataset(name='inputs',
                                data=inputs.data.cpu().numpy().squeeze())
        hdf_file.create_dataset(name='outputs',
                                data=outputs.data.cpu().numpy().squeeze())
        hdf_file.create_dataset(name='targets',
                                data=targets.data.cpu().numpy().squeeze())

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print('')
    print(f'This took {time.time() - script_start:.1f} seconds in total!')
    print('')
