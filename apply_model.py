"""
Apply the trained model to the test dataset and make predictions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import h5py
import numpy as np
import os
import time
import torch

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.configfiles import get_config
from utils.models import FCNN
from utils.datasets import InjectionDataset, RealEventDataset


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
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--apply-to',
                        default='testing',
                        type=str,
                        metavar='NAME',
                        choices=['testing', 'real_events'],
                        help='Dataset to which the trained model should be '
                             'applied: either "testing" or "real_events". '
                             'Default: testing.')
    parser.add_argument('--batch-size',
                        default=128,
                        type=int,
                        metavar='N',
                        help='Size of the mini-batches in which the data set '
                             'is split for applying the trained network. '
                             'Default: 128.')
    parser.add_argument('--checkpoint',
                        default='./checkpoints/best.pth',
                        type=str,
                        metavar='FILE',
                        help='Path to the checkpoint file from which to load '
                             'the model. Default: ./checkpoints/best.pth.')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        metavar='N',
                        help='Number of workers for the PyTorch DataLoader. '
                             'Default: 8')
    parser.add_argument('--use-cuda',
                        action='store_true',
                        default=True,
                        help='Use GPU, if available? Default: True.')

    # Parse and return the arguments (as a Namespace object)
    arguments = parser.parse_args()
    return arguments


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print('')
    print('APPLY MODEL TO TEST DATASET OR REAL EVENTS')
    print('')
    
    # Start the stopwatch
    script_start = time.time()
    
    # Get command line arguments
    args = get_arguments()

    # Load the global configuration file
    config = get_config()

    print('Preparing the prediction process:')
    print(80 * '-')

    # -------------------------------------------------------------------------
    # Set up CUDA for GPU support
    # -------------------------------------------------------------------------

    if torch.cuda.is_available() and args.use_cuda:
        args.device = 'cuda'
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f'device: \t\t GPU ({device_count} x {device_name})')
    else:
        args.device = 'cpu'
        print('device: \t\t CPU [CUDA not requested or unavailable]')

    # -------------------------------------------------------------------------
    # Create a new instance of the model and load weights from checkpoint file
    # -------------------------------------------------------------------------

    # Create a new instance of the model we have previously trained
    model = FCNN()
    print('model: \t\t\t', model.__class__.__name__)

    # Make sure the checkpoint we want to load exists!
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'{args.checkpoint} does not exist!')

    # Read the checkpoint file and load the model_state_dict
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to the correct device (CPU or GPU)
    model.to(args.device)

    # DataParallel will divide and allocate batch_size to all available GPUs
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)

    # Document again which checkpoint we have loaded
    print(f'checkpoint:\t\t {args.checkpoint}')

    # -------------------------------------------------------------------------
    # Load the data to which we want to apply the trained model
    # -------------------------------------------------------------------------

    print(f'dataset:\t\t {args.apply_to}')
    print(80 * '-' + '\n')

    # Initialize empty dictionaries for the datasets and dataloaders
    datasets = dict()
    dataloaders = dict()

    # Define options for the dataloader (same for all)
    dataloader_options = dict(batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True)

    # Get the dataset: Either the testing data, ...
    if args.apply_to == 'testing':
    
        # For the evaluation, we need to distinguish between examples with
        # and without an injection, so we need to process them separately
        for sample_type in ('injection', 'noise'):

            # Get examples with injections and construct a dataloader from it
            datasets[sample_type] = \
                InjectionDataset(mode='testing', sample_type=sample_type)
            dataloaders[sample_type] = \
                DataLoader(dataset=datasets[sample_type], **dataloader_options)

    # ... or all available real events
    else:
    
        # Open the HDF containing the real events to find the ones for
        # which we have pre-processed strain data available
        with h5py.File(config['data']['real_events'], 'r') as hdf_file:
            events = list(hdf_file.keys())

        # For each event, get the dataset and construct a dataloader
        for event in events:
            datasets[event] = RealEventDataset(event=event)
            dataloaders[event] = DataLoader(dataset=datasets[event],
                                            **dataloader_options)

    # -------------------------------------------------------------------------
    # Prepare the directory and file where to store the predictions
    # -------------------------------------------------------------------------

    # Make sure the results directory exists
    results_dir = './results/'
    Path(results_dir).mkdir(exist_ok=True)

    # Construct the path to the file that holds all predictions
    predictions_file_name = f'predictions_{args.apply_to}.hdf'
    predictions_file_path = os.path.join(results_dir, predictions_file_name)

    # -------------------------------------------------------------------------
    # Loop over the dataset and apply the model to get its predictions
    # -------------------------------------------------------------------------

    # Activate evaluation mode for the model
    model.eval()

    # Keep track of the predictions we are producing
    predictions = {sample_type: list() for sample_type in dataloaders.keys()}

    for dataset_name in dataloaders.keys():

        # At test time, we do not need to compute gradients
        print(f'Making predictions for "{dataset_name}":')
        with torch.no_grad():

            # Loop in mini batches over the validation dataset
            for data, target in tqdm(iterable=dataloaders[dataset_name],
                                     total=len(dataloaders[dataset_name]),
                                     ncols=80):

                # Fetch batch data and move to device
                data = data.to(args.device)
                target = target.to(args.device).squeeze()

                # Compute the forward pass through the model
                output = model.forward(data).squeeze()

                # Store the predictions (as a numpy array)
                predictions[dataset_name].append(output.cpu().numpy())

        # Convert the list of predictions into a numpy array
        predictions[dataset_name] = np.vstack(predictions[dataset_name])

        # Save the predictions to the output HDF file
        print('Saving predictions to HDF file...', end=' ')
        with h5py.File(predictions_file_path, 'a') as hdf_file:

            # Delete the dataset if it already exists
            if dataset_name in hdf_file.keys():
                del hdf_file[dataset_name]

            # Create a new dataset holding the predictions
            hdf_file.create_dataset(name=dataset_name,
                                    data=predictions[dataset_name])
        print('Done!\n')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print('')
    print(f'This took {time.time() - script_start:.1f} seconds!')
    print('')
