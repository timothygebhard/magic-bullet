"""
Train the fully convolutional neural network model with PyTorch.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import time
import torch

from tensorboardX import SummaryWriter
from typing import Any

from utils.checkpointing import CheckpointManager
from utils.datasets import InjectionDataset
from utils.models import FCNN
from utils.training import AverageMeter, get_log_dir, update_lr


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
    parser.add_argument('--batch-size',
                        default=64,
                        type=int,
                        metavar='N',
                        help='Size of the mini-batches during training. '
                             'Default: 64.')
    parser.add_argument('--epochs',
                        default=64,
                        type=int,
                        metavar='N',
                        help='Total number of training epochs. Default: 64.')
    parser.add_argument('--learning-rate',
                        default=3e-4,
                        type=float,
                        metavar='LR',
                        help='Initial learning rate. Default: 3e-4.')
    parser.add_argument('--log-interval',
                        default=32,
                        type=int,
                        metavar='N',
                        help='Logging interval during training. Default: 32.')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='Path to checkpoint to be used when resuming '
                             'training. Default: None.')
    parser.add_argument('--tensorboard',
                        action='store_true',
                        default=True,
                        help='Use TensorBoard to log training progress? '
                             'Default: True.')
    parser.add_argument('--use-cuda',
                        action='store_true',
                        default=True,
                        help='Train on GPU, if available? Default: True.')
    parser.add_argument('--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='Number of workers for DataLoaders. Default: 4.')

    # Parse and return the arguments (as a Namespace object)
    arguments = parser.parse_args()
    return arguments


def train(dataloader: torch.utils.data.DataLoader,
          model: torch.nn.Module,
          loss_func: Any,
          optimizer: torch.optim.Optimizer,
          epoch: int,
          args: argparse.Namespace):
    """
    Train the given model for a single epoch using the given dataloader.

    Args:
        dataloader: The dataloader containing the training data.
        model: Instance of the model that is being trained.
        loss_func: A loss function to compute the error between the
            actual and the desired output of the model.
        optimizer: An instance of an optimizer that is used to compute
            and perform the updates to the weights of the network.
        epoch: The current training epoch.
        args: Namespace object containing some global variable (e.g.,
            command line arguments, such as the batch size)
    """

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Activate training mode
    model.train()

    # Keep track the time to process a batch, as well as the batch losses
    batch_times = AverageMeter()
    batch_losses = AverageMeter()

    # -------------------------------------------------------------------------
    # Process the training dataset in mini-batches
    # -------------------------------------------------------------------------

    for batch_idx, (data, target) in enumerate(dataloader):

        # Initialize start time of the batch
        batch_start = time.time()

        # Fetch data and move to device
        data, target = data.to(args.device), target.to(args.device)
        target = target.squeeze()

        # Clear gradients
        optimizer.zero_grad()

        # Compute forward pass through model
        output = model.forward(data).squeeze()

        # Calculate the loss for the batch
        loss = loss_func(output, target)

        # Back-propagate the loss and update the weights
        loss.backward()
        optimizer.step(closure=None)

        # ---------------------------------------------------------------------
        # Log information about current batch to TensorBoard
        # ---------------------------------------------------------------------

        if args.tensorboard:
            
            # Compute how many examples we have processed already and log the
            # loss value for the current batch
            global_step = ((epoch - 1) * args.n_train_batches + batch_idx) * \
                          args.batch_size
            args.logger.add_scalar(tag='loss/train',
                                   scalar_value=loss.item(),
                                   global_step=global_step)

        # ---------------------------------------------------------------------
        # Additional logging to console
        # ---------------------------------------------------------------------

        # Store the loss and processing time for the current batch
        batch_losses.update(loss.item())
        batch_times.update(time.time() - batch_start)

        # Print information to console, if applicable
        if batch_idx % args.log_interval == 0:
            
            # Which fraction of batches have we already processed this epoch?
            percent = 100. * batch_idx / args.n_train_batches

            # Print some information about how the training is going
            print(f'Epoch: {epoch:>3}/{args.epochs}', end=' | ', flush=True)
            print(f'Batch: {batch_idx:>3}/{args.n_train_batches}',
                  flush=True, end=' ')
            print(f'({percent:>4.1f}%)', end=' | ', flush=True)
            print(f'Loss: {loss.item():.6f}', end=' | ', flush=True)
            print(f'Time: {batch_times.value:>6.3f}s', flush=True)


def validate(dataloader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             loss_func: Any,
             epoch: int,
             args: argparse.Namespace) -> float:
    """
    At the end of each epoch, run the model on the validation dataset.

    Args:
        dataloader: The dataloader containing the validation data.
        model: Instance of the model that is being trained.
        loss_func: A loss function to compute the error between the
            actual and the desired output of the model.
        epoch: The current training epoch.
        args: Namespace object containing some global variable (e.g.,
            command line arguments, such as the batch size).

    Returns:
        The average loss on the validation dataset.
    """

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Activate model evaluation mode
    model.eval()

    # Initialize validation loss as 0, because we need to sum it up over all
    # mini batches for the validation dataset
    validation_loss = 0

    # Ensure the loss function uses 'sum' as the reduction method (we don't
    # want batch averages, but just one global validation average)
    reduction = loss_func.reduction
    loss_func.reduction = 'sum'

    # -------------------------------------------------------------------------
    # Process the validation dataset in mini-batches
    # -------------------------------------------------------------------------

    # At test time, we do not need to compute gradients
    with torch.no_grad():

        # Loop in mini batches over the validation dataset
        for data, target in dataloader:
            
            # Fetch batch data and move to device
            data, target = data.to(args.device), target.to(args.device)
            target = target.squeeze()

            # Compute the forward pass through the model
            output = model.forward(data).squeeze()

            # Compute the loss for the batch
            validation_loss += loss_func(output, target).item()

    # -------------------------------------------------------------------------
    # Compute the average validation loss
    # -------------------------------------------------------------------------

    validation_loss /= np.prod(dataloader.dataset.labels.shape)
    print(f'\nAverage loss on validation set:\t {validation_loss:.5f}\n')

    # -------------------------------------------------------------------------
    # Log stuff to TensorBoard
    # -------------------------------------------------------------------------

    if args.tensorboard:
        
        # Compute the current global_step (i.e., the total number examples
        # that we've seen during training so far)
        global_step = epoch * args.n_train_batches * args.batch_size

        # Log the validation loss
        args.logger.add_scalar(tag='loss/validation',
                               scalar_value=validation_loss,
                               global_step=global_step)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    # Finally, restore the original reduction method of the loss function
    loss_func.reduction = reduction

    # Return the validation loss
    return validation_loss


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print('')
    print('TRAIN A FULLY CONVOLUTIONAL NEURAL NETWORK')
    print('')

    # Start the stopwatch
    script_start = time.time()

    # Read in command line arguments
    args = get_arguments()

    print('Preparing the training process:')
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
    # Set up the network model
    # -------------------------------------------------------------------------

    # Create a new instance of the model we want to train
    model = FCNN()

    print('model: \t\t\t', model.__class__.__name__)

    # DataParallel will divide and allocate batch_size to all available GPUs
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)

    # Move model to the correct device
    model.to(args.device)

    # -------------------------------------------------------------------------
    # Instantiate an optimizer, a loss function and a LR scheduler
    # -------------------------------------------------------------------------

    # Instantiate the specified optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate,
                                 amsgrad=True)
    print('optimizer: \t\t', optimizer.__class__.__name__)

    # Define the loss function (we use Binary Cross-Entropy)
    loss_func = torch.nn.BCELoss().to(args.device)
    print('loss_function: \t\t', loss_func.__class__.__name__)

    # Reduce the LR by a factor of 0.5 if the validation loss did not
    # go down for at least 10 training epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.5,
                                                           patience=8,
                                                           min_lr=1e-6)

    # -------------------------------------------------------------------------
    # Instantiate a CheckpointManager and load checkpoint (if desired)
    # -------------------------------------------------------------------------
    
    # Instantiate a new CheckpointManager
    checkpoint_manager = CheckpointManager(model=model,
                                           optimizer=optimizer,
                                           scheduler=scheduler,
                                           mode='min',
                                           step_size=-1)

    # Check if we are resuming training, and if so, load the checkpoint
    if args.resume is not None:
        
        # Load the checkpoint from the provided checkpoint file
        checkpoint_manager.load_checkpoint(args.resume)
        args.start_epoch = checkpoint_manager.last_epoch + 1

        # Print which checkpoint we are using and where we start to train
        print(f'checkpoint:\t\t {args.resume} '
              f'(epoch: {checkpoint_manager.last_epoch})')

    # Other, simply print that we're not using any checkpoint
    else:
        args.start_epoch = 1
        print('checkpoint: \t\t None')

    # -------------------------------------------------------------------------
    # Load datasets for training and validation and create DataLoader objects
    # -------------------------------------------------------------------------

    # Load the training and the validation dataset
    training_dataset = InjectionDataset(mode='training')
    validation_dataset = InjectionDataset(mode='validation')

    # Compute size of training / validation set and number of training batches
    args.train_size = len(training_dataset)
    args.validation_size = len(validation_dataset)
    args.n_train_batches = int(np.ceil(args.train_size / args.batch_size))
    print('train_set_size: \t', args.train_size)
    print('validation_set_size: \t', args.validation_size)
    print('n_training_batches: \t', args.n_train_batches)

    # Create DataLoaders for training and validation
    training_dataloader = \
        torch.utils.data.DataLoader(dataset=training_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.workers,
                                    pin_memory=True)
    validation_dataloader = \
        torch.utils.data.DataLoader(dataset=validation_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=True)

    # -------------------------------------------------------------------------
    # Create a TensorBoard logger and log some basics
    # -------------------------------------------------------------------------

    if args.tensorboard:

        # Create TensorBoard logger
        args.logger = SummaryWriter(log_dir=get_log_dir())

        # Add all args to as text objects (to epoch 0)
        for key, value in dict(vars(args)).items():
            args.logger.add_text(tag=key,
                                 text_string=str(value),
                                 global_step=0)

    # -------------------------------------------------------------------------
    # Train the network for the given number of epochs
    # -------------------------------------------------------------------------

    print(80 * '-' + '\n\n' + 'Training the Model:\n' + 80 * '-')

    for epoch in range(args.start_epoch, args.epochs):

        print('')
        epoch_start = time.time()

        # ---------------------------------------------------------------------
        # Train the model for one epoch
        # ---------------------------------------------------------------------

        train(dataloader=training_dataloader,
              model=model,
              loss_func=loss_func,
              optimizer=optimizer,
              epoch=epoch,
              args=args)

        # ---------------------------------------------------------------------
        # Evaluate on the validation set
        # ---------------------------------------------------------------------

        validation_loss = validate(dataloader=validation_dataloader,
                                   model=model,
                                   loss_func=loss_func,
                                   epoch=epoch,
                                   args=args)

        # ---------------------------------------------------------------------
        # Take a step with the CheckpointManager
        # ---------------------------------------------------------------------

        # This will create checkpoint if the current model is the best we've
        # seen yet, and also once every `step_size` number of epochs.
        checkpoint_manager.step(metric=validation_loss,
                                epoch=epoch)

        # ---------------------------------------------------------------------
        # Update the learning rate of the optimizer (using the LR scheduler)
        # ---------------------------------------------------------------------

        # Take a step with the LR scheduler; print message when LR changes
        current_lr = update_lr(scheduler, optimizer, validation_loss)

        # Log the current value of the LR to TensorBoard
        if args.tensorboard:
            args.logger.add_scalar(tag='learning_rate',
                                   scalar_value=current_lr,
                                   global_step=epoch)

        # ---------------------------------------------------------------------
        # Print epoch duration
        # ---------------------------------------------------------------------

        print(f'Total Epoch Time: {time.time() - epoch_start:.3f}s\n')

        # ---------------------------------------------------------------------

    print(80 * '-' + '\n\n' + 'Training complete!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print('')
    print(f'This took {time.time() - script_start:.1f} seconds!')
    print('')
