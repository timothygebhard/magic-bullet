"""
Provide a class to deal with creating and loading model checkpoints.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import os
import torch

from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class CheckpointManager(object):
    """
    A ``CheckpointManager`` is a convenience wrapper to automatically
    create model checkpoints by saving the `state_dict` of a model and
    its optimizer as *.pth files in a specified directory. This class
    closely follows the API of PyTorch optimizers and learning rate
    schedulers.

    .. note:
       For `DataParallel` modules, `model.module.state_dict()`
       is saved instead of `model.state_dict()`.

    Source: This class is a modified version of the following gist:
        https://gist.github.com/kdexd/1b78bb541612f34a8871bedd24668030

    Args:
        model: Wrapped model to be checkpointed.
        optimizer: Wrapped optimizer to be checkpointed.
        scheduler: Wrapped LR scheduler to be checkpointed.
        checkpoints_directory: Path to the directory to save
            checkpoints (may not exist and will be created).
        mode: One of `min`, `max`. In `min` mode, best checkpoint
            will be recorded when metric hits a lower value; in `max`
            mode it will be recorded when metric hits a higher value.
        step_size: Period of saving checkpoints. A value of 1 means
            that at the end of each epoch, a checkpoint is created.
            If a value of -1 is given, only the "best model" checkpoint
            is created.
        last_epoch: The index of last epoch.
        verbose: Whether or not to print messages when checkpoints are
            being saved.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Any,
                 checkpoints_directory: str = './checkpoints',
                 mode: str = 'min',
                 step_size: int = 1,
                 last_epoch: int = -1,
                 verbose: bool = True):

        # ---------------------------------------------------------------------
        # Basic sanity checks for constructor arguments
        # ---------------------------------------------------------------------
        
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f'{type(model).__name__} is not a Module')
        
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        
        if not scheduler.__module__ == 'torch.optim.lr_scheduler':
            raise TypeError(f'{type(scheduler).__name__} is not a Scheduler')
        
        # ---------------------------------------------------------------------
        # Store constructor arguments and initialize checkpoint directory
        # ---------------------------------------------------------------------
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoints_directory = checkpoints_directory
        self.mode = mode
        self.step_size = step_size
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        self.initialize_checkpoints_directory()
        
        # ---------------------------------------------------------------------
        # Create additional class variable (best metric value so far)
        # ---------------------------------------------------------------------
        
        self.best_metric = np.inf * (-1 if mode == 'max' else 1)
    
    # -------------------------------------------------------------------------
    
    def initialize_checkpoints_directory(self):
        """
        Initialize the checkpoints directory (e.g., ensure it exists).
        """
        
        # Make sure the checkpoint_directory exists. If it does not exist,
        # create it (including necessary parent directories)
        Path(self.checkpoints_directory).mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    
    def step(self,
             metric: float,
             epoch: int = None):
        """
        Save checkpoint if step size conditions meet, and update best
        checkpoint based on `metric` and `self.mode`.

        Args:
            metric: The value of the metric that is being monitored
                (e.g., the validation loss) for the current epoch.
            epoch: The current epoch. If no value is provided, we'll
                try to keep counting from the original `last_epoch`.
        """
        
        # Get the current epoch, either explicitly or by counting from the
        # initial `last_epoch` the CheckpointManager was instantiated with.
        if not epoch:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        # ---------------------------------------------------------------------
        # Save best model checkpoint if applicable
        # ---------------------------------------------------------------------
        
        # Check if the current model is the best we have seen so far according
        # to the provided metric and mode
        if (self.mode == "min" and metric < self.best_metric) or \
                (self.mode == "max" and metric > self.best_metric):
            
            # Update the value for the best metric
            self.best_metric = metric
            
            # Save the current checkpoint as the best checkpoint
            self.save_checkpoint(checkpoint=self.get_current_checkpoint(),
                                 name='best.pth')
        
        # ---------------------------------------------------------------------
        # Save regular checkpoint (every `step_size` epochs)
        # ---------------------------------------------------------------------
        
        if (self.step_size > 0) and (self.last_epoch % self.step_size == 0):
            self.save_checkpoint(checkpoint=self.get_current_checkpoint(),
                                 name=f'epoch_{epoch}.pth')
    
    # -------------------------------------------------------------------------
    
    def save_checkpoint(self,
                        checkpoint: dict,
                        name: str):
        """
        Save a new `checkpoint` in the `checkpoints_directory` with a
        given `name` (usually as a *.pth file).
        """
        
        # Save the checkpoint in the pre-specified directory
        torch.save(checkpoint,
                   os.path.join(self.checkpoints_directory, name))
        
        # Print a message if needed
        if self.verbose:
            print(f'Saved checkpoint: {name}')
    
    # -------------------------------------------------------------------------
    
    def load_checkpoint(self,
                        checkpoint_file_path: str):
        """
        Load a previously saved checkpoint into the CheckpointManager.

        Args:
            checkpoint_file_path: The path to the file containing a
                saved checkpoint.
        """
        
        # Make sure the checkpoint we want to load exists!
        if not os.path.exists(checkpoint_file_path):
            raise FileNotFoundError(f'{checkpoint_file_path} does not exist!')
        
        # Load the checkpoint file
        checkpoint = torch.load(checkpoint_file_path)
        
        # Load the state dicts of the model, optimizer and LR scheduler
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.scheduler.load_state_dict(checkpoint['sched_state_dict'])
        
        # Also load the last epoch and the best metric value
        self.last_epoch = checkpoint['last_epoch']
        self.best_metric = checkpoint['best_metric']
    
    # -------------------------------------------------------------------------
    
    def get_current_checkpoint(self):
        """
        Returns a dict containing the state dict of model (taking care
        of DataParallel case), the optimizer, the scheduler, the best
        metric and the epoch.

        Returns:
            A checkpoint, containing the current state_dict for the
            model, the optimizer, the LR scheduler, as well as the
            values for the best metric and the last_epoch.
        """
        
        # Get the state dict of the model, taking care of the DataParallel
        # case (which modifies the name of the state dict)
        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        # Collect the full checkpoint in a dict an return it
        checkpoint = dict(model_state_dict=model_state_dict,
                          optim_state_dict=self.optimizer.state_dict(),
                          sched_state_dict=self.scheduler.state_dict(),
                          best_metric=self.best_metric,
                          last_epoch=self.last_epoch)
        
        return checkpoint
