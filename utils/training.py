"""
Provide utilities and tools for the training process.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import datetime
import os
import time
import torch

from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self,
               value: float,
               steps: int = 1):
        self.value = value
        self.sum += value * steps
        self.count += steps
        self.average = self.sum / self.count


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate for a given optimizer.

    Args:
        optimizer: The optimizer whose learning rate we want to get.

    Returns:
        The learning rate of the given `optimizer`.
    """

    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_log_dir(log_base_dir: str = './tensorboard',
                ensure_exists: bool = True) -> str:
    """
    Create the path to the directory where the TensorBoard logs are
    stored, using a "YYYY-MM-DD_HH:MM:SS" naming scheme for the
    different runs.

    Args:
        log_base_dir: The base directory where the folders for
            different runs should be created.
        ensure_exists: Whether or not to create the directory in case
            it does not exist yet.

    Returns:
        The path the to log directory for the current run.
    """

    # Create log dir path based on current timestamp
    timestamp = datetime.datetime.fromtimestamp(time.time())
    formatted_timestamp = timestamp.strftime('%Y-%m-%d_%H:%M:%S')
    log_dir = os.path.join(log_base_dir, formatted_timestamp)

    # Make sure the directory exists
    if ensure_exists:
        Path(log_dir).mkdir(exist_ok=True, parents=True)

    return log_dir


def update_lr(scheduler: Any,
              optimizer: torch.optim.Optimizer,
              validation_loss: float) -> float:
    """
    Perform a `step()` with the learning rate scheduler, and print the
    new learning rate in case we have decreased it.

    Args:
        scheduler: Instance of a PyTorch LR scheduler class, for
            example, ``ReduceLROnPlateau``.
        optimizer: An instance of an optimizer that is used to compute
            and perform the updates to the weights of the network.
        validation_loss: The last loss obtained on the validation set.

    Returns:
        The current learning rate (LR).
    """

    # Get the current learning rate, then take a step with the
    # scheduler, and get the learning rate after that
    old_lr = get_lr(optimizer)
    scheduler.step(validation_loss)
    new_lr = get_lr(optimizer)

    # If the LR has changed, print a message to the console
    if new_lr != old_lr:
        print(f'Reduced Learning Rate (LR): {old_lr} -> {new_lr}')

    return new_lr
