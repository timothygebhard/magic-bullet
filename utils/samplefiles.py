"""
Provide functions to read data from sample HDF files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import h5py
import numpy as np

from typing import Tuple


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_normalization_from_samplefile(file_path: str) -> dict:
    """
    Extract the normalization parameters for a given HDF sample file.
    
    Args:
        file_path: Path to the sample file from which to read the data.

    Returns:
        A dict {'h1_mean', 'l1_mean', 'h1_std', 'l1_std'} containing
        the means and standard deviations of the sample file contents.
    """

    # Make sure that the sample file we want to read from exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Sample file "{file_path}" does not exist!')

    # Read the normalization parameters from the file
    with h5py.File(file_path, 'r') as hdf_file:
        h1_mean = float(hdf_file['normalization_parameters'].attrs['h1_mean'])
        l1_mean = float(hdf_file['normalization_parameters'].attrs['l1_mean'])
        h1_std = float(hdf_file['normalization_parameters'].attrs['h1_std'])
        l1_std = float(hdf_file['normalization_parameters'].attrs['l1_std'])

    # Return the values as a dictionary
    return dict(h1_mean=h1_mean,
                l1_mean=l1_mean,
                h1_std=h1_std,
                l1_std=l1_std)


def get_samples_from_samplefile(file_path: str,
                                normalization: dict = None,
                                sample_type: str = 'both',
                                label_length: float = 0.2,
                                cutoff_left: int = 2047,
                                cutoff_right: int = 2048) -> Tuple[np.ndarray,
                                                                   np.ndarray]:
    """
    Read in samples from a sample file and return them as properly
    shaped numpy arrays.
    
    Args:
        file_path: Path to the sample file from which to read the data.
        normalization: A dictionary specifying the parameters of the
            normalization transform that will be  applied to the data.
            This is to ensure that all samples are (approximately)
            standard normal, which is otherwise not the  case (because
            of PyCBC's normalization choice for the whitening).
        sample_type: Which samples to retrieve from the sample file.
            Options are: ['injection', 'noise', 'both'].
        label_length: Length of the label to be generated in seconds.
        cutoff_left: Number of time steps to cut off from the label
            on the left.
        cutoff_right: Number of time steps to cut off from the labels
            on the right.
            
    Returns:
        A tuple `(samples, labels)`, where `samples` is a numpy array
        with the shape `(n_samples, 2, sampling_rate * sample_length)`.
        It contains the samples from the sample file (both with and
        without injection, depending on `sample_type`).
        `labels` is also a numpy array with the shape `(n_samples, 1,
        sampling_rate * sample_length - cutoff_left - cutoff_right)`,
        containing the labels corresponding to the samples.
    """

    # -------------------------------------------------------------------------
    # Basic sanity checks
    # -------------------------------------------------------------------------

    # Make sure that the sample file we want to read from exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Sample file "{file_path}" does not exist!')

    if sample_type not in ('injection', 'noise', 'both'):
        raise ValueError(f'Invalid value for sample_type: {sample_type}')

    # -------------------------------------------------------------------------
    # Open the HDF sample file and read in the samples and labels
    # -------------------------------------------------------------------------

    with h5py.File(file_path, 'r') as hdf_file:

        # ---------------------------------------------------------------------
        # Get the label location from the static_args in the sample file
        # ---------------------------------------------------------------------

        # Get seconds_before_event and sampling_rate
        seconds_before_event = \
            float(hdf_file['static_arguments'].attrs['seconds_before_event'])
        sampling_rate = \
            float(hdf_file['static_arguments'].attrs['target_sampling_rate'])

        # Calculate the start and end indices for the label
        label_start = int((seconds_before_event - (0.5 * label_length)) *
                          sampling_rate)
        label_end = int((seconds_before_event + (0.5 * label_length)) *
                        sampling_rate + 1)

        # ---------------------------------------------------------------------
        # Initialize variables so the static code checker stops complaining
        # ---------------------------------------------------------------------

        inj_samples_h1 = None
        inj_samples_l1 = None
        inj_samples_label = None

        noise_samples_h1 = None
        noise_samples_l1 = None
        noise_samples_label = None

        # ---------------------------------------------------------------------
        # Read in injection samples if requested
        # ---------------------------------------------------------------------

        if sample_type in ('injection', 'both'):

            # Read in samples containing an injection
            inj_samples_h1 = np.array(hdf_file['/injection_samples/h1_strain'])
            inj_samples_l1 = np.array(hdf_file['/injection_samples/l1_strain'])

            # Dynamically create the label
            inj_samples_label = np.zeros(shape=inj_samples_h1.shape)
            inj_samples_label[:, label_start:label_end] = 1

        # ---------------------------------------------------------------------
        # Read in noise samples if requested
        # ---------------------------------------------------------------------

        if sample_type in ('noise', 'both'):

            # Read in samples not containing an injection
            noise_samples_h1 = np.array(hdf_file['/noise_samples/h1_strain'])
            noise_samples_l1 = np.array(hdf_file['/noise_samples/l1_strain'])

            # Dynamically create the label
            noise_samples_label = np.zeros(shape=noise_samples_h1.shape)

        # ---------------------------------------------------------------------
        # Stack samples if necessary
        # ---------------------------------------------------------------------

        if sample_type == 'injection':
            samples_h1 = inj_samples_h1
            samples_l1 = inj_samples_l1
            labels = inj_samples_label
        elif sample_type == 'noise':
            samples_h1 = noise_samples_h1
            samples_l1 = noise_samples_l1
            labels = noise_samples_label
        else:
            samples_h1 = np.concatenate([inj_samples_h1, noise_samples_h1])
            samples_l1 = np.concatenate([inj_samples_l1, noise_samples_l1])
            labels = np.concatenate([inj_samples_label, noise_samples_label])

        # ---------------------------------------------------------------------
        # Normalize the data
        # ---------------------------------------------------------------------

        # If no normalization was explicitly given, use default values.
        # Caution: Not passing a dict for normalization means that the data
        # will not be normalized at remain at the original PyCBC outputs!
        if normalization is None:
            normalization = {'h1_mean': 0.0, 'h1_std': 1.0,
                             'l1_mean': 0.0, 'l1_std': 1.0}

        # Apply the normalization transformation: Subtract the mean and divide
        # by the standard deviation.
        samples_h1 = \
            (samples_h1 - normalization['h1_mean']) / normalization['h1_std']
        samples_l1 = \
            (samples_l1 - normalization['l1_mean']) / normalization['l1_std']

        # ---------------------------------------------------------------------
        # Stack H1 and L1 and adjust axes and dimensions
        # ---------------------------------------------------------------------

        # Stack together H1 and L1 and adjust ordering of axes
        samples = np.dstack([samples_h1, samples_l1])
        samples = np.swapaxes(samples, 1, 2)

        # Add a dummy dimension to the labels and convert to float32
        labels = np.expand_dims(labels, 1).astype(np.float32)

        # ---------------------------------------------------------------------
        # Remove cutoff from labels
        # ---------------------------------------------------------------------

        # Remove left and right cutoff from label: This is what we lose due
        # to using valid convolutions without padding. The way we choose the
        # left and right cutoff determines the alignment of the input time
        # series with its label.
        if cutoff_right != 0:
            labels = labels[:, :, cutoff_left:-cutoff_right]
        else:
            labels = labels[:, :, cutoff_left:]

        # ---------------------------------------------------------------------
        # Return samples and labels
        # ---------------------------------------------------------------------

        return samples, labels
