"""
Provide dataset classes that encapsulate access to the data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import numpy as np
import torch
import torch.utils.data

from .samplefiles import get_normalization_from_samplefile, \
    get_samples_from_samplefile
from .configfiles import get_config


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class InjectionDataset(torch.utils.data.Dataset):
    """
    An ``InjectionDataset`` provides access to a synthetic (i.e.,
    injection-based) data set generated with the ggwd repository:
    https://github.com/timothygebhard/ggwd.

    Args:
        mode: Which kind of dataset to load. Options: 'training',
            'validation', 'testing'.
        sample_type: Load only examples with / without injections, or
            both? Options: 'injection', 'noise', 'both'.
    """

    def __init__(self,
                 mode: str,
                 sample_type: str = 'both'):

        # Basic sanity check: Must select valid mode
        if mode not in ():
            raise ValueError('mode must be one of the following:'
                             '"training", "validation" or "testing"!')

        # Load configuration file and get paths to data sets
        config = get_config()
        data_paths = config['data']

        # Get the normalization parameters for the data (always from training)
        normalization = \
            get_normalization_from_samplefile(data_paths['training'])

        # Load data from HDF samplefile
        self.data, self.labels = \
            get_samples_from_samplefile(file_path=data_paths[mode],
                                        normalization=normalization,
                                        sample_type=sample_type,
                                        label_length=0.2,
                                        cutoff_left=2047,
                                        cutoff_right=2048)

    # -------------------------------------------------------------------------

    def __len__(self):

        return len(self.data)

    # -------------------------------------------------------------------------

    def __getitem__(self, index):

        # Convert data to torch tensor and return
        data = torch.tensor(self.data[index]).float()
        label = torch.tensor(self.labels[index]).float()

        return data, label


class RealEventDataset(torch.utils.data.Dataset):
    """
    A ``RealEventDataset`` provides access to a data set generated with
    the ggwd repository (see above) containing confirmed GW events.

    Args:
        event: Name of the GW event to load, e.g., 'GW150914'.
    """

    def __init__(self,
                 event: str = None):

        # Load configuration file and get paths to data sets
        config = get_config()
        file_path = config['data']['real_events']

        # Get the normalization parameters for the data (always from training)
        normalization = \
            get_normalization_from_samplefile(config['data']['training'])

        # Open the HDF file and load the data for the event
        with h5py.File(file_path, 'r') as hdf_file:
            h1_strain = np.array(hdf_file[event]['h1_strain'])
            l1_strain = np.array(hdf_file[event]['l1_strain'])

        # Apply normalization
        h1_strain = \
            (h1_strain - normalization['h1_mean']) / normalization['h1_std']
        l1_strain = \
            (l1_strain - normalization['l1_mean']) / normalization['l1_std']

        # Construct the data
        self.data = np.vstack((h1_strain, l1_strain))

        # Construct a fake label (which we will never use)
        self.label = np.zeros(h1_strain.shape)

    # -------------------------------------------------------------------------

    def __len__(self):
        return 1

    # -------------------------------------------------------------------------

    def __getitem__(self, index):

        # Convert data to torch tensor and return
        data = torch.tensor(self.data).float()
        label = torch.tensor(self.label).float()

        return data, label
