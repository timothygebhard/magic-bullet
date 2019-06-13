"""
Plot the network predictions on real GW events.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from pathlib import Path

from utils.configfiles import get_config
from utils.evaluation import find_binary_peaks
from utils.datasets import RealEventDataset


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print('')
    print('PLOT NETWORK PREDICTIONS ON REAL GW EVENTS')
    print('')

    # Start the stopwatch
    script_start = time.time()

    # Get the global configuration
    config = get_config()

    # -------------------------------------------------------------------------
    # Read in the predictions for real events from the corresponding HDF file
    # -------------------------------------------------------------------------

    # Keep track of all predictions we are loading
    predictions = dict()

    # Open the HDF file and get the predictions
    print('Loading predictions from HDF file...', end=' ')
    with h5py.File('./results/predictions_real_events.hdf', 'r') as hdf_file:
        for event in hdf_file.keys():
            predictions[event] = np.array(hdf_file[event])
    print('Done!')

    # List the elements for which we will create a plot
    print('')
    print('Found predictions for the following GW events:')
    print(', '.join(predictions.keys()))
    print('')

    # -------------------------------------------------------------------------
    # Create a plot for every event
    # -------------------------------------------------------------------------

    for event in predictions.keys():

        print(f'Creating plot for event {event}...', end=' ', flush=True)

        # ---------------------------------------------------------------------
        # Plot the strains, predictions and a few auxiliary lines
        # ---------------------------------------------------------------------

        # Set up the subplots
        width = 7 * 1.21
        fig, axes = plt.subplots(figsize=(width, 2), nrows=3)

        # Add some horizontal lines first
        axes[0].axhline(y=0.0, ls='-', color='#D9D9D9', lw=0.5)
        axes[1].axhline(y=0.0, ls='-', color='#D9D9D9', lw=0.5)
        axes[2].axhline(y=0.0, ls='-', color='#D9D9D9', lw=0.5)
        axes[2].axhline(y=0.5, ls='-', color='#D9D9D9', lw=0.5)
        axes[2].axhline(y=1.0, ls='-', color='#D9D9D9', lw=0.5)

        # Get the strain data for the event
        dataset = RealEventDataset(event=event)
        strain, _ = dataset[0]
        strain = strain.numpy()

        # Plot the strain data for H1 and L1
        grid = np.linspace(-0.25, 0.25, 1024)
        axes[0].plot(grid, strain[0, 16384 - 512:16384 + 512], lw=1)
        axes[1].plot(grid, strain[1, 16384 - 512:16384 + 512], lw=1)

        # Select the unprocessed prediction
        raw_prediction = predictions[event][0]

        # Smooth and threshold the prediction
        kernel = np.full(256, 1.0 / 256)
        smoothed_prediction = np.convolve(raw_prediction, kernel, mode='same')
        thresholded_prediction = (smoothed_prediction > 0.5).astype(float)

        # Plot the raw, smoothed and thresholded prediction
        axes[2].plot(grid, raw_prediction[14336 - 512:14336 + 512],
                     label='Raw network prediction', lw=1)
        axes[2].plot(grid, smoothed_prediction[14336 - 512:14336 + 512],
                     label='Smoothed prediction', lw=1)
        axes[2].plot(grid, thresholded_prediction[14336 - 512:14336 + 512],
                     label='Thresholded prediction', lw=1)

        # Find and plot the predicted event times
        peaks = find_binary_peaks(thresholded_prediction)
        for peak in peaks:
            for ax in axes:
                ax.axvline(x=(peak / 2048 - 7), ls='-',
                           lw=0.75, color='Crimson')

        # ---------------------------------------------------------------------
        # Set up options for the plot
        # ---------------------------------------------------------------------

        # Set up axes labels
        axes[2].set_xlabel('Time from Center-of-Earth time (s)', fontsize=8)
        axes[0].set_ylabel('H1', fontsize=8)
        axes[1].set_ylabel('L1', fontsize=8)
        axes[2].set_ylabel('CNN', fontsize=8)

        # Set up axes limits for strain panels
        for i in range(2):
            axes[i].set_xlim(-0.25, 0.25)
            axes[i].set_ylim(-6, 6)
            axes[i].set_yticks([-5, 0, 5])

        # Set up axes limits for prediction panel
        axes[2].set_xlim(-0.25, 0.25)
        axes[2].set_ylim(-0.1, 1.1)

        # Adjust label sizes
        for i in range(3):
            axes[i].tick_params(axis='both', which='major', labelsize=6)
            axes[i].tick_params(axis='both', which='minor', labelsize=6)

        # Add a legend to the prediction panel
        axes[2].legend(loc='center left', fontsize=6, framealpha=1)
    
        plt.subplots_adjust(hspace=0)
        fig.align_ylabels(axes)

        # ---------------------------------------------------------------------
        # Save the plot
        # ---------------------------------------------------------------------

        # Construct path for this plot file
        plots_dir = './plots'
        Path(plots_dir).mkdir(exist_ok=True)
        file_path = os.path.join(plots_dir, f'{event}.pdf')

        # Save the plot at the constructed location (as a PDF)
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print('')
    print(f'This took {time.time() - script_start:.1f} seconds!')
    print('')
