"""
Plot the inverse false positive rate (IFPR) over delta_t, that is, the
width of the interval around the ground truth injection time in which
a prediction is still counted as a detection.
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


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print('')
    print('PLOT INVERSE FALSE POSITIVE RATE (iFPR) OVER DELTA T')
    print('')

    # Start the stopwatch
    script_start = time.time()
    
    # Get the global config
    config = get_config()

    # -------------------------------------------------------------------------
    # Load the data from the HDF file
    # -------------------------------------------------------------------------

    # Get the "slack widths", that is, the sizes of the interval around the
    # ground truth injection time in which a detection will still be counted.
    # Note: The size is given in seconds and must be converted to time steps
    # by multiplying with the sampling rate.
    slack_widths = config['evaluation']['all_slack_widths']
    slack_widths = sorted(np.unique(np.array(slack_widths).astype(float)))

    # Initialize a list for the false positive rates
    fpr_values = list()

    # Load the data from the HDF file
    with h5py.File('./results/found_triggers.hdf', 'r') as hdf_file:

        # Loop over all slack_width values
        for slack_width in slack_widths:

            # Define a shortcut
            sw = str(slack_width)

            # Get the false positive rate for the current slack width
            fpr_values.append(float(hdf_file[sw]['figures_of_merit']
                                    .attrs['false_positive_rate']))

    # Convert list to a numpy array
    fpr_values = np.array(fpr_values)

    # -------------------------------------------------------------------------
    # Create the plot: IFPR over \Delta t (slack_width)
    # -------------------------------------------------------------------------

    plt.plot(slack_widths, 1.0 / fpr_values, 'x', ms=3.5, color='black')

    # -------------------------------------------------------------------------
    # Set up global options for the plot
    # -------------------------------------------------------------------------

    plt.xscale('log')
    plt.xlim(4e-3, 1.2e0)
    plt.ylim(0, 1500)

    # Set up axes labels
    plt.xlabel(r'$\Delta$t (s)', fontsize=8)
    plt.ylabel('IFPR (s)', fontsize=8)

    # Put a grid on the plot
    plt.grid(which='both', ls='-', color='#D9D9D9', lw=0.5)

    # Adjust label sizes
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.tick_params(axis='both', which='minor', labelsize=6)

    # Adjust the plot size
    plt.tight_layout()
    width = 8.6 / 2.54 / 1.015
    plt.gcf().set_size_inches(width, 1 / 3 * width, forward=True)

    # Construct path to save this plot
    plots_dir = './plots'
    Path(plots_dir).mkdir(exist_ok=True)
    file_path = os.path.join(plots_dir, 'ifpr_over_delta_t.pdf')

    # Save the plot as a PDF
    print('Saving plot as PDF...', end=' ', flush=True)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print('')
    print(f'This took {time.time() - script_start:.1f} seconds!')
    print('')
