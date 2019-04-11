"""
Plot the Detection Ratio over the Injection SNR.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from pathlib import Path


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print('')
    print('PLOT DETECTION RATIO OVER THE INJECTION SNR')
    print('')

    # Start the stopwatch
    script_start = time.time()

    # -------------------------------------------------------------------------
    # Load results from JSON file
    # -------------------------------------------------------------------------

    # Read the data from the JSON file
    print('Reading in data...', end=' ', flush=True)
    with open('./results/dr_over_snr.json', 'r') as json_file:
        results = json.load(json_file)
    print('Done!', flush=True)

    # Fix the data types
    snr_bins = [(float(a), float(b)) for (a, b) in results['snr_bins']]
    detection_ratios = np.array(results['detection_ratios']).astype(float)

    # -------------------------------------------------------------------------
    # Actually make the plot: detection ratio over injection SNR
    # -------------------------------------------------------------------------

    print('Creating plot...', end=' ', flush=True)

    # Plot vertical lines to separate SNR bins
    for x in np.unique(np.array(snr_bins).flatten()):
        plt.axvline(x=x, color='lightgray', lw=0.5)

    # Plot the PyCBC threshold for comparison
    plt.axvline(x=np.sqrt(2 * 5.5**2), color='Crimson', lw=0.75)

    # Get the bin centers for plotting
    grid = [np.mean(_) for _ in snr_bins]

    # Plot the linear interpolation
    plt.plot(grid, detection_ratios, color='C0', lw=1)

    # Plot the data points and the error bars
    plt.errorbar(x=grid, y=detection_ratios,
                 xerr=len(grid) * [0.25],
                 ecolor='C0', ls='none', lw=1.5, zorder=99)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Set up options for the plot
    # -------------------------------------------------------------------------

    # Set up axes ticks and limits
    plt.xticks(np.linspace(5, 20, 16))
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlim(5, 20)
    plt.ylim(0.0, 1.025)

    # Set up axes labels
    plt.xlabel('Injection SNR', fontsize=8)
    plt.ylabel('Detection Ratio', fontsize=8)

    # Put a grid on the plot
    plt.grid(which='both', ls='--', alpha=0.35, lw=0.5)

    # Adjust label sizes
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.tick_params(axis='both', which='minor', labelsize=6)

    # Adjust the plot size
    plt.tight_layout()
    width = 8.6 / 2.54 / 1.015
    plt.gcf().set_size_inches(width, width, forward=True)

    # Construct path to save this plot
    plots_dir = './plots'
    Path(plots_dir).mkdir(exist_ok=True)
    file_path = os.path.join(plots_dir, 'dr_over_snr.pdf')

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
