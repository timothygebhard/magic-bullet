"""
Plot the detection ratio (DR) over the injection SNR.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from matplotlib.lines import Line2D
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
    slack_widths = [float(_) for _ in results['detection_ratios'].keys()]

    # -------------------------------------------------------------------------
    # Actually make the plot: detection ratio over injection SNR
    # -------------------------------------------------------------------------

    # Define the markers we will be using for the plot (and make it cyclic)
    def markers(index):
        markers_ = ['s', 'v', 'p', 'o', 'D', 'x', '>', '<']
        return markers_[index % len(markers_)]

    # Get the bin centers for plotting
    grid = [np.mean(_) for _ in snr_bins]

    # Plot the results for different slack_width values (i.e., \Delta t)
    for i, slack_width in enumerate(sorted(slack_widths)):

        print(f'Creating plot for slack_width={slack_width:.3f}...',
              end=' ', flush=True)

        # Get the detection ratios for this slack_width value
        detection_ratios = results['detection_ratios'][str(slack_width)]
        detection_ratios = np.array(detection_ratios).astype(float)

        # Make plot (plot step-function and actual data points)
        plt.plot(grid, detection_ratios, marker=markers(i), ms=2, mew=0.5,
                 color=f'C{i}', linestyle='None', zorder=99)
        plt.step(grid, detection_ratios, where='mid', lw=1.0, color=f'C{i}',
                 zorder=99)

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Set up global options for the plot
    # -------------------------------------------------------------------------

    # Plot additional vertical lines to separate SNR bins
    for x in np.unique(np.array(snr_bins).flatten()[1::2]):
        plt.axvline(x=x, color='#D9D9D9', lw=0.5, ls='-')

    # Plot the PyCBC threshold for comparison
    plt.axvline(x=np.sqrt(2 * 5.5**2), color='Crimson', lw=1.0)

    # Set up axes ticks and limits
    plt.xticks(np.linspace(5, 20, 16))
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlim(5, 20)
    plt.ylim(0.0, 1.025)

    # Set up axes labels
    plt.xlabel('Injection SNR', fontsize=8)
    plt.ylabel('Detection Ratio', fontsize=8)

    # Put a grid on the plot
    plt.grid(which='both', ls='-', color='#D9D9D9', lw=0.5)

    # Add a custom legend
    legend_elements = [Line2D([], [], ls='-', ms=2, mew=0.5, lw=1.0,
                              color=f'C{i}', marker=markers(i),
                              label=r'$\Delta$t = ' + f'{slack_width:.3f}')
                       for i, slack_width in enumerate(slack_widths)]
    plt.legend(handles=legend_elements, loc='lower right',
               fontsize=6, framealpha=1)

    # Adjust label sizes
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.tick_params(axis='both', which='minor', labelsize=6)

    # Adjust the plot size
    plt.tight_layout()
    width = 8.6 / 2.54 / 1.015
    plt.gcf().set_size_inches(width, 2 / 3 * width, forward=True)

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
