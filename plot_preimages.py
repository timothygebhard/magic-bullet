"""
Plot pre-images found through optimization.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import time
import os
import numpy as np

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()

    print('')
    print('PLOT PRE-IMAGES FOUND THROUGH OPTIMIZATION')
    print('')

    # -------------------------------------------------------------------------
    # FIND HDF FILES OF PRE-IMAGES
    # -------------------------------------------------------------------------

    preimages_dir = './results/preimages'
    file_names = [_ for _ in os.listdir(preimages_dir) if _.endswith('.hdf')]

    # -------------------------------------------------------------------------
    # LOOP OVER PRE-IMAGE FILES AND PLOT THE CONTENTS
    # -------------------------------------------------------------------------

    for file_name in sorted(file_names):

        # ---------------------------------------------------------------------
        # Read in the data from the HDF file
        # ---------------------------------------------------------------------

        # Construct file path
        file_path = os.path.join(preimages_dir, file_name)
        print(f'Plotting results for {file_path}...', end=' ', flush=True)

        # Read data from HDF file
        with h5py.File(file_path, 'r') as hdf_file:

            # Store command line arguments
            constraint = hdf_file.attrs['constraint']
            index = hdf_file.attrs['index']
    
            # Save the inputs, outputs and targets
            noise = np.array(hdf_file['noise'])
            inputs = np.array(hdf_file['inputs'])
            outputs = np.array(hdf_file['outputs'])
            targets = np.array(hdf_file['targets'])

        # ---------------------------------------------------------------------
        # Plot the data
        # ---------------------------------------------------------------------

        # Set up the subplots
        fig, axes = plt.subplots(nrows=3, sharex='col',
                                 gridspec_kw={'height_ratios': [3, 3, 2]})

        # Get residuals, that is, the component which do we need to add to
        # the noise to "get a signal" from the network
        difference = inputs - noise

        grid = np.linspace(-0.5, 0.5, 2048)

        # For the unconstrained examples, we plot the noise and the signal
        if constraint == 'gw_like' or constraint == 'minimal_perturbation':

            # Plot the (original) noise
            axes[0].plot(grid, noise[0, 2048:4096], lw=0.50)
            axes[1].plot(grid, noise[1, 2048:4096], lw=0.50)

            # Plot the additive component ("signal")
            axes[0].plot(grid, difference[0, 2048:4096], lw=0.75)
            axes[1].plot(grid, difference[1, 2048:4096], lw=0.75)

        # For the constrained examples, we only plot the effective inputs
        else:
            axes[0].plot(grid, inputs[0, 2048:4096], color='C2', lw=0.75)
            axes[1].plot(grid, inputs[1, 2048:4096], color='C2', lw=0.75)

        # Plot the network output and the optimization target
        axes[2].plot(grid, targets[:2048], color='Gray', linestyle=':',
                     dashes=(0, 2.5), lw=0.5, dash_capstyle='round')
        axes[2].plot(grid, outputs[:2048], color='C0', lw=0.75)

        # ---------------------------------------------------------------------
        # Add plot options
        # ---------------------------------------------------------------------

        # Add axes limits
        axes[0].set_ylim(-4.5, 4.5)
        axes[0].set_yticks([-3, 0, 3])
        axes[1].set_ylim(-4.5, 4.5)
        axes[1].set_yticks([-3, 0, 3])
        axes[2].set_ylim(-0.1, 1.1)

        # Add labels to the axes
        axes[0].set_ylabel('H1', labelpad=2, fontsize=6)
        axes[1].set_ylabel('L1', labelpad=2, fontsize=6)
        axes[2].set_xlabel('Time (in seconds)', labelpad=2, fontsize=6)
        axes[2].set_ylabel('CNN', labelpad=2, fontsize=6)

        # Adjust label sizes and x-limits
        for ax in axes:
            ax.set_xlim(-0.5, 0.5)
            kwargs = dict(axis='both', length=1.5, pad=2, labelsize=5)
            ax.tick_params(which='major', **kwargs)
            ax.tick_params(which='minor', **kwargs)

        # Adjust the plot size
        plt.tight_layout()
        width = 2.2 / 1.0305
        plt.gcf().set_size_inches(width, width / 2, forward=True)

        # Adjust space between the plots and align the y-labels
        plt.subplots_adjust(hspace=0)
        fig.align_ylabels(axes)

        # ---------------------------------------------------------------------
        # Save the plot
        # ---------------------------------------------------------------------

        # Create the output directory for the pre-image plots
        plots_dir = './plots/preimages'
        Path(plots_dir).mkdir(exist_ok=True, parents=True)

        # Construct the path where to save the figure
        plot_path = os.path.join(plots_dir, f'{constraint}__{index}.pdf')

        # Save the plot and clear the figure
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()

        print('Done!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print('')
    print(f'This took {time.time() - script_start:.1f} seconds in total!')
    print('')
