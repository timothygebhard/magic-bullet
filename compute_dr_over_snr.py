"""
Compute the detection ratio (DR) as a function of the injection SNR
(and the slack_width parameter) and store the results as a JSON file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import json
import numpy as np
import time

from utils.configfiles import get_config


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print('')
    print('COMPUTE DETECTION RATIO OVER INJECTION SNR')
    print('')

    # Start the stopwatch
    script_start = time.time()

    # Get the global config
    config = get_config()

    # -------------------------------------------------------------------------
    # Define shortcuts and initialize result variables
    # -------------------------------------------------------------------------

    # Get the sampling rate of the examples (in Hertz)
    sampling_rate = config['static_args']['sampling_rate']

    # Get the "slack widths", that is, the sizes of the interval around the
    # ground truth injection time in which a detection will still be counted.
    # In the research paper, this parameter is called "\Delta t".
    # NOTE: The size is given in seconds and must be converted to time steps
    # by multiplying with the sampling rate.
    slack_widths = config['evaluation']['slack_width']['all']
    slack_widths = sorted(np.unique(np.array(slack_widths).astype(float)))

    # Construct SNR bin edges and bins for the plot
    snr_bin_edges = np.linspace(5, 20, 31)
    snr_bins = [_ for _ in zip(snr_bin_edges[:-1], snr_bin_edges[1:])]

    # Collect all results so that we can write them into a single JSON file
    results = dict(slack_widths=slack_widths,
                   snr_bins=snr_bins,
                   detection_ratios=dict())

    # -------------------------------------------------------------------------
    # Compute the detection ratio for different slack_widths
    # -------------------------------------------------------------------------

    for slack_width in slack_widths:

        # Define a shortcut for the string representation of the slack_width
        sw = str(slack_width)

        # ---------------------------------------------------------------------
        # Load the trigger data
        # ---------------------------------------------------------------------

        # Open the predictions file and load the predictions
        with h5py.File('./results/found_triggers.hdf', 'r') as hdf_file:
            detected = np.array(hdf_file[sw]['injection']['detected'])
            injection_snr = \
                np.array(hdf_file[sw]['injection']['injection_snr'])

        # ---------------------------------------------------------------------
        # Compute the detection ratio for each SNR bin
        # ---------------------------------------------------------------------

        # Keep track of the detection ratios
        detection_ratios = []

        # Loop over all SNR bins to compute the DR
        for snr_bin in snr_bins:

            # Get upper and lower bound for the current SNR bin
            lower, upper = snr_bin
    
            # Get the indices of the injections that belong to this bin
            idx = np.where(np.logical_and(lower <= injection_snr,
                                          injection_snr <= upper))[0]
    
            # Calculate the detection ratio for the ensemble and store
            detection_ratios.append(np.mean(detected[idx]))

        # Store the detection ratios for this slack_width value
        results['detection_ratios'][sw] = detection_ratios

    # -------------------------------------------------------------------------
    # Save results as a JSON file
    # -------------------------------------------------------------------------

    print('Saving results to JSON file...', end=' ', flush=True)
    with open('./results/dr_over_snr.json', 'w') as json_file:
        json.dump(results, json_file, sort_keys=True, indent=2)
    print('Done!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print('')
    print(f'This took {time.time() - script_start:.1f} seconds!')
    print('')
