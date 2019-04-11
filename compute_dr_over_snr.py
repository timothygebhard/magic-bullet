"""
Compute the detection ratio (DR) as a function of the injection SNR
and store the results as a JSON file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import json
import numpy as np
import time


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

    # -------------------------------------------------------------------------
    # Load the trigger data
    # -------------------------------------------------------------------------

    # Open the predictions file and load the predictions
    with h5py.File('./results/found_triggers.hdf', 'r') as hdf_file:

        detected = np.array(hdf_file['injection']['detected'])
        injection_snr = np.array(hdf_file['injection']['injection_snr'])

    # -------------------------------------------------------------------------
    # Compute the detection ratio for each SNR bin
    # -------------------------------------------------------------------------

    # Construct bin edges and bins for the plot
    snr_bin_edges = np.linspace(5, 20, 31)
    snr_bins = [_ for _ in zip(snr_bin_edges[:-1], snr_bin_edges[1:])]

    # Keep track of the detection ratios and their error bars
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

    # -------------------------------------------------------------------------
    # Save results as a JSON file
    # -------------------------------------------------------------------------

    results = dict(snr_bins=snr_bins,
                   detection_ratios=detection_ratios)

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
