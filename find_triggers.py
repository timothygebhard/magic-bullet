"""
Find triggers in post-processed model predictions on testing dataset.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import numpy as np
import os
import time

from utils.configfiles import get_config
from utils.evaluation import get_detections_and_fp, get_dr_and_fpr, \
    get_avg_event_time_deviation


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print('')
    print('COUNT DETECTIONS AND FALSE POSITIVES IN PREDICTIONS')
    print('')

    # Start the stopwatch
    script_start = time.time()

    # Get the global config
    config = get_config()

    # -------------------------------------------------------------------------
    # Get a few constants from global config and compute implicit parameters
    # -------------------------------------------------------------------------

    # Get the length of the examples (in seconds)
    sample_length = config['static_args']['sample_length']

    # Get the sampling rate of the examples (in Hertz)
    sampling_rate = config['static_args']['sampling_rate']

    # Get the "slack width", that is, the size of the interval around the
    # ground truth injection time in which a detection will still be counted,
    # and convert from seconds to time steps
    slack_width = int(config['evaluation']['slack_width'] * sampling_rate)

    # Get the number of seconds before the event in the examples, and the
    # size of the receptive field of the model that was used
    seconds_before_event = config['static_args']['seconds_before_event']
    receptive_field_size = config['model']['receptive_field_size']

    # Compute the time step at which the coalescence should be located in the
    # model output for examples that contain an injection (taking into account
    # that this prediction time series is shorter than the actual input by
    # half the receptive field of the network)
    event_index = int(seconds_before_event * sampling_rate -
                      receptive_field_size / 2 + 1)

    # -------------------------------------------------------------------------
    # Load the predictions on the test dataset (and the injection SNRs)
    # -------------------------------------------------------------------------

    # Keep data with and without an injection separate
    predictions = dict(injection=None, noise=None)

    # Open the predictions file and load the predictions
    with h5py.File('./results/predictions_testing.hdf', 'r') as hdf_file:
        predictions['injection'] = np.array(hdf_file['injection'])
        predictions['noise'] = np.array(hdf_file['noise'])

    # Load the injection SNRs for the examples containing injections
    with h5py.File(config['data']['testing'], 'r') as hdf_file:
        inj_snr = np.array(hdf_file['/injection_parameters/injection_snr'])

    # -------------------------------------------------------------------------
    # Check if an injection was recovered and count the false positives
    # -------------------------------------------------------------------------

    false_positives = dict(injection=list(), noise=list())

    # Process examples containing an injection
    print(f'Processing examples of type "injection"...', end=' ', flush=True)
    detected_flags, false_positives['injection'] = \
        get_detections_and_fp(predictions=predictions['injection'],
                              event_index=event_index,
                              slack_width=slack_width)
    print('Done!')

    # Process examples containing only noise
    print(f'Processing examples of type "noise"...', end=' ', flush=True)
    _, false_positives['noise'] = \
        get_detections_and_fp(predictions=predictions['noise'],
                              event_index=None,
                              slack_width=None)
    print('Done!')

    # -------------------------------------------------------------------------
    # Compute global values for the DR and the (i)FPR
    # -------------------------------------------------------------------------

    # Compute the global figures of merit
    figures_of_merit = get_dr_and_fpr(detected_flags=detected_flags,
                                      false_positives=false_positives,
                                      sample_length=sample_length)
    
    # Define shortcuts to figures of merit
    detection_ratio = figures_of_merit['detection_ratio']
    false_positive_ratio = figures_of_merit['false_positive_ratio']
    false_positive_rate = figures_of_merit['false_positive_rate']

    # Print these results to the command line
    print('\nGLOBAL FIGURES OF MERIT:')
    print(f'Detection Ratio:               {detection_ratio:9.4f}')
    print(f'False Positive Ratio:          {false_positive_ratio:9.4f}')
    print(f'False Positive Rate:           {false_positive_rate:9.4f}/s')
    print(f'Inverse False Positive Rate:   {1.0/false_positive_rate:9.4f}s')
    print('')

    # -------------------------------------------------------------------------
    # Compute average distance between predicted event time and ground truth
    # -------------------------------------------------------------------------

    mean, std = \
        get_avg_event_time_deviation(predictions=predictions['injection'],
                                     event_index=event_index,
                                     slack_width=slack_width,
                                     sampling_rate=sampling_rate)
    print(f'mean(abs(t_pred - t_true)):    {mean:.4f}s +- {std:.4f}s\n')

    # -------------------------------------------------------------------------
    # Save results to an HDF file
    # -------------------------------------------------------------------------

    # Construct path to the output file containing detections and false ps
    results_dir = './results/'
    output_file = os.path.join(results_dir, 'found_triggers.hdf')

    print('Saving results to HDF file...', end=' ', flush=True)
    with h5py.File(output_file, 'w') as hdf_file:

        # Create groups for the results
        hdf_file.create_group(name='injection')
        hdf_file.create_group(name='noise')

        # Store results for examples containing an injection
        hdf_file['injection'].create_dataset(name='detected',
                                             data=detected_flags)
        hdf_file['injection'].create_dataset(name='false_positives',
                                             data=false_positives['injection'])
        hdf_file['injection'].create_dataset(name='injection_snr',
                                             data=inj_snr)

        # Store results for examples not containing an injection
        hdf_file['noise'].create_dataset(name='false_positives',
                                         data=false_positives['noise'])
    print('Done!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print('')
    print(f'This took {time.time() - script_start:.1f} seconds!')
    print('')
