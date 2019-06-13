"""
Compute the detection ratio (DR) and the inverse false positive rate
(IFPR) for different post-processing parameters and save the results.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import numpy as np
import pandas as pd
import time

from utils.configfiles import get_config
from utils.evaluation import get_detections_and_fp, get_dr_and_fpr


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print('')
    print('COMPUTE DR AND IFPR FOR DIFFERENT POST-PROCESSING PARAMETERS')
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
    slack_width = config['evaluation']['slack_width']['default']
    slack_width = int(slack_width * sampling_rate)

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
    # Load the raw predictions
    # -------------------------------------------------------------------------

    # Keep data with and without an injection separate
    predictions = dict(injection=None, noise=None)

    # Open the predictions file and load the predictions
    print('Loading predictions...', end=' ', flush=True)
    with h5py.File('./results/predictions_testing.hdf', 'r') as hdf_file:
        predictions['injection'] = np.array(hdf_file['injection'])
        predictions['noise'] = np.array(hdf_file['noise'])
    print('Done!')
    print('')

    # -------------------------------------------------------------------------
    # Compute the DR and iFPR for different post-processing parameters
    # -------------------------------------------------------------------------

    # Define the thresholds and window sizes which we want to evaluate
    thresholds = (0.1, 0.3, 0.5, 0.7, 0.9)
    window_sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256)

    # Keep track of the results
    results = list()

    # Post-process the results for every parameter combination
    print('Evaluating performance for different parameter combinations:\n')
    for threshold in thresholds:
        for window_size in window_sizes:

            # Keep track of the numbers of false positives
            false_positives = dict(injection=list(), noise=list())
    
            # Process examples containing an injection
            detected_flags, false_positives['injection'] = \
                get_detections_and_fp(predictions=predictions['injection'],
                                      event_index=event_index,
                                      slack_width=slack_width,
                                      window_size=window_size,
                                      threshold=threshold)

            # Process examples containing only noise
            _, false_positives['noise'] = \
                get_detections_and_fp(predictions=predictions['noise'],
                                      event_index=None,
                                      slack_width=None)

            # Compute the global figures of merit
            figures_of_merit = get_dr_and_fpr(detected_flags=detected_flags,
                                              false_positives=false_positives,
                                              sample_length=sample_length)

            # Define shortcuts to figures of merit
            detection_ratio = figures_of_merit['detection_ratio']
            false_positive_ratio = figures_of_merit['false_positive_ratio']
            false_positive_rate = figures_of_merit['false_positive_rate']

            # Store the results so that we can save them for plotting
            results += [{'det_ratio': detection_ratio,
                         'fp_rate': false_positive_ratio,
                         'ifp_rate': 1.0 / false_positive_rate,
                         'fp_ratio': false_positive_ratio,
                         'threshold': threshold,
                         'window_size': window_size}]

            # Print results to command line
            print(f'THRESHOLD = {threshold}, '
                  f'WINDOW_SIZE = {window_size:3}'
                  f'   =>   '
                  f'DR = {detection_ratio:5.4f}, '
                  f'iFPR = {1.0 / false_positive_rate:9.4f}s')
        print('')

    # -------------------------------------------------------------------------
    # Create pandas data frame from results and save them as JSON
    # -------------------------------------------------------------------------

    # Create the data frame with all results and save it as a JSON file
    print('Saving results to JSON file...', end=' ', flush=True)
    dataframe = pd.DataFrame(results)
    dataframe.to_json(path_or_buf='./results/dr_over_ifpr.json',
                      orient='records',
                      lines=True)
    print('Done!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print('')
    print(f'This took {time.time() - script_start:.1f} seconds!')
    print('')
