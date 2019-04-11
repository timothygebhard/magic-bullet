"""
Provide tools and metrics for the evaluation of the network performance.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from typing import Dict, Optional, Tuple


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def find_binary_peaks(array: np.ndarray) -> np.ndarray:
    """
    Take a numpy array whose values must be all in {0, 1}, and find
    the centers of the continuous stretches of 1s.

    Example: [0, 0, 0, 0, 0, 0, 1, 0, 1, 1] -> [6.0, 8.5]

    Args:
        array: A 1D numpy array with values only in {0, 1}.

    Returns:
        A numpy array containing the centers of the continuous
        stretches of 1s in the input `array`.
    """

    # Pad the input array with zeros left and right
    array = np.hstack(([0], array, [0]))

    # Compute the differences between neighboring elements: a[n+1] - a[n]
    differences = np.diff(array.astype(int))

    # Find the starts and ends of the intervals of ones
    starts = np.where(differences == 1)[0]
    ends = np.where(differences == -1)[0]

    # Compute the interval centers as the means of their start and end, and
    # subtract a constant offset of 0.5 for a more intuitive interpretation
    return np.mean(np.vstack((starts, ends)), axis=0) - 0.5


def postprocess_prediction(prediction: np.ndarray,
                           threshold: float = 0.5,
                           window_size: int = 256) -> np.ndarray:
    """
    Apply post-processing steps (smoothing, thresholding) to a "raw"
    network prediction.

    Args:
        prediction: A 1D numpy array with values in (0, 1) containing
            the raw prediction of the fully convolutional neural
            network for a given example.
        threshold: Threshold for rounding smoothed values to 1.
        window_size: Size of the rectangular window that is used to
            compute a rolling average to smooth the prediction.

    Returns:
        A numpy array with the same length as the input containing the
        smoothed and thresholded network prediction.
    """

    # Make sure the input has the correct dimensionality
    if prediction.ndim != 1:
        raise ValueError(f'Input should be 1D, but is {prediction.ndim}D!')

    # Construct a rectangular window and smooth the output
    kernel = np.full(window_size, 1.0 / window_size)
    output = np.convolve(prediction, kernel, mode='same')

    # Round the raw outputs based on the provided threshold
    output = (output >= threshold).astype(np.float)

    return output


def get_detections_and_fp(predictions: np.ndarray,
                          event_index: Optional[int],
                          slack_width: Optional[int],
                          window_size: int = 256,
                          threshold: float = 0.5) -> Tuple[np.ndarray,
                                                           np.ndarray]:
    """
    For every raw (i.e., not yet pre-processed) prediction in the
    `predictions` array, check whether a possible injection was found,
    and return the number of false positives.

    Args:
        predictions: A 2D numpy array containing (raw) predictions
            made by the fully convolutional neural network.
        event_index: Index of the time step which corresponds to the
            ground truth injection time. If `None` is passed, it is
            assumed that the samples do not contain an injection, that
            is, every trigger is a false positive.
        slack_width: Maximum number of time steps which the prediction
            may be off from `event_index` to still be counted as a
            recovered injection.
        window_size: Size of the smoothing window that is used to
            post-process the predictions.
        threshold: Threshold that is used to round or "binarize" the
            smoothed predictions.

    Returns:
        A tuple `(detected_flags, false_positives)` which contains for
        every prediction a boolean flag whether or not the injection
        was recovered (for noise-only samples, this list is empty), as
        well as a numpy array containing the number of false positives
        for each example.
    """

    # Keep track of the detections and false positives
    detected_flags = list()
    false_positives = list()

    # For each prediction, check if an injection was recovered, and
    # count the number of false positives produced
    for prediction in predictions:

        # Apply post-processing steps (smoothing, thresholding)
        postprocessed = postprocess_prediction(prediction=prediction,
                                               window_size=window_size,
                                               threshold=threshold)

        # Find all the peaks in the post-processed prediction
        all_peaks = find_binary_peaks(postprocessed)

        # If we are looking at a noise-only sample, every peak is a
        # false positive, and we only need to count these
        if event_index is None:
            false_positives.append(len(all_peaks))
            continue

        # Otherwise, we need to check if the injection was recovered
        else:

            # Check if any of the peaks is within the acceptance interval
            detected = any([np.abs(event_index - peak) < slack_width
                            for peak in all_peaks])
            detected_flags.append(detected)
        
            # Compute and store the number of false positives
            false_positives.append(len(all_peaks) - detected)

    return np.array(detected_flags), np.array(false_positives)


def get_dr_and_fpr(detected_flags: np.ndarray,
                   false_positives: Dict[str, np.ndarray],
                   sample_length: float) -> Dict[str, float]:
    """
    Compute the detection ratio (DR) and the false positive rate (FPR),
    as well as the false positive ratio, from the given trigger data
    in `detected_flags` and `false_positives`.

    Args:
        detected_flags: A numpy array whose length matches the number
            of examples containing an injection, and which for each
            example specifies if the injection was recovered (value 1)
            or not (value 0).
        false_positives: A dictionary with keys `{'injection', 'noise'}`
            which contains the number of false positives for each
            example (both with and without an injection).
        sample_length: The length (in seconds) of the original samples.
            This is needed to compute the (average) false positive rate.

    Returns:
        A `dict` with keys `{'detection_ratio', 'false_positive_ratio',
        'false_positive_rate'}`, which are the metrics computed from
        the input data.
    """

    # Define some shortcuts for calculating our metrics
    n_samples_total = sum([len(_) for _ in false_positives.values()])
    n_false_positives = sum([np.sum(_) for _ in false_positives.values()])
    n_all_triggers = sum(detected_flags) + n_false_positives
    total_rec_time = sample_length * n_samples_total
    
    # Compute the detection ratio and the false positive ratio / rate
    detection_ratio = np.mean(detected_flags)
    false_positive_ratio = n_false_positives / n_all_triggers
    false_positive_rate = n_false_positives / total_rec_time

    # Return the results as a dictionary
    return dict(detection_ratio=detection_ratio,
                false_positive_ratio=false_positive_ratio,
                false_positive_rate=false_positive_rate)


def get_avg_event_time_deviation(predictions: np.ndarray,
                                 event_index: int,
                                 slack_width: int,
                                 sampling_rate: int,
                                 window_size: int = 256,
                                 threshold: float = 0.5) -> Tuple[float,
                                                                  float]:
    """
    For the predictions containing an injection, find the mean and the
    standard deviation of the difference between the predicted event
    time and the ground truth event time.

    Args:
        predictions: A 2D numpy array containing (raw) predictions
            made by the fully convolutional neural network for the
            examples that do contain an injection.
        event_index: Index of the time step which corresponds to the
            ground truth injection time.
        slack_width: Maximum number of time steps which the prediction
            may be off from `event_index` to still be counted as a
            recovered injection.
        sampling_rate: Sampling rate of the predictions (to convert
            between time steps and seconds).
        window_size: Size of the smoothing window that is used to
            post-process the predictions.
        threshold: Threshold that is used to round or "binarize" the
            smoothed predictions.

    Returns:
        A dictionary `(mean, std)` containing the mean and standard
        deviation of the deviation of the predicted event time from
        the ground truth event time (if the injection was recovered).
    """

    # Keep track of the time differences between prediction and ground truth
    deviations = []

    # For each prediction, check if an injection was recovered
    for prediction in predictions:
    
        # Apply post-processing steps (smoothing, thresholding)
        postprocessed = postprocess_prediction(prediction=prediction,
                                               window_size=window_size,
                                               threshold=threshold)
    
        # Find all the peaks in the post-processed prediction
        all_peaks = find_binary_peaks(postprocessed)
        
        # Get the time difference to the ground truth for every peak
        differences = [np.abs(event_index - peak) for peak in all_peaks]

        # We only look at the peak that is closest to the ground truth
        # injection time. If this one is within the acceptance region, we
        # store its difference (in seconds); otherwise---if it is not
        # recovered---we just store NaN so that we can exclude it when
        # computing the mean and the standard deviation.
        if differences and min(differences) < slack_width:
            deviations.append(min(differences) / sampling_rate)
        else:
            deviations.append(np.nan)

    # Convert deviations to a numpy array and compute mean and std while
    # ignoring all examples where the injection as not recovered
    deviations = np.array(deviations)
    mean = float(np.nanmean(deviations))
    std = float(np.nanstd(deviations))

    return mean, std
