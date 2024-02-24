import numpy as np
from scipy.signal import find_peaks


def resample_ground_truth(ground_truth):
    """
    Resample ground truth data to match prediction sampling rate.

    Parameters:
    - ground_truth: numpy.ndarray, ground truth data.

    Returns:
    - times: numpy.ndarray, array of time points.
    - cumulative_weight: numpy.ndarray, cumulative weight over time.
    """
    total_weight = ground_truth[:, 0].sum()
    times = np.arange(ground_truth[0, 1], ground_truth[-1, 2], 0.1)  # 0.1 second intervals
    cumulative_weight = np.full(times.shape, total_weight)

    for bite in ground_truth:
        weight, start, end = bite
        total_weight -= weight
        cumulative_weight[times >= end] = total_weight

    return times, cumulative_weight


def count_bites(bite_probabilities, threshold=0.75, min_distance=20):
    """
    Count the number of bites based on the provided probability threshold and minimum distance between peaks.

    Parameters:
    - bite_probabilities: 1D numpy array containing the probability of a bite event at each time point.
    - threshold: float, the minimum height of peaks to be considered as bites.
    - min_distance: int, the minimum number of samples between consecutive peaks.

    Returns:
    - num_bites: int, the number of bites detected.
    - peaks: array, the indices of the peaks that were identified as bites.
    """
    peaks, _ = find_peaks(bite_probabilities, height=threshold, distance=min_distance)
    num_bites = len(peaks)
    return num_bites, peaks
