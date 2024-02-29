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
