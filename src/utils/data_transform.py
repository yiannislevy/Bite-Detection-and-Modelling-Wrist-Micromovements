import numpy as np


def sliding_window(data, window_length, step_size):
    """
    Generate windows from the data using a sliding window approach.

    Parameters:
    - data: numpy array of shape (n_samples, n_features), where n_samples is the number of samples
      and n_features is the number of features.
    - window_length: int, the length of the window.
    - step_size: int, the number of samples to step forward in the window sliding process.

    Returns:
    - windows: numpy array of shape (n_windows, window_length, n_features), where n_windows is
      the number of windows generated.
    """
    n_samples, n_features = data.shape
    n_windows = ((n_samples - window_length) // step_size) + 1
    # Initialize an empty array for the windows
    windows = np.empty((n_windows, window_length, n_features))

    for i in range(n_windows):
        start = i * step_size
        end = start + window_length
        windows[i] = data[start:end]

    return windows
