import numpy as np
import json


def load_data(predictions_path, bite_gt_path, subject):
    """ Load the predictions and bite_gt data from the given paths. """
    predictions = np.load(predictions_path, allow_pickle=True)
    bite_gt = np.load(bite_gt_path, allow_pickle=True)
    return predictions, bite_gt[int(subject)-1]


def load_start_time(start_time_json_path, subject):
    """ Load the start time for the given subject from the JSON file. """
    with open(start_time_json_path, 'r') as file:
        start_times = json.load(file)
    return start_times[subject]


def create_windows(predictions, start_time, window_length=35, step_size=0.01):
    """
    Create 35-sample windows from the predictions.

    Parameters:
    - predictions: A list of numpy arrays, each representing predictions for a 20ms window.
    - window_length: The length of each window (default is 35).
    - step_size: The time step size in seconds (default is 0.01s for 10ms).
    - start_time: The start time of the first prediction window.

    Returns:
    - windows_array: An array of 35-sample windows.
    - timestamps: The timestamps for each window.
    """
    # Concatenate predictions to form a continuous sequence
    concatenated_predictions = np.vstack(predictions)

    # Calculate timestamps for each prediction
    timestamps = np.arange(concatenated_predictions.shape[0]) * step_size + start_time

    # Generate windows
    windows = []
    for start_idx in range(len(concatenated_predictions) - window_length + 1):
        window = concatenated_predictions[start_idx:start_idx + window_length, :]
        windows.append(window)

    windows_array = np.array(windows)
    return windows_array, timestamps
