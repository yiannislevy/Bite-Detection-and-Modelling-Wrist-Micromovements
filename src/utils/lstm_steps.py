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


# def create_windows(predictions, start_time, window_length=35, step_size=0.01):
#     """
#     Create 35-sample windows from the predictions.
#
#     Parameters:
#     - predictions: A list of numpy arrays, each representing predictions for a 20ms window.
#     - window_length: The length of each window (default is 35).
#     - step_size: The time step size in seconds (default is 0.01s for 10ms).
#     - start_time: The start time of the first prediction window.
#
#     Returns:
#     - windows_array: An array of 35-sample windows.
#     - timestamps: The timestamps for each window.
#     """
#     # Concatenate predictions to form a continuous sequence
#     concatenated_predictions = np.vstack(predictions)
#
#     # Calculate timestamps for each prediction
#     timestamps = np.arange(concatenated_predictions.shape[0]) * step_size + start_time
#
#     # Generate windows
#     windows = []
#     for start_idx in range(len(concatenated_predictions) - window_length + 1):
#         window = concatenated_predictions[start_idx:start_idx + window_length, :]
#         windows.append(window)
#
#     windows_array = np.array(windows)
#     return windows_array, timestamps


def append_timestamps_to_predictions(predictions, start_time, sample_step_ms=10):
    """
    Appends timestamps to each row in the prediction array.

    Args:
    - predictions (numpy.array): A numpy array of shape Kx5 containing prediction data.
    - start_time (float): The start time in seconds for the first prediction.
    - sample_step_ms (float): The time step in milliseconds between each prediction sample. Default is 10ms.

    Returns:
    - numpy.array: An array of shape Kx6, where the last column represents the timestamps.
    """
    # Number of samples in the prediction array
    num_samples = len(predictions)
    predictions = np.vstack(predictions)

    # Generate timestamps
    timestamps = np.arange(start_time, start_time + num_samples * sample_step_ms / 1000, sample_step_ms / 1000)

    # Append timestamps as a new column to the predictions
    return np.column_stack((predictions, timestamps))


def find_index_for_timestamp(timestamp, predictions):
    return np.searchsorted(predictions[:, 5], timestamp, side='left')


def extract_data_for_bite_window(start_time, end_time, predictions, window_length, step):
    # Finding indices for start and end times in the predictions array
    start_index = find_index_for_timestamp(start_time, predictions)
    end_index = find_index_for_timestamp(end_time, predictions)

    # Extracting the relevant slice from the predictions for the duration of the bite
    bite_data = predictions[start_index:end_index]

    # Calculating the total number of samples required for a full 8.75-second window
    required_samples = int(window_length / (step / 1000))

    # Padding with zeros if the bite duration is less than 8.75 seconds
    if bite_data.shape[0] < required_samples:
        padding_length = required_samples - bite_data.shape[0]
        padding = np.zeros((padding_length, bite_data.shape[1]))
        bite_data = np.vstack((bite_data, padding))

    return bite_data


def create_positive_example_bites(predictions, labels, start_time, window_length=8.75, step_in_ms=10):
    bite_duration_data = []
    predictions = append_timestamps_to_predictions(predictions, start_time, step_in_ms)
    for start_time, end_time in labels:
        # Extracting data for the bite window and padding with zeros
        bite_window_data = extract_data_for_bite_window(start_time, end_time, predictions, window_length, step_in_ms)
        bite_duration_data.append((bite_window_data, 1))

    return bite_duration_data
