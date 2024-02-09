"""
This module contains functions for preprocessing data for feeding into an LSTM model, specifically
for handling and analyzing time-series data related to bite events. It includes functions for processing
CNN predictions, extracting bite data segments, and preparing these for further analysis.

Functions:
- find_index_for_timestamp(timestamp, predictions): Finds the index in the predictions array corresponding to a given timestamp.
- extract_data_for_bite_window(start_time, end_time, predictions, window_length, step): Extracts a slice of prediction data corresponding to a specific time window.
- create_positive_example_bites(predictions, labels, window_length, sample_rate): Creates positive example data for bite events with label.
- create_negative_example_bites(predictions, labels, window_length, sample_rate): Created negative example data for bite events with label.
"""
import numpy as np


def find_index_for_timestamp(timestamp, predictions):
    return np.searchsorted(predictions[:, 5], timestamp, side='left')


def extract_data_for_bite_window(start_time, end_time, predictions, window_length, sample_rate):
    # Finding indices for start and end times in the predictions array
    start_index = find_index_for_timestamp(start_time, predictions)
    end_index = find_index_for_timestamp(end_time, predictions)

    # Calculating the total number of samples required for a full_std_3_old 9-second window
    required_samples = int(window_length / sample_rate)

    # Extracting the relevant slice from the predictions for the duration of the bite
    bite_data = predictions[start_index:end_index]

    # Padding with zeros if the bite duration is less than 9 seconds
    if bite_data.shape[0] <= required_samples:
        padding_length = required_samples - bite_data.shape[0]
        padding = np.zeros((padding_length, bite_data.shape[1]))
        bite_data = np.vstack((bite_data, padding))
    else:
        print(f"Error in making bite windows. Expected length: {required_samples}, actual length: {bite_data.shape[0]}.")
        print(f"Info on window:\t\tStart time: {start_index}) {start_time}\tEnd time: {end_index}) {end_time}.\n\n")
        return None

    return bite_data


def create_positive_example_bites(predictions, labels, window_length=9, sample_rate=0.1):
    # Note for future reference: this function allows for up to 1 sample tolerance in bite length above the limit only IF
    # it is still within the (window_length / sample_rate) window.
    # True example: FIC's bite_gt[20][19] (session 21, bite 20) -> bite's length = 9.007. This passes as a positive
    # example since the raw imu's data --> prediction data's timestamps allow for precisely 90 samples in that timeframe.
    # Same as bite_gt[17][24] -> bite's length is 9.045 and prediction data fill the entirety of the window's 90 samples.
    bite_duration_data = []
    removed_bite = 0
    for start_time, end_time in labels:
        # Extracting data for the bite window and padding with zeros
        bite_window_data = extract_data_for_bite_window(start_time, end_time, predictions, window_length, sample_rate)
        if bite_window_data is not None:
            # Only append if bite_window_data is not None
            bite_duration_data.append((bite_window_data, 1))
        else:
            removed_bite += 1
    return bite_duration_data, removed_bite


def create_negative_example_bites(predictions, bite_times, window_length=9, sample_rate=0.1):
    """
    Generate negative example windows for non-bite durations.

    Parameters:
    - predictions: numpy array with shape (N, 6) where N is the number of predictions.
    - bite_times: numpy array with shape (M, 2) where M is the number of bite events,
                  each row containing the start and end times of a bite.
    - window_length: int, length of each window in seconds.
    - sample_rate: float, the time in seconds between each prediction sample.

    Returns:
    - A list of numpy arrays, each representing a negative example window, with its label.
    """
    negative_windows = []
    sample_step = int(window_length / sample_rate)  # Number of samples in a window
    step_size = 1  # Sliding window step in samples (0.1 second)

    # Add a virtual bite at the beginning and end of the session to simplify logic
    virtual_start_bite = np.array([[0, predictions[0, 5] - sample_rate]])
    virtual_end_bite = np.array([[predictions[-1, 5] + sample_rate, predictions[-1, 5] + sample_rate + window_length]])
    all_bites = np.vstack((virtual_start_bite, bite_times, virtual_end_bite))

    for i in range(len(all_bites) - 1):
        start_time = all_bites[i, 1]
        end_time = all_bites[i + 1, 0]

        start_index = find_index_for_timestamp(start_time, predictions)
        end_index = find_index_for_timestamp(end_time, predictions)

        if end_index - start_index <= sample_step:
            # Shorter or equal to 9 seconds, create one window and pad if necessary
            window_data = predictions[start_index:end_index]
            padding_length = sample_step - window_data.shape[0]
            if padding_length > 0:
                padding = np.zeros((padding_length, window_data.shape[1]))
                window_data = np.vstack((window_data, padding))
            negative_windows.append((window_data, 0)) # Add label
        else:
            # Longer than 9 seconds, use sliding window approach
            num_windows = (end_index - start_index - sample_step) // step_size + 1
            for j in range(num_windows):
                window_start_index = start_index + j * step_size
                window_end_index = window_start_index + sample_step
                window_data = predictions[window_start_index:window_end_index]
                negative_windows.append((window_data, 0)) # Add label

    return negative_windows
