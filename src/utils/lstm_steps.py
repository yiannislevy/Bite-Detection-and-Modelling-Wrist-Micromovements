"""
This module contains functions for preprocessing data for feeding into an LSTM model, specifically
for handling and analyzing time-series data related to bite events. It includes functions for processing
CNN predictions, extracting bite data segments, and preparing these for further analysis.

Functions:
- find_index_for_timestamp(timestamp, predictions): Finds the index in the predictions array corresponding to a given timestamp.
- extract_data_for_bite_window(start_time, end_time, predictions, window_length, step): Extracts a slice of prediction data corresponding to a specific time window.
- create_positive_example_bites(predictions, labels, window_length, step_in_ms): Creates positive example data for bite events.
"""
import numpy as np
import json
import pickle


def load_bite_gt_data(bite_gt_path): # TODO: move to data_io
    """ Load the bite_gt data from the given paths. """
    bite_gt = np.load(bite_gt_path, allow_pickle=True)
    return bite_gt


def load_cnn_predictions(subject_to_indices, predictions_path): # TODO: move to data_io
    predictions = {}
    for _, sessions in subject_to_indices.items():
        for session in sessions:
            with open(f"{predictions_path}/prediction_{session}.pkl", "rb") as f:
                prediction = pickle.load(f)
            predictions[session] = prediction
    return predictions


def load_start_time(start_time_json_path, subject):
    """ Load the start time for the given subject from the JSON file. """
    with open(start_time_json_path, 'r') as file:
        start_times = json.load(file)
    return start_times[subject]


def find_index_for_timestamp(timestamp, predictions):
    return np.searchsorted(predictions[:, 5], timestamp, side='left')


def extract_data_for_bite_window(start_time, end_time, predictions, window_length, step):
    # Finding indices for start and end times in the predictions array
    start_index = find_index_for_timestamp(start_time, predictions)
    end_index = find_index_for_timestamp(end_time, predictions)

    # Extracting the relevant slice from the predictions for the duration of the bite
    bite_data = predictions[start_index:end_index]

    # Calculating the total number of samples required for a full_std_3 8.75-second window
    required_samples = int(window_length / (step / 1000))

    # Padding with zeros if the bite duration is less than 8.75 seconds
    if bite_data.shape[0] < required_samples:
        padding_length = required_samples - bite_data.shape[0]
        padding = np.zeros((padding_length, bite_data.shape[1]))
        bite_data = np.vstack((bite_data, padding))
    else:
        print(f"Error in making bite windows. Expected length: {required_samples}, actual length: {bite_data.shape[0]}.")
        print(f"Info on window:\nStart time: {start_time}\tEnd time: {end_time}.")

    return bite_data


def create_positive_example_bites(predictions, labels, window_length=8.75, step_in_ms=100):
    bite_duration_data = []
    for start_time, end_time in labels:
        # Extracting data for the bite window and padding with zeros
        bite_window_data = extract_data_for_bite_window(start_time, end_time, predictions, window_length, step_in_ms)
        bite_duration_data.append((bite_window_data, 1))

    return bite_duration_data
