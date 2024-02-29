import numpy as np
import pickle
import json
import pandas as pd
from scipy.signal import find_peaks


def append_timestamps_to_predictions(predictions, session_id, path_to_timestamps):
    """
    Appends timestamps to each row in the prediction array from a provided file.

    Args:
    - predictions (numpy.array): A numpy array of shape Kx5 containing prediction data.
    - session_id (int): An integer for identifying the current session.
    - timestamps_file_path (str): Path to the file containing timestamps for each prediction.

    Returns:
    - numpy.array: An array of shape Kx6, where the last column represents the timestamps.
    """
    timestamps_file_path = f"{path_to_timestamps}/timestamps_session_{session_id}.pkl"
    # Load timestamps from the file
    with open(timestamps_file_path, "rb") as file:
        timestamps = pickle.load(file)

    # Check if the number of timestamps matches the number of predictions
    if len(timestamps) != len(predictions):
        raise ValueError("The number of timestamps does not match the number of predictions.")

    # Append timestamps as a new column to the predictions
    return np.column_stack((predictions, timestamps))


def split_predictions_to_sessions_for_all(predictions_path, subject_to_indices, session_length):
    """
        Segments prediction data into sessions for all subjects.

        Args:
            predictions_path (str): Base path for prediction files.
            subject_to_indices (dict): Subject to session index mapping.
            session_length (list): Session lengths.

        Returns:
            list: Segmented predictions per session.
        """
    pred_in_ses = []
    for subject, sessions in subject_to_indices.items():
        with open(predictions_path + f"/prediction_{subject}.pkl", "rb") as f:
            prediction = pickle.load(f)
        temp = 0
        for session in sessions:
            ses_len = session_length[session - 1]
            x_pred = prediction[temp:temp + ses_len]
            temp += ses_len
            pred_in_ses.append(x_pred)
    return pred_in_ses


def split_predictions_to_sessions(predictions, sessions, start_times_and_lengths):
    """
        Splits prediction data into sessions based on start times and lengths.

        Args:
            predictions (numpy.array): Array of prediction data.
            sessions (list): Session identifiers.
            start_times_and_lengths (dict): Start times and lengths for sessions.

        Returns:
            dict: Predictions split by session ID.
        """
    sessioned_predictions = {}
    temp = 0
    for session_id in sessions:
        session_length = start_times_and_lengths[str(session_id)][1]
        session_predictions = predictions[temp:temp + session_length]
        temp += session_length
        sessioned_predictions[session_id] = session_predictions
    return sessioned_predictions


def transform_timestamps_to_relative(data, start_time_str):
    """
    Transforms the timestamps of mm predictions in the 6th column of a numpy array to seconds since a reference start time (video's start time).

    Parameters:
    - data: A numpy array where the 6th column contains pandas datetime strings.
    - start_datetime_str: The reference start date and time as a string.

    Returns:
    - A numpy array with the timestamps transformed to seconds since the reference start time.
    """
    # Ensure the data type of the array is object to accommodate mixed types
    data = np.array(data, dtype=object)

    # Convert the 6th column to datetime format
    datetime_column = pd.to_datetime(data[:, 5])

    # Convert the reference start time to datetime
    start_time = pd.to_datetime(start_time_str)

    # Calculate the difference in seconds from the reference time
    time_differences = [(ts - start_time).total_seconds() for ts in datetime_column]

    # Update the data structure with the new time differences
    data[:, 5] = time_differences

    return data


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
