import numpy as np
import pickle
import json


def load_start_time(start_time_json_path, session): # TODO: move to data_io
    """ Load the start time for the given subject from the JSON file. """

    with open(start_time_json_path, 'r') as file:
        start_times = json.load(file)
    return start_times[f"{session}"][0]


def append_timestamps_to_predictions(predictions, session_id):
    """
    Appends timestamps to each row in the prediction array from a provided file.

    Args:
    - predictions (numpy.array): A numpy array of shape Kx5 containing prediction data.
    - timestamps_file_path (str): Path to the file containing timestamps for each prediction.

    Returns:
    - numpy.array: An array of shape Kx6, where the last column represents the timestamps.
    """
    timestamps_file_path = f"../data/ProcessedSubjects/MajorityLabel/sessions/timestamps/timestamps_session_{session_id}.pkl"
    # Load timestamps from the file
    with open(timestamps_file_path, "rb") as file:
        timestamps = pickle.load(file)

    # Check if the number of timestamps matches the number of predictions
    if len(timestamps) != len(predictions):
        raise ValueError("The number of timestamps does not match the number of predictions.")

    # Append timestamps as a new column to the predictions
    return np.column_stack((predictions, timestamps))


def split_predictions_to_sessions_for_all(predictions_path, subject_to_indices, session_length):
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
    sessioned_predictions = {}
    temp = 0
    for session_id in sessions:
        session_length = start_times_and_lengths[str(session_id)][1]
        session_predictions = predictions[temp:temp + session_length]
        temp += session_length
        sessioned_predictions[session_id] = session_predictions
    return sessioned_predictions
