import numpy as np
import pickle

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

    # Generate timestamps
    timestamps = np.arange(start_time, start_time + num_samples * sample_step_ms / 1000, sample_step_ms / 1000)

    # Append timestamps as a new column to the predictions
    return np.column_stack((predictions, timestamps))


def split_predictions_to_sessions(predictions_path, subject_to_indices, session_length):
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
