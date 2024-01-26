import numpy as np


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
    return timestamps

# np.column_stack((predictions, timestamps)),