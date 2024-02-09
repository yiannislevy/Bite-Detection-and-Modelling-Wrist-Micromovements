import numpy as np


def create_fixed_length_windows_strict(predictions, bite_events, window_length=9.0, sample_rate=0.1):
    """
    Create fixed length windows for bite events from predictions with strict gap handling.

    Parameters:
    - predictions: numpy array of shape (N, 6) where each row is a prediction with timestamp as the last column.
    - bite_events: numpy array of shape (M, 2) where each row represents a bite event with start and end timestamps.
    - window_length: float, the fixed length of the window in seconds.
    - sample_rate: float, the expected interval between predictions in seconds.

    Returns:
    - windows: List of numpy arrays, each representing a fixed length window of predictions for a bite event.
    """
    windows = []

    for start_time, end_time in bite_events:
        # Determine the range of timestamps that should be covered by this bite event
        expected_timestamps = np.arange(start_time, end_time, sample_rate)

        # Filter predictions that fall within the bite event timeframe
        relevant_preds = predictions[(predictions[:, -1] >= start_time) & (predictions[:, -1] < end_time)]

        if len(relevant_preds) != len(expected_timestamps):
            print(
                f"Bite event from {start_time} to {end_time} discarded. Expected {len(expected_timestamps)} preds, got {len(relevant_preds)}.")
            continue

        # Prepare the window with padding if necessary
        num_samples = int(window_length / sample_rate)
        window = np.zeros((num_samples, predictions.shape[1]))  # Initialize window with zeros

        window[:len(relevant_preds), :] = relevant_preds

        # Add the prepared window to the list
        windows.append(window)

    return windows


def create_fixed_length_windows_strict_v2(predictions, bite_events, window_length=9.0, sample_rate=0.1):
    """
    Adjusted function to create fixed length windows for bite events from predictions,
    considering the actual practice of taking the first sample after the bite starts,
    and the last before it ends, with strict gap handling.

    Parameters:
    - predictions: numpy array of shape (N, 6) where each row is a prediction with timestamp as the last column.
    - bite_events: numpy array of shape (M, 2) where each row represents a bite event with start and end timestamps.
    - window_length: float, the fixed length of the window in seconds.
    - sample_rate: float, the expected interval between predictions in seconds.

    Returns:
    - windows: List of numpy arrays, each representing a fixed length window of predictions for a bite event.
    """
    windows = []

    for start_time, end_time in bite_events:
        # Filter predictions that fall within the bite event timeframe
        relevant_preds = predictions[(predictions[:, -1] >= start_time) & (predictions[:, -1] < end_time)]

        # Instead of matching expected timestamps, ensure at least one sample exists within the timeframe
        if len(relevant_preds) < 1:
            continue  # Skip this bite due to no predictions within the timeframe

        # Ensure the duration between the first and last relevant prediction is continuous without significant gaps
        first_pred_time = relevant_preds[0, -1]
        last_pred_time = relevant_preds[-1, -1]
        expected_duration = last_pred_time - first_pred_time
        expected_samples = expected_duration // sample_rate + 1  # How many samples we'd expect in a perfect scenario

        if len(relevant_preds) != expected_samples:
            print(f"Skipping bite event from {start_time} to {end_time}: gap detected.")
            continue  # Skip due to gap within predictions

        # Prepare the window with padding if necessary
        num_samples = int(window_length / sample_rate)
        window = np.zeros((num_samples, predictions.shape[1]))  # Initialize window with zeros

        fill_length = min(len(relevant_preds), num_samples)
        window[:fill_length, :] = relevant_preds

        windows.append(window)

    return windows


def create_windows_and_verify_intervals(predictions, bite_events, window_length=9.0, sample_rate=0.1):
    """
    Create fixed length windows for bite events from predictions and verify sample intervals.

    Parameters:
    - predictions: numpy array of shape (N, 6) where each row is a prediction with timestamp as the last column.
    - bite_events: numpy array of shape (M, 2) where each row represents a bite event with start and end timestamps.
    - window_length: float, the fixed length of the window in seconds.
    - sample_rate: float, the expected interval between predictions in seconds.

    Returns:
    - valid_windows: List of numpy arrays, each a valid fixed length window of predictions for a bite event with correct intervals.
    """
    valid_windows = []

    for start_time, end_time in bite_events:
        # Filter predictions that fall within the bite event timeframe
        relevant_preds = predictions[(predictions[:, -1] >= start_time) & (predictions[:, -1] < end_time)]

        # Prepare the window
        num_samples = int(window_length / sample_rate)
        window = np.zeros((num_samples, predictions.shape[1]))  # Initialize window with zeros

        # Fill the window with available predictions
        fill_length = min(len(relevant_preds), num_samples)
        window[:fill_length, :] = relevant_preds[:fill_length, :]

        # Verify intervals between each sample in the filled part of the window
        if fill_length > 1:  # Need at least two points to check intervals
            time_diffs = np.diff(window[:fill_length, -1])
            if np.all(np.isclose(time_diffs, sample_rate, atol=1e-2)):  # Allow for a small tolerance in timing
                valid_windows.append(window)
            # Else, the window is discarded due to incorrect intervals
        else:
            # If only one prediction or none, discard this window
            continue

    return valid_windows


def create_fixed_length_windows(predictions, bite_events, window_length=9.0):
    """
    Create fixed length windows for bite events from predictions.

    Parameters:
    - predictions: numpy array of shape (N, 6) where each row is a prediction with timestamp as the last column.
    - bite_events: numpy array of shape (M, 2) where each row represents a bite event with start and end timestamps.
    - window_length: float, the fixed length of the window in seconds.

    Returns:
    - windows: List of numpy arrays, each representing a fixed length window of predictions for a bite event.
    """
    windows = []
    sample_rate = 0.1  # Assuming a sample rate based on the description

    for start_time, end_time in bite_events:
        # Find predictions within the bite event timeframe
        bite_preds = predictions[(predictions[:, -1] >= start_time) & (predictions[:, -1] <= end_time)]

        # Calculate duration and decide on padding or discarding
        duration = end_time - start_time
        if duration <= window_length:
            # Calculate the number of samples needed for a 9-second window
            num_samples = int(window_length / sample_rate)
            window = np.zeros((num_samples, predictions.shape[1]))  # Initialize window with zeros

            # Determine number of predictions to copy
            num_preds = min(len(bite_preds), num_samples)
            window[:num_preds, :] = bite_preds[:num_preds, :]

            windows.append(window)

    return windows


def extract_data_for_bite_window(start_time, end_time, predictions, window_length, step):
    # Finding indices for start and end times in the predictions array
    start_index = find_index_for_timestamp(start_time, predictions)
    end_index = find_index_for_timestamp(end_time, predictions)

    # Calculate the actual duration of the bite in seconds
    actual_duration = (end_index - start_index) * (step / 1000)

    # Check if the actual duration is within the acceptable range
    if actual_duration > window_length:
        return None  # Discard bites longer than 9 seconds

    # Calculate the total number of samples required for a 9-second window
    required_samples = int(window_length / (step / 1000))

    # Extracting the relevant slice from the predictions for the duration of the bite
    bite_data = predictions[start_index:end_index]

    # Padding with zeros if the bite duration is less than 9 seconds
    if bite_data.shape[0] < required_samples:
        padding_length = required_samples - bite_data.shape[0]
        padding = np.zeros((padding_length, bite_data.shape[1]))
        bite_data = np.vstack((bite_data, padding))

    return bite_data
