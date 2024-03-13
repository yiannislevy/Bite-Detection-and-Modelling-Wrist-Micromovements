import numpy as np
import json
from collections import Counter
from scipy.signal import firwin, filtfilt, medfilt


def remove_gravity(data, sample_rate=100, cutoff_hz=1):
    """
    Apply a custom high-pass filter to the accelerometer data in a NumPy array.

    Args:
        data (numpy.ndarray): Array containing sensor data with shape (N, 7),
                              where columns are [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].
        sample_rate (int): Sample rate of the sensor data.
        cutoff_hz (int): Cutoff frequency for the high-pass filter.

    Returns:
        numpy.ndarray: Array with the filtered accelerometer data, preserving the original format.
    """
    num_taps = sample_rate * 5 + 1  # Number of filter taps
    hp_filter = firwin(num_taps, cutoff_hz / (sample_rate / 2), pass_zero=False)

    # Apply the filter to accelerometer data (columns 1 to 3)
    for i in range(1, 4):
        data[:, i] = filtfilt(hp_filter, 1.0, data[:, i])

    return data


def median_filter(data, filter_order=5):
    """
    Apply a median filter to the accelerometer and gyroscope sensor data in a NumPy array.

    Args:
        data (numpy.ndarray): Array containing sensor data with shape (N, 7),
                              where columns are [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].
        filter_order (int): The order of the median filter (kernel size).

    Returns:
        numpy.ndarray: Array with the filtered sensor data, preserving the original format.
    """
    # Apply median filter to accelerometer data (columns 1 to 3) and gyroscope data (columns 4 to 6)
    for i in range(1, 7):  # Skip the timestamp column
        data[:, i] = medfilt(data[:, i], kernel_size=filter_order)

    return data


def find_common_timeframe(signals_session, mm_gt_session):
    """
    Finds the common timeframe between sensor signals and micromovement ground truth (mm_gt) sessions. It aligns both
    datasets by their timestamps to ensure that only the overlapping time period is considered for analysis.

    Args:
        signals_session (numpy.ndarray): A 2D array where each row represents a sensor signal with the first column as timestamps.
        mm_gt_session (numpy.ndarray): A 2D array where each row represents a ground truth event with start and end timestamps.
    Returns:
        tuple: A tuple containing two numpy.ndarrays: the filtered signals_session and mm_gt_session
    that fall within the common timeframe.
    """
    start_time = max(signals_session[0, 0], mm_gt_session[0, 0])
    end_time = min(signals_session[-1, 0], mm_gt_session[-1, 1])

    signals_session_common = signals_session[(signals_session[:, 0] >= start_time) & (signals_session[:, 0] <= end_time)]
    mm_gt_session_common = mm_gt_session[(mm_gt_session[:, 0] >= start_time) & (mm_gt_session[:, 1] <= end_time)]

    return signals_session_common, mm_gt_session_common


def sliding_window(session, window_length=20, step_size=10):
    """
    Segments the session data into overlapping windows using a sliding window approach. Each window captures a portion
    of the session to facilitate the extraction of temporal features for CNN processing.

    Args:
        session (numpy.ndarray): Array containing session data to be windowed.
        window_length (int): The number of data points each window should contain.
        step_size (int): The number of data points to slide between windows.
    Returns:
        numpy.ndarray: An array of windowed data, where each window is a subset of the session data.
    """
    windowed_data = []
    num_windows = (len(session) - window_length) // step_size + 1

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        window = session[start_idx:end_idx]

        if len(window) == window_length:
            windowed_data.append(window)

    return np.array(windowed_data)


def find_common_timeframe_after_windowing(windows, mm_gt_session):
    """
    Adjusts micromovement ground truth (mm_gt) session timestamps based on the timeframe of the windowed data. Ensures
    that labels correspond to the time period after data has been segmented into windows.

    Args:
        windows (numpy.ndarray): Array of windowed session data.
        mm_gt_session (numpy.ndarray): Ground truth data for micromovements.
    Returns:
        numpy.ndarray: Adjusted mm_gt_session to match the timeframe of the windowed data.
    """
    if len(windows) == 0:
        return np.empty((0, mm_gt_session.shape[1]))

    first_window_start_timestamp = windows[0][0, 0]
    last_window_end_timestamp = windows[-1][-1, 0]

    adjusted_mm_gt_session = mm_gt_session[(mm_gt_session[:, 1] >= first_window_start_timestamp) & (mm_gt_session[:, 0] <= last_window_end_timestamp)]

    return adjusted_mm_gt_session


def standardize_windows(data):
    """
    Standardizes the sensor data within each window based on pre-calculated mean and standard deviation values.
    This normalization is crucial for effective learning in the CNN model.

    Args:
        data (numpy.ndarray): Array of windowed session data to be standardized.
    Returns:
        numpy.ndarray: The standardized windowed data.
    """
    with open("../../data/dataset-info-json/mean_std_values_3.json", 'r') as f:
        mean_std = json.load(f)

    means = np.array(mean_std['means'])
    std_devs = np.array(mean_std['std_devs'])

    # Iterate through each window in the data
    for window_index in range(len(data)):
        # Exclude the first feature (timestamps) and standardize the remaining features
        for feature_index in range(1, 7):  # Assuming features 1 to 6 are to be standardized
            data[window_index, :, feature_index] = (data[window_index, :, feature_index] - means[feature_index - 1]) / std_devs[feature_index - 1]

    return data


def one_hot_encode(label, num_classes=5):
    """
    documentation:
    Performs one hot encoding of the labels provided.
    Args:
        label (int): Ground truth label on which class the window belongs to.
        num_classes (int): The number of output classes desired.
    Returns:
        numpy.ndarray: Array of the one hot encoded label
    """
    if 1 <= label <= num_classes:
        one_hot = np.zeros(num_classes)
        one_hot[label - 1] = 1
        return one_hot
    return np.zeros(num_classes)


# Option 1 to get the middle timestamp label
def process_windows_and_assign_middle_sample_label(session, mm_gt_session):
    """
    Processes a list of data windows from a session, assigns a label to each window based on its midpoint timestamp,
    and encodes the label into a one-hot format. The assignment of labels is determined by comparing the midpoint
    timestamp of each window against a set of known time intervals, each associated with a specific label. This method
    allows for the contextual labeling of data based on temporal alignment with ground truth intervals.

    This function encapsulates the entire process of identifying the relevant label for each window in a session:
    1. Finding the midpoint timestamp of each window.
    2. Assigning a label based on this midpoint timestamp by checking against predefined intervals with labels.
    3. Encoding the assigned label into a one-hot format suitable for machine learning models.

    The function ensures that each window is labeled with the most contextually relevant information based on its
    position within the session's timeline, disregarding windows that do not align with any predefined label intervals.

    Args:
        session (list of numpy.ndarray): A list containing arrays of data points, each array representing a window in the session.
        mm_gt_session (list of tuples): Ground truth data, where each tuple contains a start time, end time, and the label for that interval.

    Returns:
        list of tuples: A list where each element is a tuple containing the original data window and its corresponding one-hot encoded label.
    """

    # Internal function to find the midpoint timestamp of a window
    def find_mid_timestamp_of_window(window):
        return window[9][0]  # Assumes the midpoint is at the 10th position

    # Internal function to assign a label to a window based on the midpoint timestamp
    def assign_label_to_window(mid_timestamp, labels):
        for start, end, label in labels:
            if start <= mid_timestamp <= end:
                return int(label)
        return -1

    labeled_windows = []
    for window in session:
        mid_timestamp = find_mid_timestamp_of_window(window)
        label = assign_label_to_window(mid_timestamp, mm_gt_session)
        one_hot_label = one_hot_encode(label, 6)  # Assumes one_hot_encode function is defined elsewhere
        labeled_windows.append((window, one_hot_label))
    return labeled_windows


# ----------------------------------------------------------------
# Option 2 to get majority label

def process_windows_assign_majority_label(session, mm_gt_session):
    """
    Processes a list of data windows from a session by assigning a label based on the majority rule within each window.
    Each window is evaluated to determine the most frequent label within its timeframe, based on a comparison with
    ground truth intervals provided in `mm_gt_session`. A label is assigned to the window if a single label is dominant
    according to a specified threshold. The function then encodes this label into a one-hot format, filtering out
    windows without a clear majority label or labeled as 'Other'.

    This method is particularly useful for sessions where multiple labels might be applicable to a single window but
    where only the most frequent (dominant) label is of interest, allowing for a simplified representation suitable for
    machine learning applications.

    Args:
        session (list of numpy.ndarray): The session data, a list of windows where each window is an array of data points.
        mm_gt_session (list of tuples): Ground truth data for the session, where each tuple contains (start, end, label) for intervals.

    Returns:
        list of tuples: A list where each tuple contains a window from the session and its corresponding one-hot encoded label,
                        excluding windows without a dominant label or labeled as 'Other'.
    """

    def get_labels_for_window(window, labels):
        """
        Internal function to collect labels applicable to each timestamp within a window.
        """
        window_labels = []
        for timestamp in window:
            for start, end, label in labels:
                if start <= timestamp[0] <= end:
                    window_labels.append(label)
                    break
        return window_labels

    def assign_label_to_window_majority(window_labels, threshold=0.75):
        """
        Internal function to determine the majority label within a window, based on a specified threshold.
        """
        if not window_labels:
            return -1
        label_count = Counter(window_labels)
        most_common_label, count = label_count.most_common(1)[0]
        if count / len(window_labels) >= threshold:
            return int(most_common_label)
        return -1

    labeled_windows = []
    for window in session:
        window_labels = get_labels_for_window(window, mm_gt_session)
        label = assign_label_to_window_majority(window_labels)
        if label != -1 and label != 6:  # Discard windows with no dominant label or 'Other'
            one_hot_label = one_hot_encode(label)  # Assumes one_hot_encode function is defined elsewhere
            labeled_windows.append((window, one_hot_label))
    return labeled_windows

# ----------------------------------------------------------------


def process_single_session(signal_session, label_session, for_training=False, label='majority'):
    """
        Processes a single session of signal data through a comprehensive preprocessing pipeline. The pipeline
        includes gravity removal, noise filtering, windowing, and standardization of the signal data. For training
        data, it further includes alignment of signal and label data to a common timeframe, label adjustment after
        windowing, and a choice between two labeling strategies. The final step removes timestamps for analysis or
        modeling purposes.

        Steps:
        1. Remove gravity effects from the signal data to focus on the dynamic movements.
        2. Apply a median filter to smooth the data and reduce noise.
        3. (For training) Find the common timeframe between filtered signal data and labels to synchronize data points with their corresponding labels.
        4. Segment the data into smaller, overlapping windows using a sliding window technique. This facilitates the analysis of time-series data in manageable chunks.
        5. (For training) Adjust labels to ensure they correspond accurately to the newly created signal windows.
        6. Standardize the signal windows to have a mean of zero and a standard deviation of one, normalizing the data across all sessions.
        7. (For training) Select a labeling strategy for the windowed data:
           a. Assign labels based on the midpoint timestamp of each window.
           b. Assign labels based on the majority label present within each window.
        8. Remove timestamps from the final processed data, keeping only signal values and their corresponding labels for analysis or modeling.

        Args:
            signal_session (np.ndarray): Array containing the signal data for a single session.
            label_session (np.ndarray): Array containing the labels corresponding to the signal data.
            for_training (bool): Flag indicating whether the data is being preprocessed for training (True) or for prediction (False).
            label (str): Flag indicating which labelling technique to follow between majority/middle
        Returns:
            np.ndarray: Array of processed data, formatted according to the specified purpose (training or prediction).
    """

    removed_gravity_data = remove_gravity(signal_session)
    filtered_data = median_filter(removed_gravity_data)

    if for_training:
        common_data, common_labels = find_common_timeframe(filtered_data, label_session)
    else:
        common_data = filtered_data  # Placeholder, assuming find_common_timeframe modifies data for training

    windowed_data = sliding_window(common_data)

    if for_training:
        adjusted_labels = find_common_timeframe_after_windowing(windowed_data, common_labels)
    else:
        adjusted_labels = None  # Placeholder for non-training mode

    standardized_windows = standardize_windows(windowed_data)

    if for_training:
        if label == 'middle':
            # Option 1: Assign labels based on the midpoint timestamp of each window.
            final_processed = process_windows_and_assign_middle_sample_label(standardized_windows, adjusted_labels)
        elif label == 'majority':
            # Option 2: Assign labels based on the majority label present within each window.
            final_processed = process_windows_assign_majority_label(standardized_windows, adjusted_labels)
        else:
            final_processed = None

        return final_processed
    else:
        # For prediction, return standardized windows without further processing.
        return standardized_windows


def process_all_sessions(signals, labels):
    """
    Processes multiple sessions of signal data, applying the preprocessing steps defined in process_single_session to each. This function is designed to handle data from multiple subjects, where each subject may have multiple sessions. It sequentially processes each session, printing progress along the way, and aggregates the processed data from all sessions.

    The primary goal is to bundle preprocessed data per subject, allowing for analysis or model training on a per-subject basis. This can accommodate scenarios where subjects undergo multiple sessions, ensuring that data from the same subject is processed in a consistent manner.

    Args:
        signals (list of np.ndarray): A list containing the signal data for each session.
        labels (list of np.ndarray): A list containing the labels for each session, aligned with the signal data.

    Returns:
        list of np.ndarray: A list of processed data arrays from all sessions, bundled per subject.
    """
    all_processed_data = []
    i = 1
    for signal_session, label_session in zip(signals, labels):
        print(f"Session: {i}")
        session_data = process_single_session(signal_session, label_session)
        print(len(session_data))
        all_processed_data.extend(session_data)
        i += 1
    return all_processed_data
