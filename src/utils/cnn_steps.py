import numpy as np
import json
from collections import Counter
from scipy.signal import firwin, lfilter, medfilt


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
        data[:, i] = lfilter(hp_filter, 1.0, data[:, i])

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
    start_time = max(signals_session[0, 0], mm_gt_session[0, 0])
    end_time = min(signals_session[-1, 0], mm_gt_session[-1, 1])

    signals_session_common = signals_session[(signals_session[:, 0] >= start_time) & (signals_session[:, 0] <= end_time)]
    mm_gt_session_common = mm_gt_session[(mm_gt_session[:, 0] >= start_time) & (mm_gt_session[:, 1] <= end_time)]

    return signals_session_common, mm_gt_session_common


def sliding_window(session, window_length=20, step_size=10):
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
    if len(windows) == 0:
        return np.empty((0, mm_gt_session.shape[1]))

    first_window_start_timestamp = windows[0][0, 0]
    last_window_end_timestamp = windows[-1][-1, 0]

    adjusted_mm_gt_session = mm_gt_session[(mm_gt_session[:, 1] >= first_window_start_timestamp) & (mm_gt_session[:, 0] <= last_window_end_timestamp)]

    return adjusted_mm_gt_session


def standardize_windows(data):
    with open("../../data/dataset-info-json/mean_std_values.json", 'r') as f:
        mean_std = json.load(f)

    means = np.array(mean_std['means'])
    std_devs = np.array(mean_std['std_devs'])

    # Iterate through each window in the data
    for window_index in range(len(data)):
        # Exclude the first feature (timestamps) and standardize the remaining features
        for feature_index in range(1, 7):  # Assuming features 1 to 6 are to be standardized
            data[window_index, :, feature_index] = (data[window_index, :, feature_index] - means[feature_index - 1]) / std_devs[feature_index - 1]

    return data


def find_mid_timestamp_of_window(window):
    return window[9][0]


def assign_label_to_window(mid_timestamp, labels):
    for start, end, label in labels:
        if start <= mid_timestamp <= end:
            return int(label)
    return -1


def one_hot_encode(label, num_classes=5):
    if 1 <= label <= num_classes:
        one_hot = np.zeros(num_classes)
        one_hot[label - 1] = 1
        return one_hot
    return np.zeros(num_classes)


# Option 1 to get the middle timestamp label
def process_windows_and_assign_labels(session, mm_gt_session):
    labeled_windows = []
    for window in session:
        mid_timestamp = find_mid_timestamp_of_window(window)
        label = assign_label_to_window(mid_timestamp, mm_gt_session)
        if label != 6:  # Skip windows with label '6' aka 'Other'
            one_hot_label = one_hot_encode(label)
            labeled_windows.append((window, one_hot_label))
    return labeled_windows


# ----------------------------------------------------------------
# Option 2 to get majority label
def process_windows_assign_majority_label(session, mm_gt_session):
    labeled_windows = []
    for window in session:
        window_labels = get_labels_for_window(window, mm_gt_session)
        label = assign_label_to_window_majority(window_labels)
        if label != -1 and label != 6:  # Discard windows with no dominant label or 'Other'
            one_hot_label = one_hot_encode(label)
            labeled_windows.append((window, one_hot_label))
    return labeled_windows


def get_labels_for_window(window, labels):
    window_labels = []
    for timestamp in window:
        for start, end, label in labels:
            if start <= timestamp[0] <= end:
                window_labels.append(label)
                break
    return window_labels


def assign_label_to_window_majority(window_labels, threshold=0.75):
    if not window_labels:
        return -1
    label_count = Counter(window_labels)
    most_common_label, count = label_count.most_common(1)[0]
    if count / len(window_labels) >= threshold:
        return int(most_common_label)
    return -1
# ----------------------------------------------------------------


def process_single_session(signal_session, label_session):
    removed_gravity_data = remove_gravity(signal_session)
    filtered_data = median_filter(removed_gravity_data)
    common_data, common_labels = find_common_timeframe(filtered_data, label_session)
    windowed_data = sliding_window(common_data)
    adjusted_labels = find_common_timeframe_after_windowing(windowed_data, common_labels)
    standardized_windows = standardize_windows(windowed_data)
    # option 1:
    # final_processed = process_windows_and_assign_labels(windowed_data, adjusted_labels)
    # option 2:
    final_processed = process_windows_assign_majority_label(standardized_windows, adjusted_labels)

    # final_processed_no_timestamp = [(window[:, 1:], label) for window, label in final_processed]

    return final_processed


def process_all_sessions(signals, labels):
    all_processed_data = []
    i = 1
    for signal_session, label_session in zip(signals, labels):
        print(f"Session: {i}")
        i += 1
        session_data = process_single_session(signal_session, label_session)
        print(len(session_data))
        all_processed_data.extend(session_data)
    return all_processed_data
