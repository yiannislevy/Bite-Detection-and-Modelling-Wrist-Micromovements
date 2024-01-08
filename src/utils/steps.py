import numpy as np
import json

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
    with open("../data/processed.nosync/mean_std_values.json", 'r') as f:
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


def one_hot_encode(label, num_classes=6):
    if 1 <= label <= num_classes:
        one_hot = np.zeros(num_classes)
        one_hot[label - 1] = 1
        return one_hot
    return np.zeros(num_classes)


def process_windows_and_assign_labels(session, mm_gt_session):
    labeled_windows = []
    for window in session:
        mid_timestamp = find_mid_timestamp_of_window(window)
        label = assign_label_to_window(mid_timestamp, mm_gt_session)
        if label != 6:  # Skip windows with label '6' aka 'Other'
            one_hot_label = one_hot_encode(label)
            labeled_windows.append((window, one_hot_label))
    return labeled_windows



def process_single_session(signal_session, label_session):
    common_data, common_labels = find_common_timeframe(signal_session, label_session)
    windowed_data = sliding_window(common_data)
    adjusted_labels = find_common_timeframe_after_windowing(windowed_data, common_labels)
    standardized_windows = standardize_windows(windowed_data)
    final_processed = process_windows_and_assign_labels(standardized_windows, adjusted_labels)

    final_processed_no_timestamp = [(window[:, 1:], label) for window, label in final_processed]

    return final_processed_no_timestamp


def process_all_sessions(signals, labels):
    all_processed_data = []
    for signal_session, label_session in zip(signals, labels):
        session_data = process_single_session(signal_session, label_session)
        all_processed_data.extend(session_data)
    return all_processed_data
