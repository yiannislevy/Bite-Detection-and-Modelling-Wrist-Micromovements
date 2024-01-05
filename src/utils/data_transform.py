def segment_data_with_sliding_window(signals, window_length=20, step_size=10):
    """
    Segments the signals data using a sliding window approach.
    Assumes that the timestamp column has already been removed.
    """
    segmented = []
    for signal in signals:
        # Drop the timestamp column (first column)
        signal_no_timestamp = signal[:, 1:]

        # Segment the signal using a sliding window
        for start in range(0, len(signal_no_timestamp) - window_length + 1, step_size):
            segmented.append(signal_no_timestamp[start:start + window_length])

    return np.array(segmented)

def label_segments_with_gt(segments, ground_truth, window_length=20, step_size=10):
    """
    Labels the segments with the ground truth values.
    """
    labels = []
    for gt in ground_truth:
        # Segment the ground truth using a sliding window
        for start in range(0, len(gt) - window_length + 1, step_size):
            labels.append(gt[start:start + window_length])

    return np.array(labels)