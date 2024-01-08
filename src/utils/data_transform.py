import numpy as np


def sliding_window(signal, window_length = 20, step_size = 10):
    windowed_data = []
    for session in signal:
        num_windows = (session.shape[0] - window_length) // step_size + 1
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_length
            windowed_data.append(session[start_idx:end_idx])
    return np.array(windowed_data)
