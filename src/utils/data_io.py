import os
import pickle

import numpy as np


def save_data(data, processed_data_directory, filename):
    """
    Save data to a pickle file.

    Args:
        data: Data to be saved, can be of any type.
        processed_data_directory (str): Directory to save the data.
        filename (str): Name of the file to save.
    """
    # Create the directory for the file if it doesn't exist
    file_dir = os.path.join(processed_data_directory)
    os.makedirs(file_dir, exist_ok=True)

    # Define the file path
    file_path = os.path.join(file_dir, f"{filename}.pkl")

    # Save the data in pickle format
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_bite_gt_data(bite_gt_path):
    """ Load the bite_gt data from the given paths. """
    bite_gt = np.load(bite_gt_path, allow_pickle=True)
    return bite_gt


def load_cnn_predictions(subject_to_indices, predictions_path):
    predictions = {}
    for _, sessions in subject_to_indices.items():
        for session in sessions:
            with open(f"{predictions_path}/prediction_{session}.pkl", "rb") as f:
                prediction = pickle.load(f)
            predictions[session] = prediction
    return predictions
