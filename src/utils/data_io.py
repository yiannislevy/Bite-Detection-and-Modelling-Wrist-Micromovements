import os
import pickle

import numpy as np
import pandas as pd


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


def load_start_time(start_time_json_path, session):
    """ Load the start time for the given subject from the JSON file. """

    with open(start_time_json_path, "r") as file:
        start_times = json.load(file)
    return start_times[f"{session}"][0]


def load_data(test_subject, subject_to_indices, path_to_data):
    training_data, testing_data = [], []
    training_labels, testing_labels = [], []

    # Load data by sessions_old based on subject_to_indices mapping
    for subject, sessions in subject_to_indices.items():
        subject_data, subject_labels = [], []
        for session_id in sessions:
            session_data, session_labels = load_session_data(f"{path_to_data}/session_{session_id}.pkl")
            subject_data.append(session_data)
            subject_labels.append(session_labels)

        # Aggregate data for each subject
        subject_data = np.concatenate(subject_data, axis=0)
        subject_labels = np.concatenate(subject_labels, axis=0)

        # Distribute data into training or testing based on subject ID
        if str(subject) == str(test_subject):
            testing_data.append(subject_data)
            testing_labels.append(subject_labels)
        else:
            training_data.append(subject_data)
            training_labels.append(subject_labels)

    # Combine all training and testing data and labels
    training_data = np.concatenate(training_data, axis=0)
    training_labels = np.concatenate(training_labels, axis=0)
    testing_data = np.concatenate(testing_data, axis=0)
    testing_labels = np.concatenate(testing_labels, axis=0)

    return training_data, training_labels, testing_data, testing_labels


def load_session_data(path):
    data = pd.read_pickle(path)
    signal_data = np.array([item[0] for item in data])
    signal_data = signal_data[:, :, 1:]  # Exclude the timestamps column
    label_data = np.array([item[1] for item in data])
    return signal_data, label_data


def load_data_subjects(test_subject, subject_to_indices):
    training_data, testing_data, validation_data = [], [], []
    training_labels, testing_labels, validation_labels = [], [], []

    # Load training data (all subjects except the test and validation subjects)
    for subject in subject_to_indices.keys():
        if subject != test_subject :
            # and subject != validation_subject:
            subject_data, subject_labels = load_subject_data(f"../data/ProcessedSubjects/MajorityLabel/subjects/subject_{subject}/data.pkl")
            # subject_data, subject_labels = load_subject_data(f"../data/ProcessedSubjects/std_1/subject_{subject}/data.pkl") #TODO re-eval
            training_data.append(subject_data)
            training_labels.append(subject_labels)

    # Load testing data (only the test subject)
    test_data, test_labels = load_subject_data(f"../data/ProcessedSubjects/MajorityLabel/subjects/subject_{test_subject}/data.pkl")
    # test_data, test_labels = load_subject_data(f"../data/ProcessedSubjects/std_1/subject_{test_subject}/data.pkl")
    testing_data.append(test_data)
    testing_labels.append(test_labels)

    # # Load validation data (only the validation subject)
    # val_data, val_labels = load_subject_data(f"../data/ProcessedSubjects/subject_{validation_subject}/data.pkl")
    # validation_data.append(val_data)
    # validation_labels.append(val_labels)

    # Combine all training, testing, and validation data and labels
    training_data = np.concatenate(training_data, axis=0)
    training_labels = np.concatenate(training_labels, axis=0)
    testing_data = np.concatenate(testing_data, axis=0)
    testing_labels = np.concatenate(testing_labels, axis=0)
    # validation_data = np.concatenate(validation_data, axis=0)
    # validation_labels = np.concatenate(validation_labels, axis=0)

    return training_data, training_labels, testing_data, testing_labels


def load_subject_data(path):
    data = pd.read_pickle(path)
    signal_data = np.array([item[0] for item in data])
    label_data = np.array([item[1] for item in data])
    return signal_data, label_data
