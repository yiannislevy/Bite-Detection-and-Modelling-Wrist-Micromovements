import numpy as np
import json
import pandas as pd


def calculate_mean_std(data):
    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)

    mean_std = {'means': means.tolist(), 'std_devs': std_devs.tolist()}
    with open('../data/processed/mean_std_values.json', 'w') as f:
        json.dump(mean_std, f)

    return mean_std


def standardize_data(data, json_file):
    with open(json_file, 'r') as f:
        mean_std = json.load(f)

    means = np.array(mean_std['means'])
    std_devs = np.array(mean_std['std_devs'])

    standardized_data = (data - means) / std_devs
    return standardized_data


# Example usage:
# standardized_data = standardize_data(your_data_structure, 'mean_std_values.json')

def load_split_data(test_subject, subject_to_indices):
    training_data, testing_data, validation_data = [], [], []
    training_labels, testing_labels, validation_labels = [], [], []

    # Load training data (all subjects except the test and validation subjects)
    for subject in subject_to_indices.keys():
        if subject != test_subject :
            # and subject != validation_subject:
            subject_data, subject_labels = load_subject_data(f"../data/ProcessedSubjects/MajorityLabel/subjects/subject_{subject}/data.pkl")
            # subject_data, subject_labels = load_subject_data(f"../data/ProcessedSubjects/Standardized/subject_{subject}/data.pkl") #TODO re-eval
            training_data.append(subject_data)
            training_labels.append(subject_labels)

    # Load testing data (only the test subject)
    test_data, test_labels = load_subject_data(f"../data/ProcessedSubjects/MajorityLabel/subjects/subject_{test_subject}/data.pkl")
    # test_data, test_labels = load_subject_data(f"../data/ProcessedSubjects/Standardized/subject_{test_subject}/data.pkl")
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
