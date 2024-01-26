from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.optimizers.legacy import Adam as LegacyAdam
import numpy as np
import pickle as pkl
from src.utils.data_transform import *
from src.utils.data_io import save_data
from src.utils.utilities import append_timestamps_to_predictions
from src.utils.preprocessing import load_split_data
import pandas as pd
import os
import pickle
import json


def load_data(test_subject, subjects_sessions):
    training_data, testing_data, validation_data = [], [], []
    training_labels, testing_labels, validation_labels = [], [], []

    # Load training data (all subjects except the test and validation subjects)
    for subject in subjects_sessions.keys():
        if subject != test_subject:
            # and subject != validation_subject:
            subject_data, subject_labels = load_subject_data(f"../../data/ProcessedSubjects/MajorityLabel/subject_{subject}/data.pkl")
            training_data.append(subject_data)
            training_labels.append(subject_labels)

    # Load testing data (only the test subject)
    test_data, test_labels = load_subject_data(f"../../data/ProcessedSubjects/subject_{test_subject}/data.pkl")
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


def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=10, activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=10, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))  # Assuming 5 classes for the output layer
    optimizer = LegacyAdam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    return model


def train_model(subj_id, subject_to_indices):
    print(f"Training without {subj_id}")
    results = []
    accuracy = []
    loss = []
    model_x = build_model(input_shape=(20, 6))
    train_data, train_labels, test_data, test_labels = load_data(subj_id, subject_to_indices)
    history = model_x.fit(train_data, train_labels, epochs=32, batch_size=64)
    results.append(model_x.evaluate(test_data, test_labels))
    accuracy.append(history.history['accuracy'])
    loss.append(history.history['loss'])
    model_x.save(f"../models/full_loso/model_{subj_id}.keras")

    save_data(results, "../../models/full_loso/majority_label/training_info/", f"results_{subj_id}")
    save_data(accuracy, "../../models/full_loso/majority_label/training_info/", f"accuracy_{subj_id}")
    save_data(loss, "../../models/full_loso/majority_label/training_info/", f"loss_{subj_id}")

    return


def save_predictions(subj_id, subject_to_indices, start_time_json_path):
    start_time = load_start_time(start_time_json_path, subj_id)
    model = keras.models.load_model(f"../../models/full_loso/majority_label/model_{subj_id}.keras", compile=False)
    optimizer = LegacyAdam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    _, _, test_data, _ = load_split_data(subj_id, subject_to_indices)
    predictions = []
    for window in test_data:
        window_reshaped = window.reshape(1, 20, 6)
        prediction = model.predict(window_reshaped, verbose=False)
        predictions.append(prediction)
    predictions = np.array(predictions)
    predictions = predictions.squeeze(axis=1)
    predictions = np.vstack(predictions)
    predictions = append_timestamps_to_predictions(predictions, start_time)
    save_data(predictions, "../../data/cnn_predictions/timestamped", f"predictions_{subj_id}")
    return


def load_start_time(start_time_json_path, subject):
    """ Load the start time for the given subject from the JSON file. """

    with open(start_time_json_path, 'r') as file:
        start_times = json.load(file)
    return start_times[f"{subject}"]


def main():
    with open("../../data/dataset-info-json/subject_to_indices.json", "r") as f:
        subject_to_indices = json.load(f)

    subject_to_indices = {int(k): v for k, v in subject_to_indices.items()}

    start_time_json_path = '../../data/dataset-info-json/signal_start_times.json'

    for subject_id in subject_to_indices.keys():
        print(f"In {subject_id}")
        # train_model(subject_id, subject_to_indices)
        if subject_id <= 4:
            continue
        save_predictions(subject_id, subject_to_indices, start_time_json_path)
        print("here")
    return