"""
Prediction Generation and Session-Based Saving Script

This script performs the following key tasks:
1. Initializes paths for model and output directories.
2. Loads 'subject_to_indices' and 'session_start_time_and_length' data.
3. Defines the function 'save_predictions' to process data for each subject:
    a. Loads and compiles a Keras model for each subject.
    b. Loads test data for the subject.
    c. Generates predictions for each data window.
    d. Splits predictions into sessions_old based on subject indices and session times.
    e. Appends timestamps to each session-based prediction.
    f. Saves the timestamped, session-based predictions to the specified path.

Custom utility functions used:
- 'load_split_data': For loading and splitting the dataset.
- 'split_predictions_to_sessions': For dividing predictions into sessions_old.
- 'append_timestamps_to_predictions': For adding timestamps to predictions.
- 'save_data': For saving processed data.

Note: Ensure correct setup of paths and availability of utility functions.
"""


import json
import numpy as np
import pandas as pd
import keras
from keras.optimizers.legacy import Adam as LegacyAdam
from src.utils.data_io import save_data, load_data
from src.utils.prediction_utilities import append_timestamps_to_predictions, split_predictions_to_sessions
# Paths
path_to_models = "../models/full_loso/majority_label/processed/std_3/"
path_to_save = "../data/cnn_predictions/complete/timestamped"
path_to_data = "../data/ProcessedSubjects/for_predictions/full_imu/"

# Load subject to indices and session start times
with open("../../data/dataset-info-json/subject_to_indices.json", "r") as f:
    subject_to_indices = json.load(f)
subject_to_indices = {int(k): v for k, v in subject_to_indices.items()}

with open("../../data/dataset-info-json/signal_start_times-MAJORITY-LABELS.json", "r") as f:
    session_start_time_and_length = json.load(f)


# Function to save predictions, timestamped and in sessions
def save_predictions_subjected():
    """
        Processes and saves predictions for each test subject.
        Loads the subject-specific model, generates predictions from test data,
        organizes these predictions by session with timestamps and saves them.
    """
    for test_subject_id in subject_to_indices.keys():
        # Load model for each subject
        model = keras.models.load_model(f"{path_to_models}model_{test_subject_id}.keras", compile=False)
        model.compile(optimizer=LegacyAdam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=["accuracy"])

        # Load test data
        path_to_sessions = f"{path_to_data}/sessions"
        _, _, test_data, _ = load_data(test_subject_id, subject_to_indices, path_to_sessions)

        # Generate predictions
        predictions = []
        for window in test_data:
            window_reshaped = window.reshape(1, 20, 6)
            prediction = model.predict(window_reshaped)
            predictions.append(prediction)
        predictions = np.array(predictions)
        predictions = predictions.squeeze(axis=1)

        # Split predictions into sessions and append timestamps
        sessioned_predictions = split_predictions_to_sessions(predictions, subject_to_indices[test_subject_id], session_start_time_and_length)

        # Save predictions
        for session_id, prediction in sessioned_predictions.items():
            timestamped_predictions = append_timestamps_to_predictions(prediction, session_id, f"{path_to_data}/timestamps")
            save_data(timestamped_predictions, path_to_save, f"prediction_{session_id}")


def save_predictions_sessioned():
    """
        Processes and saves predictions for each session.
        Loads the subject-specific model, generates predictions from test data,
        organizes these predictions by session with timestamps and saves them.
    """
    for test_subject_id in subject_to_indices.keys():
        # Load model for each subject
        model = keras.models.load_model(f"{path_to_models}model_{test_subject_id}.keras", compile=False)
        model.compile(optimizer=LegacyAdam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=["accuracy"])

        for session_id in subject_to_indices[test_subject_id]:
            session_path = f"{path_to_data}/sessions/session_{session_id}.pkl"
            test_data = pd.read_pickle(session_path)
            test_data = test_data[:, :, 1:]

            # Generate predictions
            predictions = []
            for window in test_data:
                window_reshaped = window.reshape(1, 20, 6)
                prediction = model.predict(window_reshaped)
                predictions.append(prediction)
            predictions = np.array(predictions)
            predictions = predictions.squeeze(axis=1)

            timestamped_predictions = append_timestamps_to_predictions(predictions, session_id,
                                                                       f"{path_to_data}/timestamps")
            save_data(timestamped_predictions, path_to_save, f"prediction_{session_id}")


def save_predictions_sessioned_single_model(path_to_model, path_to_data, path_to_save):
    """
    Processes and saves predictions for each session using a single model.
    Loads a single model, generates predictions from test data,
    organizes these predictions by session with timestamps, and saves them.
    """
    # Load a single model here. Adjust the model loading logic as needed.
    model = keras.models.load_model(path_to_model, compile=False)
    model.compile(optimizer=LegacyAdam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=["accuracy"])

    for session_id in range(1, 22):
        session_path = f"{path_to_data}/session_{session_id}.pkl"
        test_data = pd.read_pickle(session_path)
        test_data = test_data[:, :, 1:]

        # Generate predictions
        predictions = []
        for window in test_data:
            window_reshaped = window.reshape(1, 20, 6)
            prediction = model.predict(window_reshaped)
            predictions.append(prediction)
        predictions = np.array(predictions)
        predictions = predictions.squeeze(axis=1)

        timestamped_predictions = append_timestamps_to_predictions(predictions, session_id,
                                                                   f"{path_to_data}/timestamps")
        save_data(timestamped_predictions, path_to_save, f"prediction_{session_id}")
