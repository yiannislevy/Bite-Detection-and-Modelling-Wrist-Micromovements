from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle as pkl
from src.utils.data_transform import *
import pandas as pd
import os
import pickle


def load_data(path):
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

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load the data
file_path = '../../data/LOSO.nosync/'  # Replace with your file path
loso_results = []

# for subject in range(1, 13):
print(f"Training on all but subject {1}")

# Load training and testing data
train_data_path = os.path.join(file_path, f'all_but_{1}', 'train_data.pkl')
test_data_path = os.path.join(file_path, f'all_but_{1}', 'test_data.pkl')

X_train, y_train = load_data(train_data_path)
X_test, y_test = load_data(test_data_path)

model = build_model(input_shape=X_train.shape[1:])

model.fit(X_train, y_train, epochs=32, batch_size=32)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_classes, y_pred_classes)
loso_results.append(accuracy)
