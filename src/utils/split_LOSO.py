import pandas as pd
import os
import pickle

with open ("../../data/FIC.pkl", "rb") as f:
    dataset = pkl.load(f)

df = pd.DataFrame(dataset)

# Extract unique subjects
unique_subjects = df['subject_id'].unique()

for subject in unique_subjects:
    directory = f"../../data/LOSO/all_but_{subject}"
    os.makedirs(directory, exist_ok=True)

    train_indices = df[df['subject_id'] != subject].index
    test_indices = df[df['subject_id'] == subject].index

    train_signals = [dataset['signals_proc'][i] for i in train_indices]
    test_signals = [dataset['signals_proc'][i] for i in test_indices]

    with open(f"{directory}/train_data.pkl", 'wb') as f:
        pickle.dump(train_signals, f)

    with open(f"{directory}/test_data.pkl", 'wb') as f:
        pickle.dump(test_signals, f)
