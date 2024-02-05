import pandas as pd
import os
import pickle as pkl

with open ("../../data/original/FIC.pkl", "rb") as f:
    FIC = pkl.load(f)

# Open the processed dataset
with open ("../../data/old_cnn_processed.nosync/all_final/all_final.pkl", "rb") as f:
    dataset = pkl.load(f)

subject_ids = FIC["subject_id"]
subject_to_indices = {subject: [] for subject in set(subject_ids)}
for i, subject_id in enumerate(subject_ids):
    subject_to_indices[subject_id].append(i)

for subject in set(subject_ids):
    # Define directory name
    directory = f"../../data/LOSO/all_but_{subject}" # TODO: modify path
    os.makedirs(directory, exist_ok=True)

    # Get training and testing indices
    train_indices = [i for subj, indices in subject_to_indices.items() if subj != subject for i in indices]
    test_indices = subject_to_indices[subject]

    # Select data for training and testing
    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]

    # Save to .pkl files
    with open(f"{directory}/train_data.pkl", 'wb') as f:
        pkl.dump(train_data, f)

    with open(f"{directory}/test_data.pkl", 'wb') as f:
        pkl.dump(test_data, f)
