from random import shuffle
import numpy as np


def combine_examples(examples, label, combined_sessions):
    for session_id, items in examples.items():
        if session_id not in combined_sessions:
            combined_sessions[session_id] = []
        combined_sessions[session_id].extend([(item[0], label) for item in items])
    return combined_sessions


def combine_and_balance_examples(positive_examples, negative_examples, combined_sessions):
    for session_id, items in positive_examples.items():
        if session_id not in combined_sessions:
            combined_sessions[session_id] = []
        combined_sessions[session_id].extend([(item[0], 1) for item in items])

    for session_id, items in negative_examples.items():
        if session_id in combined_sessions:
            positive_count = sum(1 for item in combined_sessions[session_id] if item[1] == 1)
            if len(items) > positive_count:
                # Generate indices to select negative examples
                indices = np.random.choice(len(items), positive_count, replace=False)
                selected_negatives = [items[i] for i in indices]
            else:
                selected_negatives = items
            combined_sessions[session_id].extend([(item[0], 0) for item in selected_negatives])

    return combined_sessions


def shuffle_examples(combined_sessions):
    for session_id in combined_sessions.keys():
        shuffle(combined_sessions[session_id])
    return combined_sessions
