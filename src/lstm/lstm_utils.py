from random import shuffle
import numpy as np


def combine_examples(examples, label, combined_sessions):
    """
        Adds examples to a dictionary of combined sessions, labeling them according to the specified label.

        Parameters:
        - examples: dict, with session IDs as keys and lists of tuples (numpy array, original label) as values.
        - label: int, the label to assign to all examples being added (1 for positive, 0 for negative).
        - combined_sessions: dict, a dictionary where each key is a session ID and each value is a list of tuples (numpy array, label).

        Returns:
        - combined_sessions: dict, the updated dictionary with the new examples added and labeled accordingly.
        """
    for session_id, items in examples.items():
        if session_id not in combined_sessions:
            combined_sessions[session_id] = []
        combined_sessions[session_id].extend([(item[0], label) for item in items])
    return combined_sessions


def combine_and_balance_examples(positive_examples, negative_examples, combined_sessions):
    """
        Combines and balances the positive and negative examples in each session to ensure an equal number of each.

        Parameters:
        - positive_examples: dict, with session IDs as keys and lists of tuples (numpy array, label=1) as values.
        - negative_examples: dict, with session IDs as keys and lists of tuples (numpy array, label=0) as values.
        - combined_sessions: dict, an initially empty dictionary that will be filled with balanced examples from both positive and negative examples.

        For each session, the function first adds all positive examples. Then, it adds an equal number of negative examples,
        randomly selected if there are more negative than positive examples in that session.

        Returns:
        - combined_sessions: dict, the updated dictionary with balanced positive and negative examples per session.
        """
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
    """
        Shuffles the examples within each session in-place.

        Parameters:
        - combined_sessions: dict, a dictionary where each key is a session ID and each value is a list of tuples (numpy array, label). The examples within each session list are shuffled to randomize their order.

        Returns:
        - combined_sessions: dict, the same dictionary passed as input, but with the examples in each session shuffled.
        """
    for session_id in combined_sessions.keys():
        shuffle(combined_sessions[session_id])
    return combined_sessions
