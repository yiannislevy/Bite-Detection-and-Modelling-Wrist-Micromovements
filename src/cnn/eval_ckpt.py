import tensorflow as tf
import os
import glob


def evaluate_checkpoints(checkpoint_dir, test_data, test_labels, model):
    """
    Evaluates a Keras model using weights from checkpoints stored in a directory.

    Args:
    - checkpoint_dir: The directory where checkpoint files are stored.
    - test_data: Numpy array or a tf.data.Dataset containing test features.
    - test_labels: Numpy array or a tf.data.Dataset containing test labels.
    - model: A tf.keras Model instance.

    Returns:
    - A dictionary containing the best checkpoint path, the best accuracy obtained,
      and a list of results for all evaluated checkpoints.
    """

    # Initialize variables to store evaluation results and track the best checkpoint.
    all_results = []
    best_ckpt = None
    best_accuracy = -float('inf')

    # Iterate over checkpoint files in the directory, only considering '.index' files
    # because each checkpoint consists of multiple files, but '.index' uniquely identifies them.
    for index_file in sorted(glob.glob(os.path.join(checkpoint_dir, "cp-*.ckpt.index"))):
        # Remove the '.index' part to get the prefix TensorFlow needs to load the checkpoint.
        ckpt_prefix = index_file[:-6]  # Removes the last 6 characters, which are '.index'

        try:
            # Load the weights into the model from the checkpoint prefix.
            model.load_weights(ckpt_prefix)

            # Evaluate the model on the test data and labels.
            loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
            print(f"Checkpoint: {ckpt_prefix}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            # If the current checkpoint's accuracy is better than the best one so far, update it.
            if accuracy > best_accuracy:
                best_ckpt = ckpt_prefix
                best_accuracy = accuracy

            # Save the checkpoint path and its evaluation metrics.
            all_results.append({'ckpt_prefix': ckpt_prefix, 'accuracy': accuracy, 'loss': loss})

        except Exception as e:
            # If an error occurs while loading the checkpoint or evaluating, print the error message.
            print(f"Error loading checkpoint {ckpt_prefix}: {e}")

    # Return a dictionary with the best checkpoint, its accuracy, and all evaluation results.
    return {
        'best_ckpt': best_ckpt,
        'best_accuracy': best_accuracy,
        'all_results': all_results
    }


# Example usage:
# checkpoint_dir = os.path.abspath("/path/to/your/checkpoints/folder")
# model = build_model((20, 6))
# results = evaluate_checkpoints(checkpoint_dir, test_data, test_labels, model)
# print(f"Best checkpoint: {results['best_ckpt']}")
# print(f"Best accuracy: {results['best_accuracy']:.4f}")
