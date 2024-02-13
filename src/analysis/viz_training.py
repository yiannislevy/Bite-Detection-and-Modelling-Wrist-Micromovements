import matplotlib.pyplot as plt


def plot_training_results(accuracy, loss):
    flattened_accuracy = [item for item in accuracy]
    flattened_loss = [item for item in loss]

    epochs = range(1, len(flattened_accuracy) + 1)  # Assuming 1 value per epoch

    # Plotting both the training accuracy and loss
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, flattened_accuracy, label='Accuracy', color='blue')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, flattened_loss, label='Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
