import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_histograms(data):
    plt.figure(figsize=(12, 8))
    for i in range(data.shape[1]):
        plt.subplot(3, 3, i + 1)
        plt.hist(data[:, i], bins=50, alpha=0.7)
        plt.title(f'Feature {i+1}')

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_histograms(standardized_data)


def plot_probability_distributions(predictions, start_index, num_windows):
    # Create a 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set the number of categories
    number_of_categories = predictions.shape[1]

    # Define colors for each category
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Initialize arrays for the positions and sizes of the bars
    x_pos = []
    y_pos = []
    z_pos = []
    dz = []
    color = []

    # Set the size of the bars
    dx = 0.8
    dy = 0.4

    # Loop through each window and category to collect bar data
    for window in range(start_index, start_index + num_windows):
        for category in range(number_of_categories):
            x_pos.append(window)
            y_pos.append(category)
            z_pos.append(0)
            dz.append(predictions[window, category])
            color.append(colors[category])

    # Plot the bars
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=color, zsort='max')

    # Set labels, titles, and category names
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Category')
    ax.set_zlabel('Probability')
    ax.set_title('Temporal Evolution of Micromovements Distribution')
    ax.set_yticks(range(number_of_categories))
    ax.set_yticklabels(['No Movement', 'Upwards', 'Downwards', 'Pick Food', 'Mouth'])

    # Show the plot
    plt.show()


def plot_mando_and_lstm_preds(times, cumulative_weight, predictions):
    """
    Plot cumulative weight and prediction probabilities over time.

    Parameters:
    - times: numpy.ndarray, array of time points.
    - cumulative_weight: numpy.ndarray, cumulative weight over time.
    - predictions: numpy.ndarray, prediction data.
    """
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cumulative Weight (grams)', color=color)
    ax1.plot(times, cumulative_weight, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Prediction Probability', color=color)
    ax2.plot(np.arange(0, len(predictions) * 0.1, 0.1), predictions, color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Cumulative Weight vs. Prediction Probability Over Time')
    plt.show()


def plot_data_with_ground_truth_events(times, cumulative_weight, predictions, ground_truth):
    _, bite_events = count_bites(predictions)

    fig, ax1 = plt.subplots(figsize=(20, 10))

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cumulative Weight (grams)', color=color)
    ax1.plot(times, cumulative_weight, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Prediction Probability', color=color)
    ax2.plot(np.arange(0, len(predictions) * 0.1, 0.1), predictions, color=color, alpha=0.5, linestyle='dotted',
             linewidth=0.8)
    ax2.scatter(bite_events * 0.1, predictions[bite_events], color='green', label='Predicted Bite Events\n(LSTM)',
                marker='*')
    ax2.tick_params(axis='y', labelcolor=color)

    # Plot ground truth bite windows
    for bite in ground_truth:
        start_time, end_time = bite[1], bite[2]
        ax1.axvspan(start_time, end_time, color='gray', alpha=0.2)

    fig.tight_layout(pad=4.0)  # Adjust padding as needed
    plt.subplots_adjust(top=0.88)  # Adjust the top padding to give more space for the title
    plt.title('Cumulative Weight vs. Prediction Probability Over Time with Bite Events')
    plt.legend(loc="upper left")
    plt.savefig("../data/my_dataset/19_cc/lstm/gt_vs_pred.png", dpi=1200, bbox_inches='tight', pad_inches=0.5)
    plt.show()
