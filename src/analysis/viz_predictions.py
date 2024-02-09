import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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