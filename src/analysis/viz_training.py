import matplotlib.pyplot as plt


def plot_metric(metric_values, metric_name, save_path):
    """
    Plots a single metric (e.g., accuracy or loss) over epochs and saves the plot as an SVG file.

    Parameters:
    - metric_values: List of metric values over epochs.
    - metric_name: The name of the metric (e.g., 'Accuracy', 'Loss').
    - save_path: Path to save the SVG plot.
    """
    epochs = range(1, len(metric_values) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, metric_values, label=metric_name, color='blue' if metric_name == 'Accuracy' else 'red')
    plt.title(f'Training {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()

    # Save the figure as an SVG for high-quality scaling
    plt.savefig(save_path, format='svg')
    # plt.close() # Optional: uncomment to close the plot after saving to preserve memory
