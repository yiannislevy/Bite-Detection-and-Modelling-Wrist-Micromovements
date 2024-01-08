import matplotlib.pyplot as plt


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
