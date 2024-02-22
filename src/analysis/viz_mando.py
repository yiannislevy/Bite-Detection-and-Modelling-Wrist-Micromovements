import matplotlib.pyplot as plt


def plot_meal_progression(data):
    """
    Plot the meal weight over time using matplotlib. This function displays the meal's total weight
    and decreases it linearly during each bite event to represent the consumption of food.

    Vertical lines are drawn at the start of each bite event to visually indicate when a bite begins.
    The meal's weight at any given time is depicted with a green line that steps down at the end of
    each bite. The line segments between the start and end of a bite show the gradual weight loss
    due to the food being removed from the plate.

    Parameters:
    - data (numpy.ndarray): A 2D numpy array where each row contains the following columns:
        - Weight of the bite (g)
        - Start time of the bite (s_t)
        - End time of the bite (e_t)

    The array is expected to be sorted by the start time of each bite.

    Returns:
    - A matplotlib figure and axes displaying the meal progression plot.

    Usage Example:
    --------------
    # Assuming `data` is a numpy array with the correct format
    fig, ax = plot_meal_progression_final(data)
    plt.show()
    """
    initial_weight = data[:, 0].sum()
    current_weight = initial_weight
    times = [0]  # Start from time 0
    weights = [initial_weight]  # Start with initial weight

    for row in data:
        weight, start_t, end_t = row
        # Maintain weight until the start of the bite
        times.append(start_t)
        weights.append(current_weight)

        # Decrease weight linearly during the bite
        times.append(end_t)
        current_weight -= weight
        weights.append(current_weight)

    # Extend the last point for visualization
    times.append(times[-1] + 10)
    weights.append(current_weight)

    fig, ax = plt.subplots()
    ax.plot(times, weights, label='Meal Weight', color='green')

    # Add vertical lines for bite start times
    for start_t in data[:, 1]:
        ax.axvline(x=start_t, color='red', linestyle='--', linewidth=1, ymin=0, ymax=1)

    ax.set_title('Meal Weight Over Time')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Weight (g)')
    plt.legend()

    return fig, ax


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
