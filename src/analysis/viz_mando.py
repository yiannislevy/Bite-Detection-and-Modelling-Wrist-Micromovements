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
