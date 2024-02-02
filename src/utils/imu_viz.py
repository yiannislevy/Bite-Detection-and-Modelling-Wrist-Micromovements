from matplotlib import pyplot as plt
import numpy as np
import matplotlib.dates as mdates


def plot_raw_sensor_np(data, title, sensor_type='accelerometer'):
    """
    Plot sensor data over time for a NumPy array format.

    Args:
        data (numpy.ndarray): Sensor data to be plotted, with shape (N, 7),
                              where columns are [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].
        title (str): Title for the plot.
        sensor_type (str): Type of sensor data to plot ('accelerometer' or 'gyroscope').
                           Determines which columns to plot.

    Plots:
        Matplotlib figure displaying the sensor data for each axis over time.
    """
    # Extract timestamps
    timestamps = data[:, 0]
    # Convert timestamps from float to numpy datetime64
    times = np.array([np.datetime64(int(ts), 's') for ts in timestamps])

    # Determine which columns to plot based on sensor type
    if sensor_type == 'accelerometer':
        x, y, z = data[:, 1], data[:, 2], data[:, 3]
        sensor_name = 'Accelerometer'
    elif sensor_type == 'gyroscope':
        x, y, z = data[:, 4], data[:, 5], data[:, 6]
        sensor_name = 'Gyroscope'
    else:
        raise ValueError("sensor_type must be either 'accelerometer' or 'gyroscope'")

    plt.figure(figsize=(12, 8))
    plt.plot(times, x, label='x')
    plt.plot(times, y, label='y')
    plt.plot(times, z, label='z')

    # Format the time axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(f'{title}: {sensor_name} Data over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()
