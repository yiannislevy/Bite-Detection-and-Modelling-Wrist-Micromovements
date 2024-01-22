import numpy as np


def calculate_bite_stats(bite_gt):
    bite_lengths = [bite[1] - bite[0] for meal in bite_gt for bite in meal]
    bite_lengths = np.array(bite_lengths)

    avg_bite_length = np.mean(bite_lengths)
    std_dev_bite_length = np.std(bite_lengths)

    Q1 = np.quantile(bite_lengths, 0.25)
    Q3 = np.quantile(bite_lengths, 0.75)
    IQR = Q3 - Q1

    # Define thresholds for bite length
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR

    filtered_bite_lengths = bite_lengths[(bite_lengths >= lower_threshold) & (bite_lengths <= upper_threshold)]

    # Calculate stats for filtered bite lengths
    avg_filtered_bite_length = np.mean(filtered_bite_lengths)
    std_dev_filtered_bite_length = np.std(filtered_bite_lengths)
    median_filtered_bite_length = np.median(filtered_bite_lengths)
    min_filtered_bite_length = np.min(filtered_bite_lengths)
    max_filtered_bite_length = np.max(filtered_bite_lengths)

    # Print stats
    return {
        'Original Mean': avg_bite_length,
        'Original Std Dev': std_dev_bite_length,
        'Lower Bound': lower_threshold,
        'Upper Bound': upper_threshold,
        'Filtered Mean': avg_filtered_bite_length,
        'Filtered Std Dev': std_dev_filtered_bite_length,
        'Filtered Median': median_filtered_bite_length,
        'Filtered Min': min_filtered_bite_length,
        'Filtered Max': max_filtered_bite_length
    }
