import numpy as np
import json


def calculate_mean_std(data):
    means = np.mean(data[:, 1:], axis=0)
    std_devs = np.std(data[:, 1:], axis=0)

    mean_std = {'means': means.tolist(), 'std_devs': std_devs.tolist()}
    path_to_save = "../data/dataset-info-json"
    with open(f'{path_to_save}/mean_std_values_3.json', 'w') as f:
        json.dump(mean_std, f)

    return mean_std


# Example usage:
# standardized_data = standardize_data(your_data_structure, 'mean_std_values.json')
def standardize_data(data, json_file):
    with open(json_file, 'r') as f:
        mean_std = json.load(f)

    means = np.array(mean_std['means'])
    std_devs = np.array(mean_std['std_devs'])

    standardized_data = (data - means) / std_devs
    return standardized_data
