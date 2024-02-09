# Experiment Overview

## Basic Information
- **Date of Experiment**: `09-02-2024`
- **Data Type**: `Sessions`
- **Directory Path**: [data/lstm_training_data/raw](data/lstm_training_data/raw)

## Preprocessing Details
- **Standardization**:
  - File: `data/dataset-info-json/mean_std_values_3.json`
- **Labeling Approach**:
  - N/A`
- **Median Filtering**: `Applied`
- **Gravity Component Removal**: `Yes`
- **Balanced**: `Yes`

## Data Structure

- **Timestamps Included**: `Yes`
- **Window Structure**: `K*Nx6` (sessions * window_size * 6)

## Additional Notes
Positive and negative example bites along with their labels combined ready for training. 6th column included with 
timestamp. Discard for training. Balancing was done to include same number of positive as negative examples per session.

[//]: # (## Results Summary)

[//]: # (## Future Considerations)

[//]: # (&#40;Suggest any future tests, changes, or improvements that could be made based on the current experiment’s outcomes. This could include trying different preprocessing steps, adjusting parameters, or testing with different data sets.&#41;)

