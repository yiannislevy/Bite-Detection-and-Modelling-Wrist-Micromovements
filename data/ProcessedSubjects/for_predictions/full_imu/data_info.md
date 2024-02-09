# Experiment Overview

## Basic Information
- **Date of Experiment**: `07-02-2024`
- **Data Type**: `Sessions`
- **Directory Path**: `data/ProcessedSubjects/for_predictions/full_imu`

## Preprocessing Details
- **Standardization**:
  - `data/dataset-info-json/mean_std_values_3.json`
- **Labeling Approach**:
  - Middle Sample Approach: `Not Applied`
  - Majority Labels: `Not Applied`
- **Median Filtering**: `Applied`
- **Gravity Component Removal**: `Yes`

## Data Structure

- **Timestamps Included**: `Yes`
- **Window Structure**: `20x7` (indicate if timestamps are included)
- **Labels Structure** N/A 
- 
## Additional Notes
Raw complete imu data from start to finish as if they were straight from an unknown source.
These are data to be used for predictions. They include the 'other' micromovement, no trimming or labels included.

## Results Summary
Did `model.evaluate`.
Used models: `../models/full_loso/majority_label/processed/std_3/`
- Results:
  - **Easy Accuracy**: 51.42%
  - **Loss**: 30.27%

[//]: # (## Future Considerations)

[//]: # (&#40;Suggest any future tests, changes, or improvements that could be made based on the current experiment’s outcomes. This could include trying different preprocessing steps, adjusting parameters, or testing with different data sets.&#41;)

