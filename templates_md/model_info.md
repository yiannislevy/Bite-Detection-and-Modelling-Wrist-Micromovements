# Experiment Overview

## Basic Information
- **Date of Experiment**: `02-02-2024`
- **Data Type**: `Sessions`
- **Directory Path**: `models/full_loso/majority_label/processed/std_1`
- **Data Used**: `data/ProcessedSubjects/MajorityLabel/sessions/grav_n_med/full_std_1`

## Preprocessing Details
- **Standardization**:
  - File: `data/dataset-info-json/mean_std_values.json`
- **Labeling Approach**:
  - Majority Labels: `Yes`
- **Median Filtering**: `Applied`
- **Gravity Component Removal**: `Yes`

[//]: # (## Data Structure)

[//]: # (- **Timestamps Included**: `Yes` | `No`)
[//]: # (- **Window Structure**: `20x6` | `20x7` &#40;indicate if timestamps are included&#41;)

## Additional Notes
First 'processed' model training (with gravity removal & median filtering). Used the first std&mean values.
 
## Results Summary
- **Easy Accuracy**: 79.77%
- **Loss**: 62.28%
- **Loss @ 32nd epoch**: 51.1%
- **Training Accuracy**: 80.43%

[//]: # (## Future Considerations)

[//]: # (&#40;Suggest any future tests, changes, or improvements that could be made based on the current experiment’s outcomes. This could include trying different preprocessing steps, adjusting parameters, or testing with different data sets.&#41;)

