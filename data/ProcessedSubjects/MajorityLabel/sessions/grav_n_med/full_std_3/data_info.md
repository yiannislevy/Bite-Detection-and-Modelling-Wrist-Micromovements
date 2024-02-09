# Experiment Overview

## Basic Information
- **Date of Experiment**: `03-02-2024`
- **Data Type**: `Sessions`
- **Directory Path**: `data/ProcessedSubjects/MajorityLabel/sessions/grav_med_meanstd3`

## Preprocessing Details
- **Standardization**:
  - File: `data/dataset-info-json`
- **Labeling Approach**:
  - Majority Labels: `Yes`
  - Mid-Window Sample Labeling: `No`
- **Median Filtering**: `Applied`
- **Gravity Component Removal**: `Yes`

## Data Structure
- **Timestamps Included**: `Yes`
- **Window Structure**:  `20x7` (indicate if timestamps are included)

## Additional Notes
Data prepared according to instructions from Kyritsis from call. Removed gravity, median filtering 5th order, calculated
mean&std values after the previous 2 steps -> not windowed though - sequenced one after the other.
## Results Summary
Did `model.evaluate`.
Used models: `../models/full_loso/majority_label/processed/std_3/`
- Results:
  - **Easy Accuracy**: 79.57%
  - **Loss**: 63.53%
  - 
[//]: # (&#40;Provide a brief overview of the results obtained from this preprocessing setup or experiment. Highlight any significant findings, challenges encountered, or any adjustments made to the initial plan based on outcomes.&#41;)

[//]: # (## Future Considerations)

[//]: # (&#40;Suggest any future tests, changes, or improvements that could be made based on the current experiment’s outcomes. This could include trying different preprocessing steps, adjusting parameters, or testing with different data sets.&#41;)
