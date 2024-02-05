# Experiment Documentation

## Introduction
This is the directory where the cnn models are stored along with their training info (for most) as well as markdown info
files with experiment details, results and future considerations.

## Directory Structure
Each subdirectory contains trained models and data for a specific experiment.

### Full Leave-One-Subject-Out (full_loso)
- [Majority Label](./full_loso/majority_label)
  - [Processed Data](./full_loso/majority_label/processed)
    - [Standardization 1](./full_loso/majority_label/processed/std_1)
      - [Model and Training Details](./full_loso/majority_label/processed/std_1/model_info.md)
    - [Standardization 2](./full_loso/majority_label/processed/std_2)
      - [Model and Training Details](./full_loso/majority_label/processed/std_2/model_info.md)
    - [Standardization 3](./full_loso/majority_label/processed/std_3)
      - [Model and Training Details](./full_loso/majority_label/processed/std_3/model_info.md)
  - [Unprocessed Data](./full_loso/majority_label/unprocessed)
    - [First Original](./full_loso/majority_label/unprocessed/first_og)
      - [Model and Training Details](./full_loso/majority_label/unprocessed/first_og/model_info.md)
    - [With Gravity No Filter](./full_loso/majority_label/unprocessed/with_grav_no_filter_plain_stdmean)
      - [Model and Training Details](./full_loso/majority_label/unprocessed/with_grav_no_filter_plain_stdmean/model_info.md)
- [Mid Timestamp Labeling](./full_loso/mid_timestamp_labeling)
  - [Model and Training Details](./full_loso/mid_timestamp_labeling/model_info.md)

# Future Reference

- We will keep the `majority_label/processed/std_3`. In its `model_info.md` file are contained future considerations.
- The `old_models` subdirectory contains the very first models trained to test basic stuff. Kept for reference and 
historic reasons.

<!-- Each experiment section should follow the format below -->

## Experiment: [Experiment Name]
### Basic Information
- **Date of Experiment**: `Date`
- **Data Type**: `Type of Data`
- **Directory Path**: `Path to the specific experiment's folder`

### Preprocessing Details
- **Standardization File**: `Path to the standardization file`
- **Labeling Approach**: `Approach used`
- **Median Filtering**: `Specify if applied`
- **Gravity Component Removal**: `Yes or No`

### Results Summary
- **Accuracy**: `Value`
- **Loss**: `Value`
- **Epoch Specific Metrics**: `Details if necessary`
- **Training Accuracy**: `Value`

### Analysis
Detailed analysis of the experiment's results.

### Future Considerations
Recommendations for future experiments, any changes to be considered, and hypotheses to be tested.

### Additional Notes
Any other relevant information or personal notes regarding this experiment.

## Conclusion
Summarize the overall findings and the next steps.

## Appendices
- **Appendix A**: Reference to any external documents or datasets.
- **Appendix B**: Additional graphs or tables if needed.

<!-- Repeat the experiment section for each experiment -->

