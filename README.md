# Bite Detection 

## Description

This project aims to reconstruct the model described in the paper: `Modeling Wrist Micromovements to Measure In-Meal 
Eating Behavior From Inertial Sensor Data` which can be found [here](https://doi.org/10.1109/jbhi.2019.2892011).

### Steps:

1. Process publicly available [dataset](https://zenodo.org/records/4421861).
2. Train CNN for micromovement prediction.
3. Process said predictions accordingly for LSTM
4. Train LSTM

## Details

The dataset `FIC` provided is a dictionary. Contains following keys:

- `subject_id`: Integers on subject ids.
- `session_id`: Integer on session ids.
- `signals_raw`: Raw unprocessed imu.
- `signals_proc`: Processed imu.
- `meal_gt`: Ground truth on micromovements.
- `bite_gt`: Ground truth on bite events.

### Step 1:

- First I paired subjects with sessions in the dataset in a [json](data/dataset-info-json/subject_to_indices.json) for 
quick reference.
- Necessary steps of further processing where required, described further in [cnn_steps.py](src/cnn/cnn_steps.py).
- Those functions where utilized in a streamlined manner from the pipeline [cnn_preprocessing.ipynb](notebooks/cnn/cnn_preprocessing.ipynb).
- Other functions where also used for the above preprocessing, which can be found under `src/utils` (e.g. `data_io.py`)

### Step 2:

- [train_cnn.py](src/cnn/train_cnn.py): contains the logic and functions to train the models in a LOSO manner.
- [train_cnn.ipynb](notebooks/cnn/train_cnn.ipynb): pipelined version to execute and monitor progress easier as well as 
stats and more.
- [ProbabilityDistributionVix.ipynb](notebooks/cnn/ProbabilityDistributionViz.ipynb): notebook for predicting and 
visualizing the predictions with custom visualization functions called.
- [visualisation.py](src/analysis/viz_predictions.py): custom functions for visualising predictions' temporal evolution.

### Step 3:

- Preprocessing for the LSTM required meticulous work, as it required obtaining windows of data for positive and 
negative examples of a bite, matching predictions and with windows corresponding to actual lengths of bite or their 
inbetween timeframes for non-bites.
- [lstm_preprocessing.ipynb](notebooks/lstm/lstm_preprocessing.ipynb): contains the pipeline for the complete preprocessing.


### Step 4:

- [lstm_steps.py](src/lstm/lstm_steps.py): contains logic for positive and negative bites and some helper functions.
- [train_lstm.ipynb](notebooks/lstm/train_lstm.ipynb): contains the pipeline to
  - train and evaluate in non-academic loso
  - train academically 1 full model and further tuning options
  - load & save from checkpoint
  - plot
