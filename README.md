# Predictive Maintenance with Multivariate Time Windows

A portfolio-style deep learning project that evolves from an academic predictive maintenance exercise into a more rigorous study with temporal validation, early warning modeling, and alert post-processing.

## Overview

This project studies predictive maintenance using multivariate time windows built from two sensor signals:

- `temperature`
- `vibration`

The original exercise was reworked into a more methodologically rigorous notebook.  
Instead of relying on random splits over overlapping windows, the final pipeline adopts:

- temporal train/validation/test split
- purge gap between splits
- normalization fitted only on training data
- threshold calibration on validation only
- final test evaluation performed only after model selection

The project compares two neural architectures:

- **LSTM**
- **1D CNN**

and then extends the problem into two practically different formulations:

- **Exact-step forecasting**: predict whether failure occurs at one exact future step
- **Early warning**: predict whether at least one failure occurs within a future interval

Finally, the project applies simple temporal post-processing to evaluate whether the alert stream can be made more stable over time.

---

## Objectives

This notebook was designed to answer four main questions:

1. Can a cleaner temporal validation protocol make the experiment more trustworthy?
2. Between LSTM and 1D CNN, which architecture works better as a baseline?
3. Is exact-step forecasting the most useful formulation for this dataset?
4. Can simple temporal smoothing improve the operational behavior of alerts?

---

## Methodological highlights

The final pipeline includes:

- **Temporal split** into train / validation / test
- **Purge gap** between splits to reduce leakage from overlapping windows
- **Normalization fitted only on training data**
- **Windowing after the split**
- **Class weights computed only on training data**
- **Threshold calibration on validation only**
- **Final evaluation on test only after model selection**

This makes the notebook much more rigorous than the original academic version.

---

## Problem formulation

### Input

Each sample is a multivariate time window with shape:

`(BACK_LOG, 2)`

where:

- `BACK_LOG = 50`
- feature 1 = `temperature`
- feature 2 = `vibration`

### Target definitions

The project explores two different target definitions.

#### 1. Exact-step forecasting

Given the last 50 time steps, predict whether failure occurs at one exact future step.

Examples:

- `horizon = 1`
- `horizon = 5`
- `horizon = 10`

#### 2. Early warning

Given the last 50 time steps, predict whether **at least one failure** will occur within the next `k` steps.

Examples:

- `warning_window = 5`
- `warning_window = 10`

This second formulation is more aligned with practical maintenance alerting scenarios.

---

## Models

### Baseline models

- **LSTM**
  - `Input -> LSTM(50) -> Dropout(0.2) -> Dense(1, sigmoid)`

- **1D CNN**
  - `Input -> Conv1D(64, kernel_size=3, relu) -> MaxPooling1D(2) -> Flatten -> Dropout(0.2) -> Dense(1, sigmoid)`

### Optimization setup

- Binary cross-entropy
- Adam optimizer
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- BackupAndRestore
- CSVLogger

---

## Experimental stages

### 1. Clean baseline with temporal validation

The project first compares LSTM and 1D CNN using the same clean temporal protocol.

**Main outcome:**  
The **1D CNN** consistently outperformed the LSTM and became the main baseline for the rest of the notebook.

### 2. Multi-step exact forecasting

The best baseline architecture (1D CNN) is tested under exact-step forecasting for:

- `horizon = 1`
- `horizon = 5`
- `horizon = 10`

**Main outcome:**  
Performance degraded as the exact forecast horizon increased.  
This showed that predicting the exact failure timestamp becomes increasingly difficult and brittle in this dataset.

### 3. Early warning

The same 1D CNN is then evaluated under an early warning formulation using:

- `warning_window = 5`
- `warning_window = 10`

**Main outcome:**  
Reformulating the target as a future warning interval produced much stronger operational results than exact-step forecasting.

### 4. Simple temporal post-processing of alerts

The best early warning model is used as the base alert generator, and two causal post-processing strategies are tested:

- **Causal moving average on scores**
- **k-of-n rule on binary alerts**

**Main outcome:**  
Post-processing did **not** improve the best core predictive metric, but it did reduce alert fragmentation and made the alert stream more stable over time.

---

## Key conclusions

1. **Temporal rigor matters.**  
   A proper temporal split with gap and train-only normalization makes the experiment much more trustworthy.

2. **The 1D CNN was the best baseline.**  
   It consistently outperformed the LSTM under the clean evaluation protocol.

3. **Formulation mattered more than architecture.**  
   The largest gain did not come from changing the model, but from changing the predictive task from exact-step forecasting to early warning.

4. **Early warning was the most useful setup.**  
   In this project, the model became much more operationally relevant when asked to predict failure within a future interval rather than at one exact future step.

5. **Simple alert smoothing improved stability, not predictive quality.**  
   Post-processing helped reduce noisy alert fragmentation, even though it did not surpass the raw early warning model in classification performance.

---

## Reproducibility and persistence

The notebook was structured to save experiment artifacts to Google Drive so training can be resumed or loaded later without retraining everything from scratch.

Each experiment stores artifacts such as:

- `best_model.keras`
- `last_model.keras`
- `history.csv`
- `thresholds.csv`
- `summary.csv`
- `metrics.json`
- `backup_state/`

This allows interrupted Colab sessions to be recovered more safely.

---

## Project structure

A typical project structure looks like this:

```text
project_root/
│
├── PORTIFOLIO_manut_preditiv.ipynb
├── portifolio_manut_preditiv.py
├── README.md
│
└── experimentos/
    ├── baseline_lstm_h1/
    ├── baseline_cnn_h1/
    ├── multistep_cnn_h1/
    ├── multistep_cnn_h5/
    ├── multistep_cnn_h10/
    ├── earlywarning_cnn_w5/
    ├── earlywarning_cnn_w10/
    └── reports/
```

---

## How to run

### Recommended environment

- Python 3.10+
- Google Colab
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- gdown

### Basic execution flow

1. Mount Google Drive
2. Load the dataset
3. Run the preprocessing pipeline
4. Build temporal windows
5. Train or load saved models
6. Evaluate on validation and test
7. Run multi-step and early warning sections
8. Compare formulations
9. Run post-processing on the best early warning model

---

## Loading saved models instead of retraining

The notebook supports loading previously saved experiments from Drive.

### Multi-step and early warning

When calling the experiment functions, use:

```python
force_retrain=False
evaluate_only_if_exists=True
```

This makes the notebook load the best saved model from Drive when available.

### Baseline models

For the baseline LSTM and CNN, the safest approach is to load them manually:

```python
lstm_paths = make_experiment_paths("baseline_lstm_h1")
cnn_paths = make_experiment_paths("baseline_cnn_h1")

LSTM_Model = tf.keras.models.load_model(str(lstm_paths["best_model_path"]))
CNN1D_Model = tf.keras.models.load_model(str(cnn_paths["best_model_path"]))
```

---

## Metrics

Because this is an imbalanced rare-event problem, the project does **not** rely on accuracy alone.

The most important metrics are:

- Positive-class precision
- Positive-class recall
- Positive-class F1
- ROC-AUC
- PR-AUC

For alert-stream analysis, the project also tracks:

- alert rate
- alert bursts
- alert transitions

---

## Limitations

- Synthetic dataset
- Only two sensor variables
- Limited architecture benchmark
- No production-grade alert policy
- Precision is still below what would be desirable for a real industrial setting

---

## Future work

Possible next steps include:

- adding derived temporal features
- testing stronger sequence architectures
- jointly optimizing threshold and post-processing rules
- adding more operational metrics such as lead time and false alarms per segment
- evaluating more realistic maintenance policies

---

## License

This repository is intended for academic and portfolio purposes.
