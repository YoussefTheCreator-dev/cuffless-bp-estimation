# MediCheck AI - BP Estimation from PPG

This project implements Blood Pressure (BP) estimation using Photoplethysmogram (PPG) signals with a Bidirectional LSTM model.

## Project Structure

- `preprocess.py`: Processes raw `.mat` data into overlapping windows and extracts SBP/DBP labels.
- `train_model.py`: Trains a BiLSTM model on the preprocessed data.
- `load_data.py`: Utility script to visualize raw signals.
- `requirements.txt`: Python dependencies.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place `part_1.mat` in the project root.

3. Run preprocessing:
   ```bash
   python preprocess.py
   ```

4. Run training:
   ```bash
   python train_model.py
   ```

## Model
The model is a Bidirectional LSTM that takes 5-second PPG windows (at 125Hz) and predicts Systolic and Diastolic Blood Pressure.
