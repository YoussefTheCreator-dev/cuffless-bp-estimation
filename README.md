# Wearable-Enabled AI for Cuffless Blood Pressure Estimation and Hypertension Risk Screening

**Future17 SDG Challenge 2025 | Ekak Innovations | Team 04**
**SDG 3: Good Health and Well-Being | SDG 9: Industry, Innovation and Infrastructure**

---

## Overview

This project delivers an AI-powered, cuffless blood pressure (BP) monitoring system designed to make cardiovascular
diagnostics accessible and affordable in low-resource settings. Instead of expensive clinical equipment, it uses a
Photoplethysmogram (PPG) signal — the same sensor found in smartwatches and $5 pulse oximeters — to predict
Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP).

**The problem it solves:** Hypertension affects 1.28 billion people worldwide, yet many live in regions where
blood pressure cuffs and clinical visits are inaccessible or unaffordable. This system turns a low-cost wearable
into a diagnostic-grade BP monitor.

---

## How It Works

```
PPG Signal (5-second window @ 125 Hz)
        ↓
Bandpass Filter (0.5–8.0 Hz, removes motion/baseline noise)
        ↓
Min-Max Normalisation (per window)
        ↓
Bidirectional LSTM Model
        ↓
SBP & DBP prediction (mmHg)
```

---

## Dataset

**Cuff-Less Blood Pressure Estimation Dataset** (Moody et al.)
- Source: Kaggle — `mkachuee/cuff-less-blood-pressure-estimation`
- Signals: PPG (channel 0), ABP/arterial blood pressure (channel 1), ECG (channel 2)
- Sampling rate: 125 Hz
- Labels: SBP = 95th percentile of ABP window; DBP = 5th percentile
- Physiological validity filter: 60 < SBP < 200, 40 < DBP < 120, SBP > DBP + 20

Place `part_1.mat` (450 MB) in the project root before running preprocessing.
The `.mat` file is excluded from git due to its size.

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Type | Bidirectional LSTM (BiLSTM) |
| Input | 625 samples × 1 channel (5-second PPG window) |
| Hidden size | 32 units per direction |
| LSTM layers | 1 |
| Output | 2 values (SBP, DBP) in normalised space |
| Loss | Mean Squared Error (MSE) |
| Optimiser | Adam (lr = 0.001) |
| Scheduler | Cosine Annealing (T_max = 50) |
| Data split | 70% train / 15% val / 15% test |

---

## Results

Evaluated against the **British Hypertension Society (BHS) Protocol**:

| Metric | Value | BHS Grade |
|--------|-------|-----------|
| SBP MAE | ~1.7 mmHg | **A** (≤5 mmHg) |
| DBP MAE | ~3.7 mmHg | **A** (≤5 mmHg) |

BHS Grade A is the highest clinical accuracy standard — equivalent to a hospital-grade sphygmomanometer.

---

## Repository Structure

```
medicheck-ai/
├── load_data.py         # Visualise raw PPG/ABP/ECG signals from .mat file
├── preprocess.py        # Extract 5-second windows and BP labels → processed_data.npz
├── train_model.py       # Train BiLSTM and evaluate on test set
├── requirements.txt     # Python dependencies
├── bp_model.pth         # Final model weights (after full training)
├── bp_model_best.pth    # Best checkpoint (lowest validation loss)
├── y_scaler.pkl         # Label scaler for inverse-transforming predictions
├── processed_data.npz   # Preprocessed windows (excluded if large)
└── part_1.mat           # Raw dataset — NOT included (download separately)
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Visualise raw signals (optional)
```bash
python load_data.py --mat_path part_1.mat
```

### 3. Preprocess the dataset
```bash
python preprocess.py
```
Outputs `processed_data.npz` (~12 MB) with windowed PPG segments and SBP/DBP labels.

### 4. Train and evaluate the model
```bash
python train_model.py
```
Saves `bp_model_best.pth`, `bp_model.pth`, `y_scaler.pkl`, and `bp_results.png`.

---

## Impact

| Existing solutions | MediCheck AI |
|---|---|
| Hospital cuff — requires clinic visit | Works on a $5 pulse oximeter or smartwatch |
| Home monitors — ~$50–200, needs literacy | Runs on any device with a PPG sensor |
| Wearable apps — cloud-dependent | Runs locally, offline-capable |

**Cost estimate for a standalone device:** ~$18 (PPG sensor + microcontroller)
vs. $500+ for a clinical-grade monitor.

---

## Team

**Team 04 — Future17 SDG Challenge 2025**
Partner organisation: Ekak Innovations (Contact: Shashank Misra, smisra@ekak.in)
