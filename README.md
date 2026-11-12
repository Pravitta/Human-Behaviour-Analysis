# 🧠 Human Behaviour Analysis — EEG Cognitive Workload Dashboard

An end-to-end Machine Learning and Visual Analytics platform designed to evaluate **cognitive workload levels** from multi-channel EEG recordings using the benchmark **STEW (Simultaneous Task EEG Workload) Dataset**.

---

## 📌 Project Overview

This project provides a complete machine learning pipeline and a sleek, interactive frontend dashboard (Navy Blue Theme) tailored for cognitive workload assessment. It processes 14-channel electroencephalogram (EEG) signals, extracts frequency-domain band power representations, and performs classification to discriminate between **Rest (Low Workload)** and **Multitasking (High Cognitive Workload)** mental states.

---

## 🏗️ Architecture & Features

```
WORKLOAD_RIO/
├── dashboard.html                  # Interactive frontend dashboard (Chart.js + Native FFT)
├── run_inference.py                # Batch feature extraction & inference pipeline
├── stew_logistic_regression.joblib # Trained scikit-learn Logistic Regression Pipeline
├── results.json                    # Precomputed inference results (all 48 subjects, 96 conditions)
├── README.md                       # Documentation
└── STEW_Dataset/                   # Raw EEG text data (ignored from git)
```

### Key Highlights
- **ML Pipeline**: StandardScaler normalization + Logistic Regression (`lbfgs`, C=1.0) saved via `joblib`.
- **Signal Processing**: Non-overlapping 4.0-second sliding windows (512 samples at 128 Hz) converted to power spectral density via Welch PSD.
- **70 EEG Features**: 5 frequency bands ($\delta, \theta, \alpha, \beta, \gamma$) $\times$ 14 electrode channels (Emotiv EPOC layout).
- **Interactive UI**:
  - Live EEG file upload (`.txt`) with client-side Cooley-Tukey FFT inference.
  - Multi-subject browser filtering by condition (`Rest` vs `Multitasking`).
  - Dynamic KPI cards displaying Prediction, Confidence, Model Accuracy, and Self-Rated Ratings (SIMKAP task difficulty & fatigue).
  - Window-by-window P(Multitasking) probability timeline with 50% threshold line.
  - EEG band power bar distributions and window-level prediction donuts.

---

## 🔬 Dataset & Preprocessing Details

- **Dataset**: [STEW (Simultaneous Task EEG Workload) Dataset](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset)
- **Subjects**: 48 participants recorded across two conditions:
  - `_lo`: Baseline Rest condition (Class `0`)
  - `_hi`: Multitasking SIMKAP test condition (Class `1`)
- **Sampling Frequency**: $128\text{ Hz}$
- **Channels (14)**: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
- **Frequency Bands**:
  - **Delta ($\delta$)**: 1 – 4 Hz
  - **Theta ($\theta$)**: 4 – 8 Hz
  - **Alpha ($\alpha$)**: 8 – 13 Hz
  - **Beta ($\beta$)**: 13 – 30 Hz
  - **Gamma ($\gamma$)**: 30 – 45 Hz

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install required Python packages:

```bash
git clone https://github.com/Pravitta/Human-Behaviour-Analysis.git
cd Human-Behaviour-Analysis
pip install scikit-learn numpy scipy joblib
```

### 2. Run Offline Feature Extraction & Model Inference
To re-run inference across the entire STEW dataset and regenerate `results.json`:

```bash
python run_inference.py
```

### 3. Launch the Dashboard
Serve the workspace over a local HTTP server (required to avoid browser CORS restrictions when fetching `results.json`):

```bash
python -m http.server 8080
```

Navigate to:
```
http://localhost:8080/dashboard.html
```

---

## 🖥️ Using the Dashboard

1. **Dataset Browser**: Select any subject (S01 to S48) on the left sidebar to inspect ground truth, model classification, confidence, and self-reported difficulty scores.
2. **Filter Conditions**: Toggle between `All`, `🔴 Multitasking`, and `🟢 Rest` to quickly view condition-specific results.
3. **Live File Upload**: 
   - Click **Select EEG file** under *Cognitive Workload Analysis*.
   - Upload any raw 14-channel `.txt` recording.
   - Hit **Analyze Workload** to compute frequency band power and display real-time predictions and confidence scores.

---

## 📈 Model Performance Note

Inter-subject EEG variability is a well-documented phenomenon in cognitive state estimation. While the generalized model achieves up to **100% accuracy** on specific participants (e.g., `sub23_hi`, `sub38_hi`, `sub40_lo`), domain adaptation and subject-specific calibration curves yield the strongest classification boundaries.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
