# Human Behaviour Analysis — EEG Cognitive Workload Dashboard

A machine learning dashboard for analysing **cognitive workload** from EEG signals using the [STEW Dataset](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset).

---

## 📁 Project Structure

```
WORKLOAD_RIO/
├── dashboard.html                  ← Interactive web dashboard
├── run_inference.py                ← EEG feature extraction + model inference
├── stew_logistic_regression.joblib ← Pre-trained sklearn pipeline
├── results.json                    ← Inference output (generated)
└── STEW_Dataset/                   ← Raw EEG data (not tracked by git)
```

---

## 🧠 Model

| Property | Value |
|---|---|
| Type | Logistic Regression (sklearn Pipeline) |
| Preprocessing | StandardScaler |
| Features | 70 (5 EEG bands × 14 channels) |
| Bands | Delta, Theta, Alpha, Beta, Gamma |
| Sampling rate | 128 Hz |
| Window size | 4 seconds (non-overlapping) |
| Classes | Rest (0) · Multitasking (1) |
| Overall accuracy | 42.26% (high inter-subject variability) |

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install scikit-learn scipy numpy joblib
```

### 2. Download the STEW Dataset
Place the extracted dataset at `STEW_Dataset/STEW Dataset/` (`.txt` subject files + `ratings.txt`).

### 3. Run inference
```bash
python run_inference.py
```
This generates `results.json` with predictions for all 96 subject recordings.

### 4. Launch the dashboard
```bash
python -m http.server 8080
```
Then open **http://localhost:8080/dashboard.html** in your browser.

---

## 📊 Dashboard Features

- **Cognitive / Activity / Emotion** tab navigation
- **Subject browser** — filter by Rest or Multitasking condition
- **Live EEG upload** — drop any `.txt` EEG file for instant inference
- **Probability timeline** — P(Multitasking) across 4s windows
- **Band power chart** — Δ θ α β γ power distribution
- **Prediction split donut** — window-level Rest vs Multitasking breakdown
- **Self-ratings** — participant difficulty & fatigue scores

---

## 📌 Notes

- The STEW dataset contains EEG from 48 subjects performing rest and multitasking (SIMKAP) tasks.
- Inter-subject variability is high — some subjects reach 100% accuracy, others near 0%.
- Subject-specific models typically improve accuracy significantly.
