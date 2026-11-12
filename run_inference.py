"""
STEW EEG Cognitive Workload – Inference Pipeline
Reads all subject files, extracts band-power features, runs the
pre-trained logistic-regression model, and writes results to JSON.
"""

import os, json, warnings
import numpy as np
import joblib
from scipy.signal import welch

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "STEW_Dataset", "STEW Dataset")
MODEL_PATH  = os.path.join(BASE_DIR, "stew_logistic_regression.joblib")
OUTPUT_PATH = os.path.join(BASE_DIR, "results.json")

# ── Load model metadata ────────────────────────────────────────────────────────
print("Loading model …")
artifact      = joblib.load(MODEL_PATH)
pipeline      = artifact["model"]
FS            = artifact["sampling_rate"]       # 128 Hz
WIN_SEC       = artifact["window_seconds"]       # 4 s
STEP_SEC      = artifact["step_seconds"]         # 4 s
BANDS         = artifact["bands"]               # dict of (lo, hi)
N_CHANNELS    = artifact["channel_count"]       # 14
CLASS_LABELS  = artifact["class_labels"]        # {0:'Rest', 1:'Multitasking'}

WIN_SAMPLES  = int(WIN_SEC  * FS)   # 512
STEP_SAMPLES = int(STEP_SEC * FS)   # 512

CHANNEL_NAMES = [
    "AF3","F7","F3","FC5","T7","P7",
    "O1","O2","P8","T8","FC6","F4","F8","AF4"
]

# ── Feature extraction ─────────────────────────────────────────────────────────
def bandpower(signal, fs, fmin, fmax):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

def extract_features(epoch):
    """epoch: (N_CHANNELS, WIN_SAMPLES) → feature vector of length 70"""
    feats = []
    for ch in range(N_CHANNELS):
        for band, (lo, hi) in BANDS.items():
            feats.append(bandpower(epoch[ch], FS, lo, hi))
    return np.array(feats)

# ── Process all subjects ───────────────────────────────────────────────────────
ratings_path = os.path.join(DATASET_DIR, "ratings.txt")
ratings = {}
with open(ratings_path) as f:
    for line in f:
        parts = [x.strip() for x in line.strip().split(",")]
        if len(parts) == 3:
            sub_id, difficulty, fatigue = int(parts[0]), int(parts[1]), int(parts[2])
            ratings[sub_id] = {"difficulty": difficulty, "fatigue": fatigue}

subjects_data = []

files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".txt") and f.startswith("sub")])
print(f"Found {len(files)} subject files …")

for fname in files:
    parts   = fname.replace(".txt","").split("_")
    sub_id  = int(parts[0].replace("sub",""))
    cond    = parts[1]  # 'hi' or 'lo'
    true_label = 1 if cond == "hi" else 0

    fpath   = os.path.join(DATASET_DIR, fname)
    raw     = np.loadtxt(fpath)          # shape: (T, 14)
    raw     = raw.T                      # → (14, T)

    n_samples = raw.shape[1]
    windows, probs, preds = [], [], []

    start = 0
    while start + WIN_SAMPLES <= n_samples:
        epoch = raw[:, start:start + WIN_SAMPLES]
        feats = extract_features(epoch).reshape(1, -1)
        prob  = pipeline.predict_proba(feats)[0][1]      # P(Multitasking)
        pred  = int(pipeline.predict(feats)[0])
        windows.append(start / FS)                       # time in seconds
        probs.append(round(float(prob), 4))
        preds.append(pred)
        start += STEP_SAMPLES

    n_windows   = len(preds)
    n_correct   = sum(1 for p in preds if p == true_label)
    accuracy    = round(n_correct / n_windows * 100, 2) if n_windows else 0
    mean_prob   = round(float(np.mean(probs)), 4)

    # Band-power averages across the whole recording
    band_powers = {band: [] for band in BANDS}
    start = 0
    while start + WIN_SAMPLES <= n_samples:
        epoch = raw[:, start:start + WIN_SAMPLES]
        for b_idx, (band, (lo, hi)) in enumerate(BANDS.items()):
            bp = np.mean([bandpower(epoch[ch], FS, lo, hi) for ch in range(N_CHANNELS)])
            band_powers[band].append(float(bp))
        start += STEP_SAMPLES

    band_avg = {b: round(float(np.mean(v)), 4) for b, v in band_powers.items()}

    # Per-channel mean multitasking probability proxy (alpha suppression heuristic)
    ch_probs = []
    start = 0
    epoch_count = 0
    ch_acc = np.zeros(N_CHANNELS)
    while start + WIN_SAMPLES <= n_samples:
        epoch = raw[:, start:start + WIN_SAMPLES]
        for ch in range(N_CHANNELS):
            ch_feats = np.array([bandpower(epoch[ch], FS, lo, hi)
                                 for _, (lo, hi) in BANDS.items()]).reshape(1, -1)
        epoch_count += 1
        start += STEP_SAMPLES

    subjects_data.append({
        "subject":      sub_id,
        "condition":    cond,
        "true_label":   CLASS_LABELS[true_label],
        "n_windows":    n_windows,
        "accuracy":     accuracy,
        "mean_prob_multitasking": mean_prob,
        "windows_time": windows,
        "probabilities": probs,
        "predictions":  preds,
        "band_avg":     band_avg,
        "ratings":      ratings.get(sub_id, {}),
    })

    print(f"  sub{sub_id:02d}_{cond}: {n_windows} windows | acc={accuracy}% | mean_P={mean_prob:.3f}")

# ── Summary stats ──────────────────────────────────────────────────────────────
all_acc   = [s["accuracy"] for s in subjects_data]
overall   = round(float(np.mean(all_acc)), 2)

output = {
    "overall_accuracy": overall,
    "n_subjects":       len(set(s["subject"] for s in subjects_data)),
    "n_records":        len(subjects_data),
    "class_labels":     CLASS_LABELS,
    "channel_names":    CHANNEL_NAMES,
    "bands":            {k: list(v) for k,v in BANDS.items()},
    "subjects":         subjects_data
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅  Done! Overall accuracy: {overall}%")
print(f"📄  Results saved → {OUTPUT_PATH}")

