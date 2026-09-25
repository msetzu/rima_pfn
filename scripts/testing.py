import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tabpfn_client import set_access_token

import os

os.environ["TABPFN_TOKEN"] = "tabpfn_sk_zg3Z7dYugpXnnP9YmjDVhTGQI4YIH8-CWAgCmPkuLbE"
os.environ["TABPFN_ALLOW_LICENSE_DOWNLOAD"] = "1"
set_access_token("tabpfn_sk_zg3Z7dYugpXnnP9YmjDVhTGQI4YIH8-CWAgCmPkuLbE")

from tabpfn import TabPFNClassifier

# 1. Generate Synthetic Time-Series Data
# --------------------------------------------------
np.random.seed(42)
num_samples = 300
time_steps = 50

# Class 0: Sine wave with noise
# Class 1: Square-like wave with noise
X_time_series = []
y = []

for i in range(num_samples):
    label = np.random.choice([0, 1])
    t = np.linspace(0, 4 * np.pi, time_steps)

    if label == 0:
        signal = np.sin(t) + np.random.normal(0, 0.2, size=time_steps)
    else:
        signal = np.sign(np.sin(t)) + np.random.normal(0, 0.2, size=time_steps)

    X_time_series.append(signal)
    y.append(label)

X_time_series = np.array(X_time_series)
y = np.array(y)


# 2. Extract Features from Time-Series (Recommended for Tabular Models)
# --------------------------------------------------
def extract_time_series_features(X_raw):
    """Extract summary statistics across time steps for each sample."""
    features = []
    for sample in X_raw:
        sample_features = [
            np.mean(sample),
            np.std(sample),
            np.min(sample),
            np.max(sample),
            np.median(sample),
            np.percentile(sample, 25),
            np.percentile(sample, 75),
            # Trend / slope feature
            np.polyfit(np.arange(len(sample)), sample, 1)[0],
        ]
        features.append(sample_features)
    return np.array(features)


# Extract features or pass raw flattened sequences directly
X_features = extract_time_series_features(X_time_series)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, random_state=42, stratify=y
)

# 3. Initialize TabPFN for Local Inference
# --------------------------------------------------
# Setting device='auto' or device='cuda' / 'cpu' ensures local execution.
# N_ensemble_configurations controls the number of ensemble passes (default is 32).
classifier = TabPFNClassifier(
    device="auto",
)

# 4. Train and Evaluate
# --------------------------------------------------
# TabPFN fits instantly as it performs in-context learning
classifier.fit(X_train, y_train)

# Make local predictions
y_pred = classifier.predict(X_test)
y_probs = classifier.predict_proba(X_test)

# 5. Output Results
# --------------------------------------------------
print(f"Local TabPFN Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))