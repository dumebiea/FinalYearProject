"""
retrain_svm_prob.py
===================
Retrains the SVM model wrapped in CalibratedClassifierCV so it can
produce real probability estimates via predict_proba().

Uses the EXISTING tfidf_vectorizer.pkl and label_encoder.pkl so the
rest of the pipeline does not change.

Run from the project root:
    python scripts/retrain_svm_prob.py
"""

import os
import pickle
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'cleaned_tweets.csv')
TFIDF_PATH = os.path.join(BASE_DIR, 'model', 'tfidf_vectorizer.pkl')
ENC_PATH   = os.path.join(BASE_DIR, 'model', 'label_encoder.pkl')
SVM_PATH   = os.path.join(BASE_DIR, 'model', 'svm_model.pkl')

print("Loading existing TF-IDF vectorizer and label encoder ...")
with open(TFIDF_PATH, 'rb') as f:
    tfidf = pickle.load(f)
with open(ENC_PATH, 'rb') as f:
    encoder = pickle.load(f)

print("Loading dataset ...")
df = pd.read_csv(DATA_PATH).dropna(subset=['cleaned_text', 'label'])
df['cleaned_text'] = df['cleaned_text'].astype(str)

# Keep only rows whose labels the encoder already knows
known = set(encoder.classes_)
df = df[df['label'].isin(known)].reset_index(drop=True)
print(f"  Rows after label filter: {len(df):,}")

X = tfidf.transform(df['cleaned_text'].tolist())
y = encoder.transform(df['label'].tolist())

print("Training CalibratedClassifierCV(LinearSVC) ...")
base = LinearSVC(C=0.1, max_iter=2000, random_state=42)
calibrated = CalibratedClassifierCV(base, cv=5, method='sigmoid')
calibrated.fit(X, y)

print(f"  predict_proba available: {hasattr(calibrated, 'predict_proba')}")

# Quick sanity check
sample = tfidf.transform(["food is too expensive in Lagos"])
proba  = calibrated.predict_proba(sample)[0]
pred   = calibrated.predict(sample)[0]
label  = encoder.classes_[pred]
conf   = round(float(max(proba)) * 100, 1)
print(f"  Sanity check — label: {label}, confidence: {conf}%")

print(f"Saving to {SVM_PATH} ...")
with open(SVM_PATH, 'wb') as f:
    pickle.dump(calibrated, f)

print("Done. SVM model now supports real predict_proba() confidence scores.")
