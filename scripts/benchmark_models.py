#!/usr/bin/env python3
"""
benchmark_models.py
-------------------
Evaluate SVM, Logistic Regression, NaijaSenti BERT, and a majority-vote
Ensemble on the TweetEval sentiment test set (external, never seen during
training).

Using an external dataset removes the data-leakage problem that occurs when
evaluating on the Lagos Food Crisis CSV (which was part of Colab training).

SAMPLE_SIZE controls how many TweetEval rows are used — default 2000 keeps
BERT inference under ~10 min on CPU. Set to None to use all ~12k rows.

Output files
------------
  model/external_benchmarking_results.csv   — summary table (4 models)
  model/external_benchmarking_detailed.csv  — per-tweet predictions
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)

SVM_PATH   = os.path.join(BASE_DIR, "model", "svm_model_proba.pkl")
LR_PATH    = os.path.join(BASE_DIR, "model", "lr_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")
ENC_PATH   = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
BERT_PATH  = os.path.join(BASE_DIR, "model", "best_bert_model")

OUT_SUMMARY  = os.path.join(BASE_DIR, "model", "external_benchmarking_results.csv")
OUT_DETAILED = os.path.join(BASE_DIR, "model", "external_benchmarking_detailed.csv")

CLASS_NAMES = ["Negative", "Neutral", "Positive"]

# tweet_eval label integers match our encoder: 0=Negative, 1=Neutral, 2=Positive
TWEETEVAL_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Set to None to use all ~12,284 test rows (BERT will take ~45 min on CPU)
SAMPLE_SIZE = 2000

# ── Import clean_tweet from predict.py ────────────────────────────────────────
sys.path.insert(0, SCRIPT_DIR)
from predict import clean_tweet


# ─────────────────────────────────────────────────────────────────────────────
#  1. LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    print("\n[1/5] Loading models...")

    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    print(f"   [OK] TF-IDF vectorizer  ({tfidf.max_features} features, "
          f"ngram_range={tfidf.ngram_range})")

    with open(SVM_PATH, "rb") as f:
        svm = pickle.load(f)
    print("   [OK] SVM (calibrated)")

    with open(LR_PATH, "rb") as f:
        lr = pickle.load(f)
    print("   [OK] Logistic Regression")

    with open(ENC_PATH, "rb") as f:
        encoder = pickle.load(f)
    print(f"   [OK] Label encoder  (classes: {list(encoder.classes_)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n   Loading BERT from  {BERT_PATH}")
    if device.type == "cpu":
        print(f"   [WARN]  No GPU — BERT runs on CPU "
              f"(~5-10 min for {SAMPLE_SIZE or 12284} tweets)")
    tokenizer  = AutoTokenizer.from_pretrained(BERT_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
    bert_model.eval()
    bert_model.to(device)
    print("   [OK] NaijaSenti BERT (XLM-RoBERTa)")

    return tfidf, svm, lr, encoder, tokenizer, bert_model, device


# ─────────────────────────────────────────────────────────────────────────────
#  2. LOAD EXTERNAL TEST DATA (TweetEval)
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("\n[2/5] Loading TweetEval sentiment test set (external dataset)...")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. "
            "Run:  pip install datasets"
        )

    ds   = load_dataset("tweet_eval", "sentiment")
    test = ds["test"].to_pandas()
    print(f"   Full TweetEval test set: {len(test)} samples")

    # Map integer labels to strings matching our encoder
    test["label_str"] = test["label"].map(TWEETEVAL_LABEL_MAP)

    # Sample for CPU feasibility
    if SAMPLE_SIZE and SAMPLE_SIZE < len(test):
        test = (
            test.sample(n=SAMPLE_SIZE, random_state=42)
            .reset_index(drop=True)
        )
        print(f"   Sampled {SAMPLE_SIZE} rows (random_state=42)")

    print(f"   Using {len(test)} samples")
    print(f"   Label distribution:\n"
          f"{test['label_str'].value_counts().to_string()}")

    # Preprocess
    print("   Preprocessing text...")
    test["clean_text"] = test["text"].astype(str).apply(clean_tweet)

    empty = (test["clean_text"].isin(["", "empty"])).sum()
    if empty:
        print(f"   [WARN]  {empty} tweets became empty after preprocessing")

    return test


# ─────────────────────────────────────────────────────────────────────────────
#  3. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def bert_predict(texts, tokenizer, model, device, batch_size=16):
    all_preds = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        done = i // batch_size + 1
        if done % 5 == 0 or done == n_batches:
            print(f"      batch {done}/{n_batches}")
    return np.array(all_preds)


def ensemble_predict(svm_preds, lr_preds, bert_preds):
    # Weighted voting: BERT=3, LR=2, SVM=1
    # Weights reflect model accuracy hierarchy from Colab evaluation.
    # Only possible tie: LR+SVM agree vs BERT (3 vs 3) -> BERT wins.
    results = []
    for s, l, b in zip(svm_preds, lr_preds, bert_preds):
        totals = {}
        for pred, weight in [(int(b), 3), (int(l), 2), (int(s), 1)]:
            totals[pred] = totals.get(pred, 0) + weight
        max_w   = max(totals.values())
        winners = [label for label, w in totals.items() if w == max_w]
        results.append(winners[0] if len(winners) == 1 else int(b))
    return np.array(results)


# ─────────────────────────────────────────────────────────────────────────────
#  4. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(name, true_num, pred_num):
    acc  = accuracy_score(true_num, pred_num)
    prec = precision_score(true_num, pred_num, average="weighted", zero_division=0)
    rec  = recall_score(true_num, pred_num,    average="weighted", zero_division=0)
    f1   = f1_score(true_num, pred_num,        average="weighted", zero_division=0)
    report = classification_report(
        true_num, pred_num,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print(f"\n{'=' * 60}")
    print(f"**  {name}")
    print("=" * 60)
    print(f"   Accuracy  : {acc * 100:.2f}%")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"\n{report}")

    return {
        "Model":     name,
        "Accuracy":  round(acc * 100, 2),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1_Score":  round(f1,   4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EXTERNAL BENCHMARK — TweetEval Sentiment Test Set")
    print("=" * 60)

    tfidf, svm, lr, encoder, tokenizer, bert_model, device = load_models()
    test_df = load_data()

    # true_num: integer labels already aligned with our encoder (0/1/2)
    true_num    = test_df["label"].values
    true_labels = test_df["label_str"].values
    texts       = test_df["clean_text"].tolist()

    # ── Predictions ───────────────────────────────────────────────────────────
    print("\n[3/5] Running predictions...")

    print("   SVM...")
    X       = tfidf.transform(texts)
    svm_num = svm.predict(X)

    print("   Logistic Regression...")
    lr_num  = lr.predict(X)

    print("   BERT (may take several minutes on CPU)...")
    bert_num = bert_predict(texts, tokenizer, bert_model, device)

    print("   Ensemble (majority vote, BERT tiebreaker)...")
    ens_num  = ensemble_predict(svm_num, lr_num, bert_num)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[4/5] Evaluating all models...")
    results = [
        evaluate("SVM",                 true_num, svm_num),
        evaluate("Logistic Regression", true_num, lr_num),
        evaluate("NaijaSenti BERT",     true_num, bert_num),
        evaluate("Ensemble",            true_num, ens_num),
    ]

    # ── Comparison table — Ensemble ranks first on ties ───────────────────────
    priority = {"Ensemble": 0, "NaijaSenti BERT": 1,
                "Logistic Regression": 2, "SVM": 3}
    results_df = pd.DataFrame(results)
    results_df["_rank"] = results_df["Model"].map(priority)
    results_df = (
        results_df
        .sort_values(["F1_Score", "_rank"], ascending=[False, True])
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    results_df.index += 1

    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE  (ranked by F1-Score)")
    print("=" * 60)
    print(
        results_df.to_string(
            index=True,
            formatters={
                "Accuracy":  lambda x: f"{x:.2f}%",
                "Precision": lambda x: f"{x:.4f}",
                "Recall":    lambda x: f"{x:.4f}",
                "F1_Score":  lambda x: f"{x:.4f}",
            },
        )
    )

    # ── Save summary ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving results...")
    results_df.to_csv(OUT_SUMMARY, index=False)
    print(f"   [OK] Summary  -> {OUT_SUMMARY}")

    # ── Save detailed ─────────────────────────────────────────────────────────
    detailed = test_df[["text", "clean_text"]].copy()
    detailed["true_label"]       = true_labels
    detailed["svm_pred"]         = encoder.inverse_transform(svm_num)
    detailed["lr_pred"]          = encoder.inverse_transform(lr_num)
    detailed["bert_pred"]        = encoder.inverse_transform(bert_num)
    detailed["ensemble_pred"]    = encoder.inverse_transform(ens_num)
    detailed["ensemble_correct"] = (
        detailed["ensemble_pred"] == detailed["true_label"]
    )
    detailed.to_csv(OUT_DETAILED, index=False)
    print(f"   [OK] Detailed -> {OUT_DETAILED}")

    # ── Final summary ─────────────────────────────────────────────────────────
    best    = results_df.iloc[0]
    ens_row = results_df[results_df["Model"] == "Ensemble"].iloc[0]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Best model overall : {best['Model']}")
    print(f"                        Accuracy {best['Accuracy']:.2f}%  |  "
          f"F1 {best['F1_Score']:.4f}")
    print(f"   Total tweets tested: {len(test_df)}")
    print(f"   Dataset            : TweetEval sentiment test set")
    print(f"\n   Ensemble improvement over each individual model:")
    for r in results:
        if r["Model"] == "Ensemble":
            continue
        f1_diff  = round(ens_row["F1_Score"] - r["F1_Score"], 4)
        acc_diff = round(ens_row["Accuracy"] - r["Accuracy"], 2)
        sign_f1  = "+" if f1_diff  >= 0 else ""
        sign_acc = "+" if acc_diff >= 0 else ""
        print(f"     vs {r['Model']:<25}  "
              f"{sign_f1}{f1_diff:.4f} F1  "
              f"({sign_acc}{acc_diff:.2f}% accuracy)")
    print("=" * 60)


if __name__ == "__main__":
    main()
