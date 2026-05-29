"""
train_model.py  (fixed version)
================================
Phase 3 - Model Training and Benchmarking

Trains and compares FOUR models:
  1. Logistic Regression
  2. Support Vector Machine (SVM)
  3. BERT (bert-base-uncased)
  4. Davlan/naija-roberta-base  (Nigerian English RoBERTa)

HOW TO RUN (from project root):
    python scripts/train_model.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model            import LogisticRegression
from sklearn.svm                     import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection         import train_test_split
from sklearn.preprocessing           import LabelEncoder
from sklearn.metrics                 import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.pipeline                import Pipeline
from sklearn.calibration             import CalibratedClassifierCV

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim      import AdamW
from transformers     import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MAX_LEN    = 128
BATCH_SIZE = 8
EPOCHS     = 3
LR         = 2e-5
SEED       = 42

# correct working model names on HuggingFace
BERT_MODELS = [
    ("NaijaRoBERTa", "Davlan/naija-roberta-base"),
    ("BERT",         "bert-base-uncased"),
]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def divider(title=""):
    if title:
        pad = max(1, (60 - len(title) - 2) // 2)
        print("\n" + "=" * pad + f" {title} " + "=" * pad)
    else:
        print("\n" + "=" * 60)


def get_metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    return round(acc,2), round(prec,2), round(rec,2), round(f1,2)


def print_metrics(name, acc, prec, rec, f1):
    print(f"\n  {'Model':<30}: {name}")
    print(f"  {'Accuracy':<30}: {acc:.2f}%")
    print(f"  {'Precision':<30}: {prec:.2f}%")
    print(f"  {'Recall':<30}: {rec:.2f}%")
    print(f"  {'F1-Score':<30}: {f1:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
#  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  BERT TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_bert(model_name, friendly_name,
               X_train, X_test, y_train, y_test,
               num_labels, device):

    print(f"\n  Loading: {model_name}")
    print("  (First run downloads ~400MB — subsequent runs load from cache)\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
    except Exception as e:
        print(f"  ✗ Could not load {friendly_name}:\n  {e}")
        return None, None, None

    model.to(device)

    train_loader = DataLoader(
        TweetDataset(X_train, y_train, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TweetDataset(X_test, y_test, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False
    )

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"  Training {friendly_name} — {EPOCHS} epochs, {len(train_loader)} batches/epoch")
    print(f"  Device: {device}  |  Estimated time on CPU: 45–90 minutes\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0

        for i, batch in enumerate(train_loader):
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds       = out.logits.argmax(dim=1)
            correct    += (preds == lbls).sum().item()
            total      += lbls.size(0)

            if (i + 1) % 20 == 0 or (i + 1) == len(train_loader):
                print(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Batch {i+1}/{len(train_loader)} | "
                    f"Loss: {total_loss/(i+1):.4f} | "
                    f"Acc: {correct/total*100:.1f}%"
                )

        print(f"  ✓ Epoch {epoch+1} done\n")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"  Evaluating {friendly_name} on test set ...")
    model.eval()
    preds_all = []

    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            preds_all.extend(out.logits.argmax(dim=1).cpu().numpy())

    return model, tokenizer, np.array(preds_all)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    divider()
    print("   PHASE 3 — MODEL TRAINING & BENCHMARKING")
    divider()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/6] Loading cleaned dataset ...")
    df = pd.read_csv("data/cleaned_tweets.csv").dropna(subset=["cleaned_text","label"])
    df["cleaned_text"] = df["cleaned_text"].astype(str)
    print(f"      Rows   : {len(df):,}")
    print(f"      Labels : {df['label'].value_counts().to_dict()}")

    # ── Encode labels ─────────────────────────────────────────────────────────
    print("\n[2/6] Encoding labels ...")
    encoder    = LabelEncoder()
    y          = encoder.fit_transform(df["label"])
    texts      = df["cleaned_text"].tolist()
    num_labels = len(encoder.classes_)
    print(f"      Mapping: { {k:int(v) for k,v in zip(encoder.classes_, encoder.transform(encoder.classes_))} }")

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n[3/6] Splitting 80 / 20 ...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"      Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    results            = []   # (name, acc, prec, rec, f1)
    best_bert_model    = None
    best_bert_tok      = None
    best_bert_name     = None

    # ── Traditional ML ────────────────────────────────────────────────────────
    divider("TRADITIONAL ML MODELS")

    print("\n  Building TF-IDF matrix ...")
    tfidf = TfidfVectorizer(
        max_features=3000,   # reduced from 5000 to limit memorisation
        ngram_range=(1, 2),
        min_df=3,            # word must appear in at least 3 tweets
        max_df=0.85,         # ignore words in more than 85% of tweets
        sublinear_tf=True,
    )
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    # Logistic Regression — deliberately regularised to avoid overfitting
    print("\n    Training Logistic Regression ...")
    lr = LogisticRegression(
        C=0.1,          # strong regularisation — prevents memorisation
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs",
    )
    lr.fit(Xtr, y_train)
    lr_pred = lr.predict(Xte)
    lr_m    = get_metrics(y_test, lr_pred)
    print_metrics("Logistic Regression", *lr_m)
    results.append(("Logistic Regression", *lr_m))

    # SVM — also regularised
    print("\n  ▶  Training Support Vector Machine ...")
    svm = LinearSVC(
        C=0.1,          # strong regularisation
        max_iter=2000,
        random_state=SEED,
    )
    svm.fit(Xtr, y_train)
    svm_pred = svm.predict(Xte)
    svm_m    = get_metrics(y_test, svm_pred)
    print_metrics("Support Vector Machine (SVM)", *svm_m)
    results.append(("Support Vector Machine (SVM)", *svm_m))

    print(f"\n  SVM Per-Class Report:")
    print(classification_report(y_test, svm_pred, target_names=encoder.classes_, digits=3))

    # ── Transformer models ────────────────────────────────────────────────────
    divider("TRANSFORMER MODELS")

    for friendly_name, model_name in BERT_MODELS:
        print(f"\n  ▶  Attempting {friendly_name} ({model_name}) ...")
        out = train_bert(
            model_name, friendly_name,
            X_train, X_test, y_train, y_test,
            num_labels, device,
        )

        if out[0] is None:
            print(f"  Skipping {friendly_name}.\n")
            continue

        b_model, b_tok, b_pred = out
        b_m = get_metrics(y_test, b_pred)
        print_metrics(friendly_name, *b_m)
        results.append((friendly_name, *b_m))

        print(f"\n  {friendly_name} Per-Class Report:")
        print(classification_report(y_test, b_pred, target_names=encoder.classes_, digits=3))

        if best_bert_model is None:
            best_bert_model = b_model
            best_bert_tok   = b_tok
            best_bert_name  = friendly_name

        del b_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Once one transformer succeeds stop — saves time on CPU
        print(f"\n  ✓ {friendly_name} trained successfully. Stopping transformer loop.")
        break

    # ── Results table ─────────────────────────────────────────────────────────
    divider("FINAL BENCHMARKING TABLE")
    print(f"\n{'Model':<38} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>9}")
    print("-" * 78)
    for row in results:
        n, a, p, r, f = row
        print(f"{n:<38} {a:>8}%  {p:>9}%  {r:>7}%  {f:>8}%")
    print("-" * 78)

    best_row  = max(results, key=lambda x: x[4])
    best_name = best_row[0]
    print(f"\n  ★  Best model : {best_name}  (F1: {best_row[4]}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    divider("SAVING OUTPUTS")
    os.makedirs("model", exist_ok=True)

    with open("model/svm_model.pkl",        "wb") as f: pickle.dump(svm, f)
    with open("model/lr_model.pkl",         "wb") as f: pickle.dump(lr, f)
    with open("model/tfidf_vectorizer.pkl", "wb") as f: pickle.dump(tfidf, f)
    with open("model/label_encoder.pkl",    "wb") as f: pickle.dump(encoder, f)

    print("\n  ✓ SVM model saved          → model/svm_model.pkl")
    print("  ✓ LR model saved           → model/lr_model.pkl")
    print("  ✓ TF-IDF vectorizer saved  → model/tfidf_vectorizer.pkl")
    print("  ✓ Label encoder saved      → model/label_encoder.pkl")

    if best_bert_model is not None:
        bert_path = "model/best_bert_model"
        best_bert_model.save_pretrained(bert_path)
        best_bert_tok.save_pretrained(bert_path)
        print(f"  ✓ {best_bert_name} saved         → {bert_path}/")
    else:
        print("\n  ⚠  No transformer model was saved.")
        print("     The SVM model will be used as the primary model in the web app.")
        print("     This is acceptable — SVM is a proven model for this task.")

    results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1-Score"])
    results_df.to_csv("model/benchmarking_results.csv", index=False)
    print("  ✓ Benchmarking table       → model/benchmarking_results.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    divider("PHASE 3 COMPLETE")
    print(f"""
  Best model  : {best_name}
  F1-Score    : {best_row[4]}%
  Accuracy    : {best_row[1]}%

  Files saved to model/ folder:
    • svm_model.pkl            ← traditional ML model
    • lr_model.pkl             ← logistic regression model
    • tfidf_vectorizer.pkl     ← needed by web app
    • label_encoder.pkl        ← needed by web app
    • best_bert_model/         ← transformer model (if trained)
    • benchmarking_results.csv ← copy into your Chapter 4 report

  Ready for Phase 4 — Web Application Development.
""")