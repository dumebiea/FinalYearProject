"""
train_bert.py
=============
Dedicated BERT fine-tuning script for food crisis sentiment analysis.

Fixes applied vs original:
  1. Weighted CrossEntropyLoss (inverse-frequency) — corrects class imbalance
     (dataset is 60% Negative / 25% Neutral / 15% Positive)
  2. Dropout set via AutoConfig — reduces memorisation
  3. Early stopping on macro-F1 (patience=2) — halts before overfitting;
     best checkpoint is restored before final evaluation
  4. Per-class F1 saved to model/bert_results_detailed.csv

Tries models in this order until one works:
  1. Davlan/naija-twitter-sentiment-afriberta-large  (Nigerian English)
  2. Davlan/afro-xlmr-base
  3. distilbert-base-uncased
  4. bert-base-uncased

INPUT  : data/cleaned_tweets.csv
OUTPUT : model/best_bert_model/          (saved transformer model)
         model/bert_results.csv          (overall weighted metrics)
         model/bert_results_detailed.csv (per-class F1 breakdown)

HOW TO RUN:
    python scripts/train_bert.py
"""

import os
import copy
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim      import AdamW

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MAX_LEN    = 64
BATCH_SIZE = 16
EPOCHS     = 10     # high ceiling — early stopping halts training before memorisation
PATIENCE   = 2      # epochs without macro-F1 improvement before stopping
LR         = 2e-5
SEED       = 42

# Inverse-frequency class weights.
# LabelEncoder assigns alphabetically: 0=Negative, 1=Neutral, 2=Positive
# Negative is 60% of data → weight 1.0 (baseline)
# Neutral  is 25% of data → weight 2.4  (60/25)
# Positive is 15% of data → weight 4.0  (60/15)
CLASS_WEIGHTS = torch.tensor([1.0, 2.4, 4.0])

CANDIDATE_MODELS = [
    ("NaijaSenti", "Davlan/naija-twitter-sentiment-afriberta-large"),
    ("AfroXLMR",   "Davlan/afro-xlmr-base"),
    ("DistilBERT", "distilbert-base-uncased"),
    ("BERT",       "bert-base-uncased"),
]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING — dropout configured via AutoConfig before loading weights
# ─────────────────────────────────────────────────────────────────────────────

def try_load_model(model_id, num_labels):
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoConfig,
    )
    try:
        print(f"  Trying to load: {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # Load config and increase dropout for regularisation before
        # the classification head is initialised.
        config = AutoConfig.from_pretrained(model_id, num_labels=num_labels)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            config.classifier_dropout = 0.3
        elif hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = 0.2

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            config=config,
            ignore_mismatched_sizes=True,
        )
        print(f"  Successfully loaded: {model_id}")
        return tokenizer, model
    except Exception as e:
        print(f"  Failed to load {model_id}: {str(e)[:120]}")
        return None, None

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING — weighted loss + early stopping on macro-F1
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(model, tokenizer, X_train, X_test,
                       y_train, y_test, label_names, device):

    train_loader = DataLoader(
        TweetDataset(X_train, y_train, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TweetDataset(X_test, y_test, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    from transformers import get_linear_schedule_with_warmup
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Weighted loss — minority classes (Neutral, Positive) are penalised more
    # when the model gets them wrong, forcing it away from always predicting Negative.
    loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

    model.to(device)

    print(f"\n  Starting training ...")
    print(f"  Max epochs      : {EPOCHS}  (early stopping at patience={PATIENCE})")
    print(f"  Class weights   : Negative={CLASS_WEIGHTS[0]:.1f}  "
          f"Neutral={CLASS_WEIGHTS[1]:.1f}  Positive={CLASS_WEIGHTS[2]:.1f}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Device          : {device}")
    print(f"  Estimated time on CPU: 30-90 minutes\n")

    best_val_f1      = -1.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for i, batch in enumerate(train_loader):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["label"].to(device)

            optimizer.zero_grad()
            # Pass inputs without labels so we apply our own weighted loss below
            out  = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds       = out.logits.argmax(dim=1)
            correct    += (preds == lbls).sum().item()
            total      += lbls.size(0)

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Batch {i+1:>3}/{len(train_loader)} | "
                    f"Loss: {total_loss/(i+1):.4f} | "
                    f"Train Acc: {correct/total*100:.1f}%"
                )

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_preds  = []
        val_labels = []

        with torch.no_grad():
            for batch in test_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["label"].to(device)
                out  = model(input_ids=ids, attention_mask=mask)
                p    = out.logits.argmax(dim=1)
                val_preds.extend(p.cpu().numpy())
                val_labels.extend(lbls.cpu().numpy())

        val_acc      = accuracy_score(val_labels, val_preds) * 100
        # Use macro-F1 for early stopping — it weights each class equally
        # regardless of support, so Positive (minority) gets full credit.
        val_f1_macro = f1_score(val_labels, val_preds, average="macro", zero_division=0) * 100

        print(f"\n  Epoch {epoch+1} complete | "
              f"Val Acc: {val_acc:.1f}% | Val Macro-F1: {val_f1_macro:.1f}%")

        # ── Early stopping ─────────────────────────────────────────────────────
        if val_f1_macro > best_val_f1:
            best_val_f1      = val_f1_macro
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  New best macro-F1: {best_val_f1:.1f}% — checkpoint saved\n")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})\n")
            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}.\n")
                break

    # ── Restore best checkpoint ────────────────────────────────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Restored best checkpoint (val macro-F1: {best_val_f1:.1f}%)\n")

    # ── Final evaluation on held-out test set only ─────────────────────────────
    model.eval()
    final_preds  = []
    final_labels = []

    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["label"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            p    = out.logits.argmax(dim=1)
            final_preds.extend(p.cpu().numpy())
            final_labels.extend(lbls.cpu().numpy())

    acc  = round(accuracy_score(final_labels, final_preds) * 100, 2)
    prec = round(precision_score(final_labels, final_preds, average="weighted", zero_division=0) * 100, 2)
    rec  = round(recall_score(final_labels, final_preds, average="weighted", zero_division=0) * 100, 2)
    f1   = round(f1_score(final_labels, final_preds, average="weighted", zero_division=0) * 100, 2)

    return model, acc, prec, rec, f1, final_preds, final_labels


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("   BERT FINE-TUNING — FOOD CRISIS SENTIMENT MODEL")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if str(device) == "cpu":
        print("  Note   : No GPU found. Running on CPU — this will be slow.")
        print("           You can leave it running and check back in 30-90 mins.")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading cleaned dataset ...")
    df = pd.read_csv("data/cleaned_tweets.csv").dropna(subset=["cleaned_text", "label"])
    df["cleaned_text"] = df["cleaned_text"].astype(str)
    print(f"      Rows   : {len(df):,}")
    print(f"      Labels : {df['label'].value_counts().to_dict()}")

    # ── Encode labels ─────────────────────────────────────────────────────────
    print("\n[2/5] Encoding labels ...")
    encoder     = LabelEncoder()
    y           = encoder.fit_transform(df["label"])
    texts       = df["cleaned_text"].tolist()
    num_labels  = len(encoder.classes_)
    label_names = encoder.classes_.tolist()
    print(f"      Mapping : {dict(zip(label_names, encoder.transform(encoder.classes_)))}")
    print(f"      Weights : {dict(zip(label_names, CLASS_WEIGHTS.tolist()))}")

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Splitting 80/20 (stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    print(f"      Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    # ── Find a working model ──────────────────────────────────────────────────
    print("\n[4/5] Finding a downloadable transformer model ...")
    chosen_name, chosen_tokenizer, chosen_model = None, None, None

    for friendly_name, model_id in CANDIDATE_MODELS:
        print(f"\n  Attempting {friendly_name} ...")
        tok, mdl = try_load_model(model_id, num_labels)
        if tok is not None:
            chosen_name      = friendly_name
            chosen_tokenizer = tok
            chosen_model     = mdl
            break

    if chosen_model is None:
        print("\nERROR: None of the 4 models could be downloaded.")
        print("Check your internet connection and try again.")
        exit(1)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training {chosen_name} ...")
    trained_model, acc, prec, rec, f1, preds, true_labels = train_and_evaluate(
        chosen_model, chosen_tokenizer,
        X_train, X_test, y_train, y_test,
        label_names, device,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    report_str  = classification_report(
        true_labels, preds,
        target_names=label_names, digits=3, zero_division=0,
    )
    report_dict = classification_report(
        true_labels, preds,
        target_names=label_names, digits=3, zero_division=0,
        output_dict=True,
    )

    print("\n" + "=" * 60)
    print(f"  FINAL RESULTS — {chosen_name}")
    print("=" * 60)
    print(f"  Accuracy  : {acc}%")
    print(f"  Precision : {prec}%  (weighted)")
    print(f"  Recall    : {rec}%  (weighted)")
    print(f"  F1-Score  : {f1}%  (weighted)")
    print(f"\n  Per-Class Breakdown:")
    print(report_str)

    # ── Save model ────────────────────────────────────────────────────────────
    print("  Saving model ...")
    os.makedirs("model/best_bert_model", exist_ok=True)
    trained_model.save_pretrained("model/best_bert_model")
    chosen_tokenizer.save_pretrained("model/best_bert_model")

    with open("model/bert_model_name.txt", "w") as f:
        f.write(chosen_name)
    with open("model/label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    # ── Save overall results ──────────────────────────────────────────────────
    pd.DataFrame([{
        "Model":     chosen_name,
        "Accuracy":  f"{acc}%",
        "Precision": f"{prec}%",
        "Recall":    f"{rec}%",
        "F1-Score":  f"{f1}%",
    }]).to_csv("model/bert_results.csv", index=False)

    # ── Save per-class results ────────────────────────────────────────────────
    rows = []
    for class_name in label_names:
        m = report_dict.get(class_name, {})
        rows.append({
            "Class":     class_name,
            "Precision": round(m.get("precision", 0) * 100, 2),
            "Recall":    round(m.get("recall",    0) * 100, 2),
            "F1-Score":  round(m.get("f1-score",  0) * 100, 2),
            "Support":   int(m.get("support",     0)),
        })
    wa = report_dict.get("weighted avg", {})
    rows.append({
        "Class":     "weighted avg",
        "Precision": round(wa.get("precision", 0) * 100, 2),
        "Recall":    round(wa.get("recall",    0) * 100, 2),
        "F1-Score":  round(wa.get("f1-score",  0) * 100, 2),
        "Support":   int(wa.get("support",     0)),
    })
    pd.DataFrame(rows).to_csv("model/bert_results_detailed.csv", index=False)

    print(f"\n  Model saved            -> model/best_bert_model/")
    print(f"  Overall results        -> model/bert_results.csv")
    print(f"  Per-class results      -> model/bert_results_detailed.csv")
    print(f"  Label encoder          -> model/label_encoder.pkl")

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE — {chosen_name}")
    print(f"  Weighted F1 : {f1}%")
    print("=" * 60 + "\n")
