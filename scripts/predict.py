"""
predict.py
==========
Standalone tweet sentiment predictor.
Tests your trained model by letting you type tweets and see predictions.

TWO MODES:
  1. Interactive mode  — type tweets one by one in the terminal
  2. Batch mode        — pass a CSV file of tweets to classify all at once

HOW TO RUN:

  Interactive (type tweets manually):
    python scripts/predict.py

  Batch (classify a CSV file):
    python scripts/predict.py --file data/lagos_food_crisis_tweets.csv

  Single tweet from command line:
    python scripts/predict.py --tweet "Rice don too expensive for Lagos"
"""

import os
import re
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─────────────────────────────────────────────────────────────────────────────
#  PIDGIN MAP — same as preprocessing
# ─────────────────────────────────────────────────────────────────────────────
PIDGIN_MAP = {
    "dey":"is","don":"has","wan":"want","chop":"eat",
    "abeg":"please","wahala":"problem","pikin":"child",
    "pikini":"child","pikins":"children","fam":"family",
    "sef":"even","nkor":"what about","na":"is","wetin":"what",
    "dem":"them","dis":"this","dat":"that","oga":"boss",
    "naija":"nigeria","9ja":"nigeria","wey":"that",
    "make":"let","comot":"remove","fit":"can","una":"you",
    "pple":"people","ppl":"people","govt":"government",
    "govnt":"government","cos":"because","cus":"because",
    "bcuz":"because","mkt":"market","fud":"food","fd":"food",
    "childrn":"children","childn":"children","rt":"",
    "no":"not","nor":"not","go":"will","o":"","oo":"","ooo":"",
}


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING — same pipeline as training
# ─────────────────────────────────────────────────────────────────────────────

def clean_tweet(text):
    """Clean a single tweet using the same pipeline as training."""
    # Remove RT prefix
    text = re.sub(r"^RT\s+@\w+:\s*", "", text, flags=re.IGNORECASE)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    # Keep words from hashtags
    text = re.sub(r"#(\w+)", r"\1", text)
    # Remove non-ASCII (emojis etc)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove non-letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Lowercase
    text = text.lower()
    # Apply Pidgin map
    words  = text.split()
    words  = [PIDGIN_MAP.get(w, w) for w in words]
    words  = [w for w in words if len(w) > 1]
    text   = " ".join(words)
    # Normalize whitespace
    text   = re.sub(r"\s+", " ", text).strip()
    return text if text else "empty"


# ─────────────────────────────────────────────────────────────────────────────
#  SENTIMENT PREDICTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SentimentPredictor:
    """
    Loads your trained models and predicts sentiment for any tweet.
    Uses BERT if available, falls back to SVM automatically.
    """

    # Emoji and colour for each sentiment
    SENTIMENT_DISPLAY = {
        "Negative": ("🔴", "NEGATIVE — Crisis Signal Detected"),
        "Neutral":  ("⚪", "NEUTRAL  — No Strong Signal"),
        "Positive": ("🟢", "POSITIVE — Reassuring Signal"),
    }

    def __init__(self):
        self.svm        = None
        self.vectorizer = None
        self.encoder    = None
        self.bert_model = None
        self.tokenizer  = None
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode       = None   # "bert" or "svm"
        self._load_models()

    def _load_models(self):
        """Load all available models from the model/ folder."""

        # Always load SVM and vectorizer as fallback
        try:
            with open("model/svm_model_proba.pkl",   "rb") as f:
                self.svm = pickle.load(f)
            with open("model/tfidf_vectorizer.pkl",  "rb") as f:
                self.vectorizer = pickle.load(f)
            with open("model/label_encoder.pkl",     "rb") as f:
                self.encoder = pickle.load(f)
            print("  ✓ SVM model loaded")
        except FileNotFoundError as e:
            print(f"  ✗ Could not load SVM model: {e}")
            print("    Please run train_model.py first.")
            sys.exit(1)

        # Try to load BERT — use it if available (more accurate)
        bert_path = "model/best_bert_model"
        if os.path.exists(bert_path):
            try:
                self.tokenizer  = AutoTokenizer.from_pretrained(bert_path)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
                self.bert_model.to(self.device)
                self.bert_model.eval()
                self.mode = "bert"
                print(f"  ✓ BERT model loaded (NaijaSenti)")
            except Exception as e:
                print(f"  ⚠  BERT model found but could not load: {e}")
                print("     Falling back to SVM.")
                self.mode = "svm"
        else:
            self.mode = "svm"
            print("  ℹ  No BERT model found — using SVM")

        print(f"  ℹ  Active model : {self.mode.upper()}")
        print(f"  ℹ  Device       : {self.device}")
        print(f"  ℹ  Labels       : {self.encoder.classes_.tolist()}")

    def predict_one(self, raw_tweet):
        """
        Predict sentiment for a single tweet.
        Returns dict with keys: tweet, cleaned, label, confidence, emoji
        """
        cleaned = clean_tweet(raw_tweet)

        if self.mode == "bert":
            label, confidence = self._predict_bert(cleaned)
        else:
            label, confidence = self._predict_svm(cleaned)

        emoji, display = self.SENTIMENT_DISPLAY.get(label, ("❓", label))

        return {
            "tweet":      raw_tweet,
            "cleaned":    cleaned,
            "label":      label,
            "display":    display,
            "emoji":      emoji,
            "confidence": confidence,
            "model_used": self.mode.upper(),
        }

    def predict_batch(self, tweets):
        """
        Predict sentiment for a list of tweets.
        Returns a list of result dicts.
        """
        results = []
        for i, tweet in enumerate(tweets):
            result = self.predict_one(tweet)
            results.append(result)
            # Show progress for large batches
            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{len(tweets)} tweets ...")
        return results

    def _predict_bert(self, cleaned_text):
        """Run BERT inference on cleaned text."""
        enc = self.tokenizer(
            cleaned_text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            out    = self.bert_model(input_ids=ids, attention_mask=mask)
            probs  = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            pred   = int(np.argmax(probs))
            conf   = round(float(probs[pred]) * 100, 1)

        label = self.encoder.classes_[pred]
        return label, conf

    def _predict_svm(self, cleaned_text):
        X = self.vectorizer.transform([cleaned_text])
        proba = self.svm.predict_proba(X)[0]
        pred = int(np.argmax(proba))
        label = self.encoder.classes_[pred]
        confidence = round(float(proba[pred]) * 100, 1)
        return label, confidence


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result):
    """Print a single prediction result clearly."""
    print("\n" + "─" * 60)
    print(f"  Tweet     : {result['tweet'][:80]}")
    print(f"  Cleaned   : {result['cleaned'][:80]}")
    print(f"  Result    : {result['emoji']}  {result['display']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  Model     : {result['model_used']}")
    print("─" * 60)


def print_summary(results):
    """Print a summary of batch prediction results."""
    labels  = [r["label"] for r in results]
    total   = len(labels)
    neg     = labels.count("Negative")
    neu     = labels.count("Neutral")
    pos     = labels.count("Positive")

    neg_pct = round(neg / total * 100, 1)
    neu_pct = round(neu / total * 100, 1)
    pos_pct = round(pos / total * 100, 1)

    print("\n" + "=" * 60)
    print("  BATCH PREDICTION SUMMARY")
    print("=" * 60)
    print(f"\n  Total tweets analysed : {total:,}")
    print(f"\n  🔴 Negative : {neg:>5,}  ({neg_pct}%)")
    print(f"  ⚪ Neutral  : {neu:>5,}  ({neu_pct}%)")
    print(f"  🟢 Positive : {pos:>5,}  ({pos_pct}%)")

    # Crisis alert threshold — if >50% tweets are negative
    if neg_pct > 50:
        print(f"\n  🚨 ALERT: {neg_pct}% negative sentiment detected!")
        print("     This exceeds the 50% crisis threshold.")
        print("     Potential food crisis signal identified.")
    elif neg_pct > 35:
        print(f"\n  ⚠️  WARNING: {neg_pct}% negative sentiment.")
        print("     Elevated food-related distress detected.")
    else:
        print(f"\n  ✅ Sentiment levels within normal range.")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE MODE — type tweets one by one
# ─────────────────────────────────────────────────────────────────────────────

def interactive_mode(predictor):
    """Let the user type tweets and see predictions in real time."""

    print("\n" + "=" * 60)
    print("  INTERACTIVE TWEET SENTIMENT ANALYSER")
    print("  Food Crisis Detection System — Lagos, Nigeria")
    print("=" * 60)
    print("\n  Type any tweet and press Enter to see the sentiment.")
    print("  Type 'quit' or press Ctrl+C to exit.")
    print("  Type 'batch' to switch to batch mode.\n")

    session_results = []

    try:
        while True:
            tweet = input("  Enter tweet: ").strip()

            if not tweet:
                print("  (empty input — please type something)")
                continue

            if tweet.lower() == "quit":
                break

            if tweet.lower() == "batch":
                print("  Switch to batch mode: run with --file yourfile.csv")
                break

            result = predictor.predict_one(tweet)
            print_result(result)
            session_results.append(result)

    except KeyboardInterrupt:
        print("\n\n  Exiting...")

    # Show session summary if more than one tweet was tested
    if len(session_results) > 1:
        print(f"\n  You tested {len(session_results)} tweets this session.")
        print_summary(session_results)

        # Offer to save results
        save = input("\n  Save session results to CSV? (y/n): ").strip().lower()
        if save == "y":
            df = pd.DataFrame(session_results)[["tweet","label","confidence","model_used"]]
            df.to_csv("data/interactive_predictions.csv", index=False)
            print("  ✓ Saved to data/interactive_predictions.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH MODE — classify a whole CSV file
# ─────────────────────────────────────────────────────────────────────────────

def batch_mode(predictor, filepath):
    """Classify all tweets in a CSV file."""

    print(f"\n  Loading tweets from: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"  ✗ File not found: {filepath}")
        sys.exit(1)

    # Detect which column has the tweets
    tweet_col = None
    for candidate in ["tweet_text", "tweet", "text", "cleaned_text", "content"]:
        if candidate in df.columns:
            tweet_col = candidate
            break

    if tweet_col is None:
        print(f"  ✗ Could not find tweet column.")
        print(f"    Available columns: {df.columns.tolist()}")
        print(f"    Please rename your tweet column to 'tweet_text'")
        sys.exit(1)

    tweets = df[tweet_col].astype(str).tolist()
    print(f"  Found {len(tweets):,} tweets in column '{tweet_col}'")
    print(f"  Running predictions ...\n")

    results = predictor.predict_batch(tweets)

    # Add predictions back to dataframe
    df["predicted_label"]      = [r["label"]      for r in results]
    df["confidence"]           = [r["confidence"] for r in results]
    df["model_used"]           = [r["model_used"] for r in results]

    # Save output
    out_path = filepath.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Predictions saved to: {out_path}")

    # Print summary
    print_summary(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Food Crisis Tweet Sentiment Predictor")
    parser.add_argument("--tweet", type=str, help="Single tweet to classify")
    parser.add_argument("--file",  type=str, help="CSV file of tweets to classify")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  FOOD CRISIS DETECTION SYSTEM")
    print("  Loading models ...")
    print("=" * 60)

    # Load models
    predictor = SentimentPredictor()

    if args.tweet:
        # Single tweet from command line argument
        result = predictor.predict_one(args.tweet)
        print_result(result)

    elif args.file:
        # Batch mode — classify a whole file
        batch_mode(predictor, args.file)

    else:
        # Default — interactive mode
        interactive_mode(predictor)