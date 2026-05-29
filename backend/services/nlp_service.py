import os
import pickle
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import sys
import warnings
warnings.filterwarnings("ignore")

# Import clean_tweet from scripts/predict.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.predict import clean_tweet

class NLPService:
    """
    Loads and manages the NLP models for sentiment prediction.
    Primary model is BERT, fallback is SVM.
    """

    def __init__(self, config):
        self.config = config
        self.svm = None
        self.lr_model = None
        self.vectorizer = None
        self.encoder = None
        self.bert_model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = None  # "bert" or "svm"
        self._load_models()

    def _load_models(self):
        """Load all available models from the model/ folder."""

        # Load SVM, LR, vectorizer and encoder
        try:
            svm_path     = self.config.get('SVM_MODEL_PATH')
            tfidf_path   = self.config.get('TFIDF_PATH')
            encoder_path = self.config.get('ENCODER_PATH')

            with open(svm_path, "rb") as f:
                self.svm = pickle.load(f)
            with open(tfidf_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(encoder_path, "rb") as f:
                self.encoder = pickle.load(f)
            print("[OK] SVM model loaded")
        except FileNotFoundError as e:
            print(f"[ERROR] Could not load SVM model: {e}")
            raise

        lr_path = self.config.get('LR_MODEL_PATH')
        if lr_path and os.path.exists(lr_path):
            try:
                with open(lr_path, "rb") as f:
                    self.lr_model = pickle.load(f)
                print("[OK] Logistic Regression model loaded")
            except Exception as e:
                print(f"[WARN] Could not load LR model: {e}")

        # Try to load BERT
        bert_path = self.config.get('BERT_MODEL_PATH')
        if os.path.exists(bert_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
                self.bert_model.to(self.device)
                self.bert_model.eval()
                self.mode = "bert"
                print("[OK] BERT model loaded (NaijaSenti)")
            except Exception as e:
                print(f"[WARN] BERT model found but could not load: {e}")
                print("Falling back to SVM.")
                self.mode = "svm"
        else:
            self.mode = "svm"
            print("[INFO] No BERT model found -- using SVM")

        print(f"[INFO] Active model: {self.mode.upper()}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Labels: {self.encoder.classes_.tolist()}")

    def predict_one(self, raw_tweet):
        """
        Predict sentiment for a single tweet.
        Returns dict: { tweet, cleaned, label, confidence, model_used }
        """
        cleaned = clean_tweet(raw_tweet)

        if self.mode == "bert":
            label, confidence = self._predict_bert(cleaned)
        else:
            label, confidence = self._predict_svm(cleaned)

        return {
            "tweet": raw_tweet,
            "cleaned": cleaned,
            "label": label,
            "confidence": confidence,
            "model_used": self.mode.upper(),
        }

    def predict_batch(self, tweets):
        """
        Predict sentiment for a list of tweets.
        Returns a list of result dicts.
        Process in batches of 16 for memory efficiency.
        """
        results = []
        batch_size = 16
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i+batch_size]
            if self.mode == "bert":
                batch_results = self._predict_bert_batch([clean_tweet(t) for t in batch])
                for j, (label, conf) in enumerate(batch_results):
                    results.append({
                        "tweet": batch[j],
                        "cleaned": clean_tweet(batch[j]),
                        "label": label,
                        "confidence": conf,
                        "model_used": "BERT",
                    })
            else:
                for tweet in batch:
                    result = self.predict_one(tweet)
                    results.append(result)
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
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.bert_model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            conf = round(float(probs[pred]) * 100, 1)

        label = self.encoder.classes_[pred]
        return label, conf

    def _predict_bert_batch(self, cleaned_texts):
        """Run BERT inference on a batch of cleaned texts."""
        enc = self.tokenizer(
            cleaned_texts,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.bert_model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1) * 100

        labels = [self.encoder.classes_[pred] for pred in preds]
        return list(zip(labels, confs.round(1)))

    def _adjust_prediction(self, proba_array, threshold=0.25):
        """
        If Positive probability is above threshold but model predicted Neutral,
        override to Positive. This corrects for class imbalance bias.
        Classes order is Negative, Neutral, Positive based on label_encoder.
        """
        classes = self.encoder.classes_.tolist()
        pos_idx = classes.index('Positive')
        neu_idx = classes.index('Neutral')

        pred_idx = int(np.argmax(proba_array))
        pred_label = classes[pred_idx]

        # If predicted Neutral but Positive prob exceeds threshold, switch to Positive
        if pred_label == 'Neutral' and proba_array[pos_idx] >= threshold:
            pred_idx = pos_idx
            pred_label = 'Positive'

        confidence = round(float(proba_array[pred_idx]) * 100, 1)
        return pred_label, confidence

    def _predict_svm(self, cleaned_text):
        X = self.vectorizer.transform([cleaned_text])
        proba = self.svm.predict_proba(X)[0]
        label, confidence = self._adjust_prediction(proba)
        return label, confidence

    def _predict_lr(self, cleaned_text):
        """Run Logistic Regression inference on cleaned text."""
        X = self.vectorizer.transform([cleaned_text])
        proba = self.lr_model.predict_proba(X)[0]
        label, confidence = self._adjust_prediction(proba)
        return label, confidence

    def predict_with_model(self, raw_tweet, model_key):
        """
        Predict using a specific model.
        model_key: 'bert' | 'svm' | 'lr'
        """
        cleaned = clean_tweet(raw_tweet)

        if model_key == 'bert':
            if self.bert_model is None:
                raise ValueError("BERT model is not available")
            label, confidence = self._predict_bert(cleaned)
            model_name = "NaijaSenti (XLM-RoBERTa)"
        elif model_key == 'svm':
            label, confidence = self._predict_svm(cleaned)
            model_name = "SVM"
        elif model_key == 'lr':
            if self.lr_model is None:
                raise ValueError("Logistic Regression model is not available")
            label, confidence = self._predict_lr(cleaned)
            model_name = "Logistic Regression"
        else:
            raise ValueError(f"Unknown model key: {model_key}")

        return {
            "tweet": raw_tweet,
            "cleaned": cleaned,
            "label": label,
            "confidence": confidence,
            "model_used": model_name,
        }

    def predict_batch_with_model(self, raw_tweets, model_key):
        """Batch-classify using a specific model. Uses vectorized BERT for efficiency."""
        cleaned = [clean_tweet(t) for t in raw_tweets]

        if model_key == 'bert':
            if self.bert_model is None:
                raise ValueError("BERT model not available")
            batch_preds = self._predict_bert_batch(cleaned)
            return [
                {"tweet": raw_tweets[i], "cleaned": cleaned[i],
                 "label": lbl, "confidence": conf,
                 "model_used": "NaijaSenti (XLM-RoBERTa)"}
                for i, (lbl, conf) in enumerate(batch_preds)
            ]

        results = []
        for i, c in enumerate(cleaned):
            if model_key == 'svm':
                lbl, conf = self._predict_svm(c)
                mname = "SVM"
            elif model_key == 'lr':
                if self.lr_model is None:
                    raise ValueError("Logistic Regression model not available")
                lbl, conf = self._predict_lr(c)
                mname = "Logistic Regression"
            else:
                raise ValueError(f"Unknown model key: {model_key}")
            results.append({"tweet": raw_tweets[i], "cleaned": c,
                            "label": lbl, "confidence": conf, "model_used": mname})
        return results

    def predict_batch_ensemble(self, raw_tweets):
        """Ensemble-classify a batch using majority voting across all models."""
        cleaned = [clean_tweet(t) for t in raw_tweets]

        bert_preds = self._predict_bert_batch(cleaned) if self.bert_model else None
        svm_preds  = [self._predict_svm(c) for c in cleaned]
        lr_preds   = [self._predict_lr(c) for c in cleaned] if self.lr_model else None

        results = []
        for i, raw in enumerate(raw_tweets):
            individual = []
            if bert_preds:
                individual.append({"model": "NaijaSenti (XLM-RoBERTa)",
                                   "label": bert_preds[i][0], "confidence": bert_preds[i][1]})
            individual.append({"model": "SVM",
                               "label": svm_preds[i][0], "confidence": svm_preds[i][1]})
            if lr_preds:
                individual.append({"model": "Logistic Regression",
                                   "label": lr_preds[i][0], "confidence": lr_preds[i][1]})

            votes = Counter(r["label"] for r in individual)
            final_label, vote_count = votes.most_common(1)[0]
            total = len(individual)
            agreeing = [r["confidence"] for r in individual if r["label"] == final_label]
            avg_conf = round(sum(agreeing) / len(agreeing), 1)

            results.append({"tweet": raw, "cleaned": cleaned[i],
                            "ensemble": True, "individual": individual,
                            "label": final_label, "confidence": avg_conf,
                            "votes": vote_count, "total": total, "model_used": "Ensemble"})
        return results

    def predict_ensemble(self, raw_tweet):
        """
        Run all available models and combine via majority voting.
        Returns individual predictions plus the final ensemble decision.
        """
        cleaned = clean_tweet(raw_tweet)
        individual = []

        if self.bert_model is not None:
            label, conf = self._predict_bert(cleaned)
            individual.append({"model": "NaijaSenti (XLM-RoBERTa)", "label": label, "confidence": conf})

        label, conf = self._predict_svm(cleaned)
        individual.append({"model": "SVM", "label": label, "confidence": conf})

        if self.lr_model is not None:
            label, conf = self._predict_lr(cleaned)
            individual.append({"model": "Logistic Regression", "label": label, "confidence": conf})

        votes = Counter(r["label"] for r in individual)
        final_label, vote_count = votes.most_common(1)[0]
        total = len(individual)

        agreeing_confs = [r["confidence"] for r in individual if r["label"] == final_label]
        avg_confidence = round(sum(agreeing_confs) / len(agreeing_confs), 1)

        return {
            "tweet": raw_tweet,
            "cleaned": cleaned,
            "ensemble": True,
            "individual": individual,
            "label": final_label,
            "confidence": avg_confidence,
            "votes": vote_count,
            "total": total,
        }