
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
 
 
if __name__ == "__main__":
 
    print("=" * 60)
    print("PHASE 2 - FEATURE EXTRACTION (TF-IDF)")
    print("=" * 60)
 
    # ── Load cleaned data ─────────────────────────────────────────────────────
    clean_path = "data/cleaned_tweets.csv"
    print(f"\n[1/5] Loading cleaned dataset from: {clean_path}")
 
    df = pd.read_csv(clean_path)
    print(f"      Rows loaded: {len(df):,}")
 
    # ── Separate text and labels ──────────────────────────────────────────────
    X_text = df["cleaned_text"].astype(str).tolist()
    y_raw  = df["label"].tolist()
 
    # ── Encode labels to numbers ──────────────────────────────────────────────
    # Negative → 0,  Neutral → 1,  Positive → 2  (alphabetical order)
    print("\n[2/5] Encoding labels to numbers...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"      Label mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
    # e.g.  {'Negative': 0, 'Neutral': 1, 'Positive': 2}
 
    # ── Build TF-IDF matrix ───────────────────────────────────────────────────
    # max_features=5000  → keep only the 5,000 most important words
    # ngram_range=(1,2)  → use single words AND two-word phrases (bigrams)
    #                       e.g. "food price" as one feature, not just "food" + "price"
    # min_df=2           → ignore words that appear in fewer than 2 tweets
    print("\n[3/5] Building TF-IDF feature matrix...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,    # apply log normalization to term frequencies
    )
    X = vectorizer.fit_transform(X_text)
 
    print(f"      Feature matrix shape: {X.shape}")
    print(f"      Meaning: {X.shape[0]} tweets  x  {X.shape[1]} features (words/phrases)")
 
    # Show top 20 most important features
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n      Top 20 TF-IDF features (most informative words/phrases):")
    import numpy as np
    mean_tfidf = X.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-20:][::-1]
    print("      " + ", ".join([feature_names[i] for i in top_indices]))
 
    # ── Save everything ───────────────────────────────────────────────────────
    print("\n[4/5] Saving feature matrix and vectorizer...")
 
    features_out   = "data/features_tfidf.pkl"
    vectorizer_out = "data/tfidf_vectorizer.pkl"
 
    with open(features_out, "wb") as f:
        pickle.dump({"X": X, "y": y, "labels": encoder.classes_.tolist()}, f)
 
    with open(vectorizer_out, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "encoder": encoder}, f)
 
    print(f"      Features saved to  : {features_out}")
    print(f"      Vectorizer saved to: {vectorizer_out}")
 
    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n[5/5] Summary:")
    print(f"      Total tweets     : {len(df):,}")
    print(f"      Features (words) : {X.shape[1]:,}")
    print(f"      Labels           : {encoder.classes_.tolist()}")
    print(f"      Label counts     : {dict(zip(encoder.classes_, [sum(y == i) for i in range(len(encoder.classes_))]))}")
    print("\n✓ Feature extraction complete. You are ready for Phase 3 - Model Training.\n")
 