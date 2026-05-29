
#Phase 2 - Part A: Clean raw tweets for the food crisis sentiment model.
 
import re
import pandas as pd
import nltk
 
# ── Download NLTK data (safe - skips if already present or no internet) ──────
for pkg in ["stopwords", "wordnet", "punkt"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
 
# ── Try NLTK stopwords, fall back to built-in list ───────────────────────────
try:
    from nltk.corpus import stopwords as nltk_sw
    _base_stopwords = set(nltk_sw.words("english"))
except Exception:
    # Built-in fallback — covers all the important English stop words
    _base_stopwords = {
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "by","from","is","was","are","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may","might",
        "shall","can","need","dare","ought","used","not","no","nor","so","yet",
        "both","either","neither","each","every","all","any","few","more","most",
        "other","some","such","than","too","very","just","as","if","then","that",
        "this","these","those","it","its","i","me","my","we","our","you","your",
        "he","him","his","she","her","they","them","their","what","which","who",
        "when","where","why","how","up","down","out","about","into","through",
        "during","before","after","above","below","between","among","over",
        "under","again","further","once","here","there","now","only","own","same",
        "also","however","although","because","since","while","though","until",
    }
 
# ── Try NLTK lemmatizer, fall back to simple suffix stripping ─────────────────
try:
    from nltk.stem import WordNetLemmatizer as _WNL
    _wnl = _WNL()
    # test it actually works before committing to it
    _wnl.lemmatize("prices")
    def _lemmatize_word(w):
        return _wnl.lemmatize(w)
except Exception:
    def _lemmatize_word(w):
        # simple suffix rules as fallback
        for suffix, replacement in [("ies","y"),("ves","f"),("ing",""),("ed",""),("es",""),("s","")]:
            if w.endswith(suffix) and len(w) - len(suffix) > 2:
                return w[:-len(suffix)] + replacement
        return w
 
 
# ── Nigerian Pidgin / slang normalization dictionary ─────────────────────────
# This maps common Pidgin words to their Standard English equivalents
# so the model understands them correctly
PIDGIN_MAP = {
    "dey":      "is",
    "don":      "has",
    "wan":      "want",
    "chop":     "eat",
    "abeg":     "please",
    "wahala":   "problem",
    "pikin":    "child",
    "pikini":   "child",
    "pikins":   "children",
    "fam":      "family",
    "sef":      "even",
    "nkor":     "what about",
    "sha":      "though",
    "na":       "is",
    "wetin":    "what",
    "dem":      "them",
    "dis":      "this",
    "dat":      "that",
    "oga":      "boss",
    "naija":    "nigeria",
    "9ja":      "nigeria",
    "tuale":    "salute",
    "e":        "it",
    "o":        "",          # filler word, remove
    "oo":       "",          # filler word, remove
    "ooo":      "",          # filler word, remove
    "ehn":      "",          # filler word, remove
    "nah":      "no",
    "kpakpa":   "exactly",
    "wey":      "that",
    "make":     "let",
    "comot":    "remove",
    "put":      "put",
    "no":       "not",
    "nor":      "not",
    "fit":      "can",
    "go":       "will",
    "come":     "came",
    "una":      "you all",
    "unu":      "you all",
    "pple":     "people",
    "ppl":      "people",
    "govt":     "government",
    "govnt":    "government",
    "cos":      "because",
    "cus":      "because",
    "bcuz":     "because",
    "mkt":      "market",
    "fud":      "food",
    "fd":       "food",
    "childrn":  "children",
    "childn":   "children",
    "rt":       "",          # retweet label, remove
}
 
# Extra food-crisis keywords to preserve even after stop word removal
DOMAIN_KEYWORDS = {
    "hunger", "hungry", "food", "eat", "starve", "starvation",
    "price", "prices", "expensive", "afford", "scarcity", "scarce",
    "inflation", "crisis", "shortage", "relief", "palliative",
    "poor", "poverty", "suffer", "hardship", "malnutrition",
    "rice", "garri", "yam", "beans", "tomato", "pepper",
}
 
# Standard English stop words, but we will KEEP domain keywords even if they
# appear in the stop word list (rare, but safe to handle)
STOP_WORDS = _base_stopwords - DOMAIN_KEYWORDS
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  CLEANING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
 
def remove_rt_prefix(text):
    """Remove retweet prefix:  RT @username:  """
    return re.sub(r"^RT\s+@\w+:\s*", "", text, flags=re.IGNORECASE)
 
def remove_urls(text):
    """Remove any http / https links"""
    return re.sub(r"http\S+|www\.\S+", "", text)
 
def remove_mentions(text):
    """Remove @username mentions"""
    return re.sub(r"@\w+", "", text)
 
def remove_hashtag_symbol(text):
    """Keep the word after # but remove the # symbol itself
       #FoodCrisis  →  FoodCrisis
    """
    return re.sub(r"#(\w+)", r"\1", text)
 
def remove_special_characters(text):
    """Remove punctuation, numbers, emojis, and non-ASCII characters"""
    # Remove emojis and non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove everything that is not a letter or space
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text
 
def normalize_whitespace(text):
    """Collapse multiple spaces into one and strip ends"""
    return re.sub(r"\s+", " ", text).strip()
 
def to_lowercase(text):
    return text.lower()
 
def apply_pidgin_map(text):
    """Replace Nigerian Pidgin words with Standard English equivalents"""
    words  = text.split()
    result = [PIDGIN_MAP.get(w, w) for w in words]
    # Remove empty strings left by filler word removal
    result = [w for w in result if w.strip()]
    return " ".join(result)
 
def remove_stopwords(text):
    """Remove common stop words, preserving food-crisis domain keywords"""
    words = text.split()
    return " ".join([w for w in words if w not in STOP_WORDS])
 
def lemmatize(text):
    """Reduce words to their base form: prices → price, struggling → struggle"""
    words = text.split()
    return " ".join([_lemmatize_word(w) for w in words])
 
def remove_short_words(text, min_length=2):
    """Remove single-character leftover tokens"""
    words = text.split()
    return " ".join([w for w in words if len(w) >= min_length])
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  MASTER PIPELINE  — applies all steps in correct order
# ─────────────────────────────────────────────────────────────────────────────
 
def clean_tweet(text):
    """Run a raw tweet through the full cleaning pipeline."""
    text = remove_rt_prefix(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtag_symbol(text)
    text = remove_special_characters(text)
    text = to_lowercase(text)
    text = apply_pidgin_map(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = remove_short_words(text)
    text = normalize_whitespace(text)
    return text
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  MAIN  — load → clean → save
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
 
    # ── Load raw data ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 2 - PREPROCESSING")
    print("=" * 60)
 
    raw_path = "data/lagos_food_crisis_tweets.csv"
    print(f"\n[1/5] Loading raw dataset from:  {raw_path}")
 
    df = pd.read_csv(raw_path)
    print(f"      Rows loaded: {len(df):,}")
    print(f"      Columns    : {list(df.columns)}")
 
    # ── Show a before sample ──────────────────────────────────────────────────
    print("\n[2/5] Sample BEFORE cleaning:")
    for i, row in df.head(3).iterrows():
        print(f"      [{row['label']}] {row['tweet_text'][:100]}...")
 
    # ── Apply cleaning pipeline ───────────────────────────────────────────────
    print("\n[3/5] Cleaning all tweets (this may take a moment)...")
    df["cleaned_text"] = df["tweet_text"].astype(str).apply(clean_tweet)
 
    # Drop rows where cleaning left an empty string
    before = len(df)
    df = df[df["cleaned_text"].str.strip() != ""]
    dropped = before - len(df)
    print(f"      Done. Dropped {dropped} empty rows after cleaning.")
 
    # ── Show an after sample ──────────────────────────────────────────────────
    print("\n[4/5] Sample AFTER cleaning:")
    for i, row in df.head(3).iterrows():
        print(f"      [{row['label']}] {row['cleaned_text'][:100]}")
 
    # ── Save cleaned dataset ──────────────────────────────────────────────────
    out_path = "data/cleaned_tweets.csv"
    df[["tweet_id", "cleaned_text", "label", "date", "location"]].to_csv(
        out_path, index=False
    )
 
    print(f"\n[5/5] Saved cleaned dataset to: {out_path}")
    print(f"      Total clean rows: {len(df):,}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\n✓ Preprocessing complete. Run features.py next.\n")
 

