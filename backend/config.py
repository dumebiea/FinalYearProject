import os
from datetime import timedelta
from dotenv import load_dotenv

# Load .env from the project root (one level above backend/)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

class Config:
    SECRET_KEY = 'food-crisis-secret-key-2025'
    JWT_SECRET_KEY = 'jwt-food-crisis-2025'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)

    # Database — SQLite is used by default for development and pilot study purposes.
    # Set DATABASE_URL in .env to use PostgreSQL in production.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get('DATABASE_URL')
        or f'sqlite:///{os.path.join(BASE_DIR, "backend", "instance", "food_crisis.db")}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Email — values loaded from .env
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')
    ALERT_RECIPIENT_EMAIL = os.environ.get('ALERT_RECIPIENT_EMAIL')

    # Model paths (absolute paths)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BERT_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_bert_model')
    SVM_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'svm_model_proba.pkl')
    LR_MODEL_PATH  = os.path.join(BASE_DIR, 'model', 'lr_model.pkl')
    TFIDF_PATH = os.path.join(BASE_DIR, 'model', 'tfidf_vectorizer.pkl')
    ENCODER_PATH = os.path.join(BASE_DIR, 'model', 'label_encoder.pkl')
    DATASET_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_tweets.csv')

    # NewsAPI
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

    # Alert thresholds
    WARNING_THRESHOLD = 35.0
    CRISIS_THRESHOLD = 50.0