import uuid
from datetime import datetime
from extensions import db
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), nullable=False, default='viewer')  # 'admin' or 'viewer'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class Tweet(db.Model):
    __tablename__ = 'tweets'
    tweet_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source = db.Column(db.String(20), nullable=False)  # 'upload', 'search', 'scheduled'
    raw_text = db.Column(db.Text, nullable=False)
    location = db.Column(db.String(100), nullable=True)
    tweet_date = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='tweet', lazy=True)

    def to_dict(self):
        return {
            'tweet_id': self.tweet_id,
            'source': self.source,
            'raw_text': self.raw_text,
            'location': self.location,
            'tweet_date': self.tweet_date.isoformat() if self.tweet_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Prediction(db.Model):
    __tablename__ = 'predictions'
    prediction_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tweet_id = db.Column(db.String(36), db.ForeignKey('tweets.tweet_id'), nullable=False)
    label = db.Column(db.String(10), nullable=False)  # 'Negative','Neutral','Positive'
    confidence = db.Column(db.Float, nullable=False)
    model_used = db.Column(db.String(50), nullable=False)
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'prediction_id': self.prediction_id,
            'tweet_id': self.tweet_id,
            'label': self.label,
            'confidence': self.confidence,
            'model_used': self.model_used,
            'predicted_at': self.predicted_at.isoformat() if self.predicted_at else None
        }

class SentimentSummary(db.Model):
    __tablename__ = 'sentiment_summaries'
    summary_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    summary_date = db.Column(db.Date, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    total_tweets = db.Column(db.Integer, nullable=False, default=0)
    negative_count = db.Column(db.Integer, nullable=False, default=0)
    neutral_count = db.Column(db.Integer, nullable=False, default=0)
    positive_count = db.Column(db.Integer, nullable=False, default=0)
    negative_pct = db.Column(db.Float, nullable=False, default=0.0)
    avg_confidence = db.Column(db.Float, nullable=False, default=0.0)
    top_keywords = db.Column(db.Text, nullable=True)  # JSON string e.g. '["hunger","price","garri"]'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    alerts = db.relationship('Alert', backref='summary', lazy=True)
    keywords = db.relationship('KeywordFrequency', backref='summary', lazy=True)

    def to_dict(self):
        return {
            'summary_id': self.summary_id,
            'summary_date': self.summary_date.isoformat() if self.summary_date else None,
            'location': self.location,
            'total_tweets': self.total_tweets,
            'negative_count': self.negative_count,
            'neutral_count': self.neutral_count,
            'positive_count': self.positive_count,
            'negative_pct': self.negative_pct,
            'avg_confidence': self.avg_confidence,
            'top_keywords': self.top_keywords,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Alert(db.Model):
    __tablename__ = 'alerts'
    alert_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    summary_id = db.Column(db.String(36), db.ForeignKey('sentiment_summaries.summary_id'), nullable=False)
    alert_type = db.Column(db.String(10), nullable=False)  # 'warning' or 'crisis'
    negative_pct = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    alert_date = db.Column(db.Date, nullable=False)
    is_emailed = db.Column(db.Boolean, default=False)
    acknowledged = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'alert_id': self.alert_id,
            'summary_id': self.summary_id,
            'alert_type': self.alert_type,
            'negative_pct': self.negative_pct,
            'location': self.location,
            'alert_date': self.alert_date.isoformat() if self.alert_date else None,
            'is_emailed': self.is_emailed,
            'acknowledged': self.acknowledged,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class KeywordFrequency(db.Model):
    __tablename__ = 'keyword_frequencies'
    keyword_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    summary_id = db.Column(db.String(36), db.ForeignKey('sentiment_summaries.summary_id'), nullable=False)
    keyword = db.Column(db.String(100), nullable=False)
    frequency = db.Column(db.Integer, nullable=False)
    sentiment_label = db.Column(db.String(10), nullable=False)

    def to_dict(self):
        return {
            'keyword_id': self.keyword_id,
            'summary_id': self.summary_id,
            'keyword': self.keyword,
            'frequency': self.frequency,
            'sentiment_label': self.sentiment_label
        }