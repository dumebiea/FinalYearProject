from flask import Flask, jsonify
from extensions import db, jwt, mail, cors
from config import Config
from models import User, Tweet, Prediction, SentimentSummary, Alert, KeywordFrequency
from services.nlp_service import NLPService
from services.scheduler_service import init_scheduler
from services.streaming_service import init_stream
from services.news_service import init_news_fetcher
from routes.auth import auth_bp, init_nlp_service as init_auth_nlp
from routes.predict import predict_bp, init_nlp_service as init_predict_nlp
from routes.tweets import tweets_bp, init_nlp_service as init_tweets_nlp
from routes.dashboard import dashboard_bp, init_nlp_service as init_dashboard_nlp
from routes.alerts import alerts_bp
import pandas as pd
from datetime import datetime, timedelta
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)
    jwt.init_app(app)
    mail.init_app(app)
    cors.init_app(app, origins='*')

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(predict_bp, url_prefix='/api/predict')
    app.register_blueprint(tweets_bp, url_prefix='/api/tweets')
    app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
    app.register_blueprint(alerts_bp, url_prefix='/api/alerts')

    # Health check endpoint
    @app.route('/api/health', methods=['GET'])
    def health_check():
        model_name = "bert" if os.path.exists(app.config['BERT_MODEL_PATH']) else "svm"
        return jsonify({
            "status": "ok",
            "model": model_name,
            "timestamp": datetime.now().isoformat()
        })

    # Initialize services and database
    with app.app_context():
        # Initialize NLP Service for routes
        nlp_service = NLPService(app.config)
        init_auth_nlp(nlp_service)
        init_predict_nlp(nlp_service)
        init_tweets_nlp(nlp_service)
        init_dashboard_nlp(nlp_service)

        # Create database tables
        db.create_all()

        # 1. Import and call init_scheduler(app) after db.create_all()
        init_scheduler(app)

        # 2. Initialize streaming service ← ADD THIS BLOCK
        init_stream(app)

        # 3. Initialize background news fetcher (no-op if NEWS_API_KEY is absent)
        init_news_fetcher(app, nlp_service)

        # Seed data if database is empty, then ensure tweet dates are recent
        seed_data(app)
        migrate_tweet_dates(app)

    return app

def migrate_tweet_dates(app):
    """
    Redistribute tweet dates to span the last 90 days so the date range
    filter on the dashboard returns meaningfully different subsets.
    Only runs when the majority of tweets carry dates older than 100 days.
    """
    with app.app_context():
        today = datetime.now().date()
        cutoff = today - timedelta(days=100)

        old_tweets = (
            Tweet.query
            .filter(db.or_(Tweet.tweet_date == None, Tweet.tweet_date < cutoff))
            .order_by(Tweet.created_at)
            .all()
        )

        if not old_tweets:
            return

        total = len(old_tweets)
        print(f"Migrating {total} tweet dates to the last 90 days...")

        for i, tweet in enumerate(old_tweets):
            # Spread evenly: first tweet → 89 days ago, last tweet → today
            days_ago = int((1 - i / total) * 89)
            tweet.tweet_date = today - timedelta(days=days_ago)

        db.session.commit()
        print(f"Migration complete: {total} tweet dates redistributed across last 90 days.")


def seed_data(app):
    """Seed initial data if database is empty."""
    if SentimentSummary.query.count() > 0:
        return

    print("Seeding initial data...")

    if User.query.filter_by(username='admin').first() is None:
        admin = User(username='admin', email='admin@foodcrisis.ng', role='admin')
        admin.set_password('Admin@123')
        db.session.add(admin)
        print("[OK] Admin user created")

    if User.query.filter_by(username='viewer').first() is None:
        viewer = User(username='viewer', email='viewer@foodcrisis.ng', role='viewer')
        viewer.set_password('Viewer@123')
        db.session.add(viewer)
        print("[OK] Viewer user created")

    db.session.commit()

    tweet_count = 0
    try:
        df = pd.read_csv(app.config['DATASET_PATH'])
        sample = df.head(200).reset_index(drop=True)
        from services.nlp_service import NLPService
        nlp = NLPService(app.config)
        total_to_seed = len(sample)
        today = datetime.now().date()

        for i, row in sample.iterrows():
            tweetText = row['cleaned_text'] if pd.notna(row['cleaned_text']) else ""
            if not tweetText:
                continue

            # Spread dates across last 89 days (oldest → 89 days ago, newest → today)
            days_ago = int((1 - i / total_to_seed) * 89)
            tweet_date = today - timedelta(days=days_ago)

            tweet = Tweet(
                source='scheduled',
                raw_text=tweetText,
                location=row.get('location', 'Lagos'),
                tweet_date=tweet_date,
            )
            db.session.add(tweet)
            db.session.flush()

            res = nlp.predict_one(tweet.raw_text)
            pred = Prediction(
                tweet_id=tweet.tweet_id,
                label=res['label'],
                confidence=res['confidence'],
                model_used=res['model_used'],
            )
            db.session.add(pred)
            tweet_count += 1

        db.session.commit()
        print(f"[OK] {tweet_count} tweets seeded with dates spread across last 90 days")
    except Exception as e:
        print(f"Seed data error: {e}")
        import traceback
        traceback.print_exc()

    try:
        from services.summary_service import generate_daily_summary
        result = generate_daily_summary(app)
        print(f"[OK] Initial summary generated ({result.get('summaries_created', 0)} summaries, {result.get('alerts_created', 0)} alerts)")
    except Exception as e:
        print(f"Summary generation error: {e}")
        import traceback
        traceback.print_exc()

    print("Seeding completed.")

if __name__ == '__main__':
    app = create_app()
    # Server starts at http://localhost:5000
    app.run(debug=True, port=5000)