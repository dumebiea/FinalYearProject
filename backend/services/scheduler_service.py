import pandas as pd
import random
from datetime import date, datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from extensions import db
from models import Tweet, Prediction
from services.summary_service import generate_daily_summary
from config import Config

def init_scheduler(app):
    """
    Initialize APScheduler BackgroundScheduler with requested jobs.
    """
    scheduler = BackgroundScheduler()
    
    # Add daily summary job: runs every day at 23:55
    scheduler.add_job(
        func=generate_daily_summary,
        args=[app],
        trigger='cron',
        hour=23,
        minute=55,
        id='daily_summary_job'
    )
    
    # Add collection job: runs every 6 hours
    scheduler.add_job(
        func=scheduled_collection,
        args=[app],
        trigger='interval',
        hours=6,
        id='collection_job'
    )
    
    scheduler.start()
    print("APScheduler started: Daily summary at 23:55, Collection every 6 hours.")
    return scheduler

def scheduled_collection(app):
    """
    Simulate scheduled collection by taking 50 random rows not yet in the database.
    Fetches from data/cleaned_tweets.csv and classifies them.
    """
    with app.app_context():
        try:
            # Load dataset from config path
            df = pd.read_csv(Config.DATASET_PATH)
            
            # Fetch existing tweet texts to ensure we only add new ones
            existing_texts = {t.raw_text for t in db.session.query(Tweet.raw_text).all()}
            
            # Filter the dataframe for tweets not already in our database
            # Handling potential NaN in 'cleaned_text'
            df = df.dropna(subset=['cleaned_text'])
            new_tweets_df = df[~df['cleaned_text'].isin(existing_texts)]
            
            if new_tweets_df.empty:
                print("Scheduled Collection: No new tweets found to ingest.")
                return
            
            # Take 50 random rows (or all available if less than 50)
            count = min(50, len(new_tweets_df))
            sample = new_tweets_df.sample(n=count)
            
            # Import NLP service to classify the new tweets
            from services.nlp_service import NLPService
            nlp = NLPService(app.config)
            
            tweets_added = 0
            for _, row in sample.iterrows():
                # Create primary Tweet record
                # Use a random date within the last 89 days so scheduled tweets
                # are spread across the full date-range the dashboard can display.
                new_tweet = Tweet(
                    source='scheduled',
                    raw_text=row['cleaned_text'],
                    location=row.get('location', 'Lagos'),
                    tweet_date=datetime.now().date() - timedelta(days=random.randint(0, 89)),
                )
                db.session.add(new_tweet)
                db.session.flush() # To populate new_tweet.tweet_id
                
                # Generate prediction using the AI model
                pred_res = nlp.predict(new_tweet.raw_text)
                
                # Create corresponding Prediction record
                new_prediction = Prediction(
                    tweet_id=new_tweet.tweet_id,
                    label=pred_res['label'],
                    confidence=pred_res['confidence'],
                    model_used=pred_res['model_used']
                )
                db.session.add(new_prediction)
                tweets_added += 1
            
            db.session.commit()
            print(f"Scheduled Collection: Successfully ingested {tweets_added} tweets.")
            
        except Exception as e:
            print(f"Error in scheduled_collection job: {str(e)}")
            db.session.rollback()