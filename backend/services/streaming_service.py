import json
import random
import time
from datetime import datetime, timedelta
import pandas as pd
from threading import Thread
import logging

logger = logging.getLogger(__name__)

class DataStreamSimulator:
    """
    Simulates real-time Twitter stream by releasing uncleaned tweets in batches.
    The backend preprocessing pipeline cleans them just like real tweets would be.
    """
    
    def __init__(self, dataset_path='data/lagos_food_crisis_tweets.csv', 
                 batch_size=10, interval_seconds=60):
        """
        Args:
            dataset_path: path to CSV with RAW uncleaned tweets
            batch_size: how many tweets per batch (like 10 tweets per minute)
            interval_seconds: how often to release a batch (60 = every minute)
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.interval = interval_seconds
        self.current_index = 0
        self.tweets = []
        self.is_running = False
        self.stream_thread = None
        
        # Load dataset on init
        self._load_dataset()
    
    def _load_dataset(self):
        """Load raw uncleaned tweets from CSV"""
        try:
            df = pd.read_csv(self.dataset_path)
            
            # Handle different column names for raw tweet text
            text_col = None
            if 'tweet_text' in df.columns:
                text_col = 'tweet_text'
            elif 'raw_text' in df.columns:
                text_col = 'raw_text'
            elif 'text' in df.columns:
                text_col = 'text'
            elif 'cleaned_text' in df.columns:
                text_col = 'cleaned_text'
            elif 'tweet' in df.columns:
                text_col = 'tweet'
            else:
                logger.error(f"No raw text column found. Available: {df.columns.tolist()}")
                return
            
            # Extract tweets as list of dicts
            self.tweets = []
            for idx, row in df.iterrows():
                tweet_dict = {
                    'text': str(row[text_col]),  # RAW uncleaned text
                }
                self.tweets.append(tweet_dict)
            
            logger.info(f"Loaded {len(self.tweets)} raw uncleaned tweets from {self.dataset_path}")
            
            # Shuffle so we get variety
            random.shuffle(self.tweets)
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.tweets = []
    
    def get_next_batch(self):
        """
        Get the next batch of uncleaned tweets.
        Returns a list of tweet dicts with 'text' and 'source' fields.
        The text is RAW and uncleaned — preprocessing happens in the API.
        """
        if not self.tweets:
            logger.warning("No tweets loaded in stream")
            return []
        
        # Wrap around if we reach the end
        if self.current_index >= len(self.tweets):
            self.current_index = 0
            random.shuffle(self.tweets)  # Reshuffle for next round
            logger.info("Stream cycled — starting over with shuffled data")
        
        # Get batch
        batch_end = min(self.current_index + self.batch_size, len(self.tweets))
        batch = self.tweets[self.current_index:batch_end]
        
        # Add metadata
        for tweet in batch:
            tweet['source'] = 'simulated'  # Mark as simulated
            # Use a random date within the last 30 days so the date-range
            # filter on the dashboard produces meaningfully different subsets.
            tweet['tweet_date'] = (datetime.now() - timedelta(days=random.randint(0, 29))).date()
            tweet['location'] = random.choice([
                'Oshodi', 'Lagos Island', 'Ikeja', 'Ikorodu',
                'Alimosho', 'Badagry', 'Epe', 'All Lagos'
            ])
            # Note: 'text' field is UNCLEANED raw tweet
        
        self.current_index = batch_end
        
        logger.info(
            f"Generated batch of {len(batch)} RAW uncleaned tweets "
            f"(index {self.current_index}/{len(self.tweets)})"
        )
        
        return batch
    
    def start_streaming(self, callback):
        """
        Start the stream in a background thread.
        callback(batch) is called with each new batch of uncleaned tweets.
        
        Args:
            callback: function that takes a list of tweet dicts with raw text
        """
        if self.is_running:
            logger.warning("Stream already running")
            return
        
        self.is_running = True
        
        def stream_loop():
            logger.info(
                f"Starting simulated stream with RAW uncleaned tweets "
                f"(batch_size={self.batch_size}, interval={self.interval}s)"
            )
            while self.is_running:
                try:
                    batch = self.get_next_batch()
                    if batch:
                        callback(batch)
                    time.sleep(self.interval)
                except Exception as e:
                    logger.error(f"Error in stream loop: {e}")
                    time.sleep(5)  # Wait before retrying
        
        self.stream_thread = Thread(target=stream_loop, daemon=True)
        self.stream_thread.start()
        logger.info("Stream thread started")
    
    def stop_streaming(self):
        """Stop the stream"""
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        logger.info("Stream stopped")


# Global stream instance
data_stream = None

def init_stream(app):
    """Initialize the simulated data stream with raw uncleaned tweets"""
    global data_stream
    
    with app.app_context():
        from services.nlp_service import NLPService
        from models import db, Tweet, Prediction
        
        # Create NLPService instance
        nlp = NLPService(app.config)
        
        # Create stream — uses raw uncleaned tweets
        data_stream = DataStreamSimulator(
            dataset_path='data/lagos_food_crisis_tweets.csv',
            batch_size=10,
            interval_seconds=60
        )
        
        def process_batch(batch):
            """
            Called when new batch of RAW uncleaned tweets arrives.
            The nlp.predict_one() function handles the cleaning internally.
            """
            logger.info(f"Processing batch of {len(batch)} RAW uncleaned tweets")
            try:
                # Create application context for this batch processing
                with app.app_context():
                    for tweet in batch:
                        raw_text = tweet['text']
                        
                        # nlp.predict_one() automatically cleans the tweet
                        result = nlp.predict_one(raw_text)
                        
                        # Save to database
                        tweet_record = Tweet(
                            source=tweet.get('source', 'simulated'),
                            raw_text=raw_text,
                            location=tweet.get('location'),
                            tweet_date=tweet.get('tweet_date'),
                        )
                        db.session.add(tweet_record)
                        db.session.flush()
                        
                        prediction_record = Prediction(
                            tweet_id=tweet_record.tweet_id,
                            label=result['label'],
                            confidence=result['confidence'],
                            model_used=result['model_used'],
                        )
                        db.session.add(prediction_record)
                    
                    db.session.commit()
                    logger.info(f"Batch of {len(batch)} tweets saved to database")
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                db.session.rollback()
        
        # Start the stream
        data_stream.start_streaming(process_batch)
        logger.info("Data stream initialized with RAW uncleaned tweets")
    
    return data_stream

def get_stream():
    """Get the global stream instance"""
    return data_stream