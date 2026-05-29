import logging
import time
import random
from datetime import datetime
from threading import Thread

logger = logging.getLogger(__name__)

SEARCH_QUERIES = [
    "food price Nigeria",
    "hunger Lagos",
    "food shortage Nigeria",
    "food scarcity Lagos",
    "cost of food Nigeria",
]

FETCH_INTERVAL = 15 * 60  # 15 minutes

LOCATIONS = [
    'Oshodi', 'Lagos Island', 'Ikeja', 'Ikorodu',
    'Alimosho', 'Badagry', 'Epe', 'All Lagos',
]

_seen_urls = set()
_fetcher_thread = None


def init_news_fetcher(app, nlp_service):
    """
    Start the background news-fetching thread.
    If NEWS_API_KEY is absent, logs a warning and returns without starting.
    """
    global _fetcher_thread

    api_key = app.config.get('NEWS_API_KEY')
    if not api_key:
        logger.warning(
            "NEWS_API_KEY not set — news fetching disabled, "
            "continuing with Twitter data only"
        )
        return None

    def fetch_loop():
        logger.info("News fetcher started (15-minute interval)")
        while True:
            try:
                _fetch_and_process(app, nlp_service, api_key)
            except Exception as e:
                logger.error(f"News fetcher unexpected error: {e}")
            time.sleep(FETCH_INTERVAL)

    _fetcher_thread = Thread(target=fetch_loop, daemon=True)
    _fetcher_thread.start()
    logger.info("News fetcher thread started")
    return _fetcher_thread


def _fetch_and_process(app, nlp_service, api_key):
    """Fetch articles from NewsAPI and run them through the NLP pipeline."""
    try:
        from newsapi import NewsApiClient
    except ImportError:
        logger.error("newsapi-python not installed — run: pip install newsapi-python")
        return

    newsapi = NewsApiClient(api_key=api_key)

    articles = []
    for query in SEARCH_QUERIES:
        try:
            response = newsapi.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                page_size=5,
            )
            if response.get('status') == 'ok':
                articles.extend(response.get('articles', []))
        except Exception as e:
            logger.warning(f"NewsAPI query '{query}' failed: {e}")

    if not articles:
        logger.info("News fetcher: no articles returned from NewsAPI")
        return

    new_articles = [a for a in articles if a.get('url') not in _seen_urls]
    if not new_articles:
        logger.info("News fetcher: no new articles since last fetch")
        return

    logger.info(f"News fetcher: processing {len(new_articles)} new articles")

    with app.app_context():
        from models import db, Tweet, Prediction

        saved = 0
        for article in new_articles:
            url = article.get('url', '')
            title = (article.get('title') or '').strip()
            description = (article.get('description') or '').strip()

            # Combine headline and description into a single text string
            if title and description:
                text = f"{title}. {description}"
            elif title:
                text = title
            elif description:
                text = description
            else:
                continue

            published_at = article.get('publishedAt')
            if published_at:
                try:
                    pub_dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    tweet_date = pub_dt.date()
                except Exception:
                    tweet_date = datetime.now().date()
            else:
                tweet_date = datetime.now().date()

            try:
                result = nlp_service.predict_one(text)

                tweet_record = Tweet(
                    source='simulated',
                    raw_text=text,
                    location=random.choice(LOCATIONS),
                    tweet_date=tweet_date,
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

                _seen_urls.add(url)
                saved += 1

            except Exception as e:
                logger.error(f"News fetcher: error processing article: {e}")
                db.session.rollback()
                continue

        try:
            db.session.commit()
            logger.info(f"News fetcher: saved {saved} articles to database")
        except Exception as e:
            logger.error(f"News fetcher: commit failed: {e}")
            db.session.rollback()
