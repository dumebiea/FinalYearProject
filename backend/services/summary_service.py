import json
from datetime import date
from collections import Counter
from extensions import db
from models import Prediction, Tweet, SentimentSummary, KeywordFrequency, Alert
from services.alert_service import send_alert_email

def generate_daily_summary(app):
    """
    Generate daily sentiment summary for each location and 'All Lagos'.
    Must be called inside Flask app context.
    """
    with app.app_context():
        today = date.today()
        # 2. Get all Prediction records from today joined with Tweet records
        results = db.session.query(Prediction, Tweet).join(Tweet, Prediction.tweet_id == Tweet.tweet_id).filter(
            db.func.date(Prediction.predicted_at) == today
        ).all()

        if not results:
            print(f"No predictions found for {today}")
            return {"summaries_created": 0, "alerts_created": 0}

        # 3. Group predictions by tweet.location
        location_groups = {}
        for pred, tweet in results:
            loc = tweet.location or "Unknown"
            if loc not in location_groups:
                location_groups[loc] = []
            location_groups[loc].append((pred, tweet))

        # 5. Also create/update an 'All Lagos' summary combining all locations
        location_groups['All Lagos'] = results

        summaries_created = 0
        alerts_created = 0

        # 4. For each location group
        for location, group in location_groups.items():
            total = len(group)
            if total == 0: continue

            # a. Count total, negative, neutral, positive predictions
            neg = sum(1 for p, t in group if p.label == 'Negative')
            neu = sum(1 for p, t in group if p.label == 'Neutral')
            pos = sum(1 for p, t in group if p.label == 'Positive')
            
            # b. Calculate negative_pct = negative_count / total * 100
            neg_pct = round((neg / total * 100), 2) if total > 0 else 0
            
            # c. Calculate avg_confidence
            avg_conf = round(sum(p.confidence for p, t in group) / total, 4) if total > 0 else 0

            # d. Extract top 10 keywords from the tweet texts using simple word frequency count
            all_text = " ".join([t.raw_text for p, t in group])
            # Filter out words shorter than 3 characters
            words = [w.lower() for w in all_text.split() if len(w) >= 3]
            top_keywords_with_counts = Counter(words).most_common(10)
            top_keywords = [item[0] for item in top_keywords_with_counts]

            # e. Check if a SentimentSummary already exists for today + location
            summary = SentimentSummary.query.filter_by(summary_date=today, location=location).first()
            if summary:
                summary.total_tweets = total
                summary.negative_count = neg
                summary.neutral_count = neu
                summary.positive_count = pos
                summary.negative_pct = neg_pct
                summary.avg_confidence = avg_conf
                summary.top_keywords = json.dumps(top_keywords)
            else:
                summary = SentimentSummary(
                    summary_date=today,
                    location=location,
                    total_tweets=total,
                    negative_count=neg,
                    neutral_count=neu,
                    positive_count=pos,
                    negative_pct=neg_pct,
                    avg_confidence=avg_conf,
                    top_keywords=json.dumps(top_keywords)
                )
                db.session.add(summary)
            
            db.session.flush() # ensure id is available

            # f. Save KeywordFrequency records for top 10 keywords (delete old ones first)
            KeywordFrequency.query.filter_by(summary_id=summary.summary_id).delete()
            word_counts = Counter(words)
            for kw in top_keywords:
                # Use dominant sentiment for the keyword label
                k_neg = sum(1 for p, t in group if kw in t.raw_text.lower() and p.label == 'Negative')
                k_pos = sum(1 for p, t in group if kw in t.raw_text.lower() and p.label == 'Positive')
                k_neu = sum(1 for p, t in group if kw in t.raw_text.lower() and p.label == 'Neutral')
                
                dominant_sentiment = 'Neutral'
                if k_neg >= k_pos and k_neg >= k_neu: dominant_sentiment = 'Negative'
                elif k_pos >= k_neg and k_pos >= k_neu: dominant_sentiment = 'Positive'
                
                kf = KeywordFrequency(
                    summary_id=summary.summary_id,
                    keyword=kw,
                    frequency=word_counts[kw],
                    sentiment_label=dominant_sentiment
                )
                db.session.add(kf)
            
            db.session.commit()
            summaries_created += 1

            # 6. After summaries saved, call check_and_create_alerts() for each
            if check_and_create_alerts(summary, app):
                alerts_created += 1

        # 7. Return { summaries_created, alerts_created }
        return {"summaries_created": summaries_created, "alerts_created": alerts_created}

def check_and_create_alerts(summary, app):
    """
    Check thresholds and create alerts.
    """
    with app.app_context():
        alert_created = False
        # If negative_pct > 50 and no crisis alert exists for today + location:
        if summary.negative_pct > 50:
            existing = Alert.query.filter_by(alert_date=summary.summary_date, location=summary.location, alert_type='crisis').first()
            if not existing:
                alert = Alert(
                    summary_id=summary.summary_id,
                    alert_type='crisis',
                    negative_pct=summary.negative_pct,
                    location=summary.location,
                    alert_date=summary.summary_date
                )
                db.session.add(alert)
                db.session.commit()
                # Call send_alert_email()
                try:
                    send_alert_email(alert, summary)
                except Exception as e:
                    print(f"Failed to trigger email for {alert.alert_id}: {e}")
                alert_created = True
        # Elif negative_pct > 35 and no warning alert exists for today + location:
        elif summary.negative_pct > 35:
            existing = Alert.query.filter_by(alert_date=summary.summary_date, location=summary.location, alert_type='warning').first()
            if not existing:
                alert = Alert(
                    summary_id=summary.summary_id,
                    alert_type='warning',
                    negative_pct=summary.negative_pct,
                    location=summary.location,
                    alert_date=summary.summary_date
                )
                db.session.add(alert)
                db.session.commit()
                # Call send_alert_email()
                try:
                    send_alert_email(alert, summary)
                except Exception as e:
                    print(f"Failed to trigger email for {alert.alert_id}: {e}")
                alert_created = True
        return alert_created