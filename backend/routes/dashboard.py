from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from sqlalchemy import func, case
from datetime import datetime, timedelta
from collections import Counter
import re
from extensions import db
from models import Prediction, Tweet
from utils.forecast import linear_regression_forecast, get_next_7_dates

dashboard_bp = Blueprint('dashboard', __name__)

nlp_service = None

def init_nlp_service(service):
    global nlp_service
    nlp_service = service

def _tweet_query(start_date, end_date, location):
    """Base query joining Tweet + Prediction filtered by tweet_date and location."""
    q = db.session.query(Prediction, Tweet).join(Tweet, Prediction.tweet_id == Tweet.tweet_id).filter(
        Tweet.tweet_date >= start_date,
        Tweet.tweet_date <= end_date,
    )
    if location != 'All Lagos':
        q = q.filter(Tweet.location == location)
    return q

@dashboard_bp.route('/summary', methods=['GET'])
@jwt_required()
def get_summary():
    days = int(request.args.get('days', 30))
    location = request.args.get('location', 'All Lagos')

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    rows = _tweet_query(start_date, end_date, location).all()
    total_tweets = len(rows)

    if total_tweets == 0:
        return jsonify({
            'total_tweets': 0,
            'negative_pct': 0,
            'neutral_pct': 0,
            'positive_pct': 0,
            'avg_confidence': 0,
            'active_alerts': 0,
            'period_days': days,
            'has_data': False
        }), 200

    total_negative = sum(1 for p, _ in rows if p.label == 'Negative')
    total_neutral  = sum(1 for p, _ in rows if p.label == 'Neutral')
    total_positive = sum(1 for p, _ in rows if p.label == 'Positive')

    negative_pct = round(total_negative / total_tweets * 100, 1)
    neutral_pct  = round(total_neutral  / total_tweets * 100, 1)
    positive_pct = round(total_positive / total_tweets * 100, 1)
    avg_confidence = round(sum(p.confidence for p, _ in rows) / total_tweets, 1)

    from models import Alert
    active_alerts = Alert.query.filter_by(acknowledged=False).count()

    return jsonify({
        'total_tweets': total_tweets,
        'negative_pct': negative_pct,
        'neutral_pct': neutral_pct,
        'positive_pct': positive_pct,
        'avg_confidence': avg_confidence,
        'active_alerts': active_alerts,
        'period_days': days,
        'has_data': True
    }), 200

@dashboard_bp.route('/trends', methods=['GET'])
@jwt_required()
def get_trends():
    days = int(request.args.get('days', 30))
    location = request.args.get('location', 'All Lagos')

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    # Aggregate per day directly from Tweet + Prediction
    q = db.session.query(
        Tweet.tweet_date,
        func.count(Prediction.prediction_id).label('total'),
        func.sum(case((Prediction.label == 'Negative', 1), else_=0)).label('neg'),
        func.sum(case((Prediction.label == 'Neutral',  1), else_=0)).label('neu'),
        func.sum(case((Prediction.label == 'Positive', 1), else_=0)).label('pos'),
    ).join(Prediction, Prediction.tweet_id == Tweet.tweet_id).filter(
        Tweet.tweet_date >= start_date,
        Tweet.tweet_date <= end_date,
    )
    if location != 'All Lagos':
        q = q.filter(Tweet.location == location)
    daily = {row.tweet_date: row for row in q.group_by(Tweet.tweet_date).all()}

    labels   = []
    negative = []
    neutral  = []
    positive = []

    current_date = start_date
    while current_date <= end_date:
        labels.append(current_date.strftime('%Y-%m-%d'))
        row = daily.get(current_date)
        if row and row.total > 0:
            negative.append(round(row.neg / row.total * 100, 1))
            neutral.append( round(row.neu / row.total * 100, 1))
            positive.append(round(row.pos / row.total * 100, 1))
        else:
            negative.append(0)
            neutral.append(0)
            positive.append(0)
        current_date += timedelta(days=1)

    # Build response
    response_data = {
        'labels': labels,
        'negative': negative,
        'neutral': neutral,
        'positive': positive
    }
    
    # Add forecast data for any date range with enough data points
    non_zero_count = sum(1 for v in negative if v > 0)
    if non_zero_count >= 7:
        try:
            forecast_result = linear_regression_forecast(negative, days_ahead=7)

            if forecast_result and forecast_result['forecast']:
                forecast_dates = get_next_7_dates(end_date)
                response_data['forecast_labels'] = forecast_dates
                response_data['forecast_values'] = forecast_result['forecast']
                response_data['forecast_upper'] = forecast_result['confidence_upper']
                response_data['forecast_lower'] = forecast_result['confidence_lower']
                response_data['trend_direction'] = forecast_result['trend_direction']
                response_data['slope'] = forecast_result['slope']
                response_data['r_squared'] = forecast_result['r_squared']
                response_data['forecast_warning'] = forecast_result['warning']
                response_data['has_sufficient_data'] = forecast_result['has_sufficient_data']
                response_data['forecast_message'] = None
            else:
                response_data['forecast_labels'] = []
                response_data['forecast_values'] = []
                response_data['forecast_warning'] = forecast_result.get('warning')
                response_data['has_sufficient_data'] = False
                response_data['forecast_message'] = (
                    'Not enough data points to generate a forecast. '
                    'Please select a wider date range.'
                )
        except Exception as e:
            print(f"Error calculating forecast: {str(e)}")
            response_data['forecast_warning'] = 'Unable to calculate forecast'
            response_data['has_sufficient_data'] = False
            response_data['forecast_message'] = 'Unable to calculate forecast at this time.'
    else:
        response_data['forecast_labels'] = []
        response_data['forecast_values'] = []
        response_data['has_sufficient_data'] = False
        response_data['forecast_message'] = (
            'Not enough data points to generate a forecast. '
            'Please select a wider date range.'
        )

    return jsonify(response_data), 200

@dashboard_bp.route('/keywords', methods=['GET'])
@jwt_required()
def get_keywords():
    days = int(request.args.get('days', 7))
    sentiment = request.args.get('sentiment', 'Negative')

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    # Query directly from tweets/predictions so the date filter is always respected
    rows = (
        _tweet_query(start_date, end_date, 'All Lagos')
        .filter(Prediction.label == sentiment)
        .all()
    )

    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'is', 'are', 'was', 'be', 'have', 'has', 'it', 'of', 'that', 'this',
        'with', 'from', 'not', 'as', 'by', 'we', 'i', 'you', 'he', 'she', 'they',
        'rt', 'via', 'amp', 'http', 'https', 'co', 'our', 'my', 'so', 'can', 'do',
        'no', 'up', 'out', 'if', 'will', 'just', 'been', 'its', 'also', 'than',
    }

    word_counter = Counter()
    for _, tweet in rows:
        words = re.findall(r'\b[a-z]{3,}\b', tweet.raw_text.lower())
        for word in words:
            if word not in stop_words:
                word_counter[word] += 1

    top = word_counter.most_common(10)
    return jsonify({
        'keywords': [k for k, _ in top],
        'frequencies': [f for _, f in top],
    }), 200

@dashboard_bp.route('/heatmap', methods=['GET'])
@jwt_required()
def get_heatmap():
    days = int(request.args.get('days', 30))

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    # Query directly from tweets/predictions so the date filter is always respected
    rows = (
        db.session.query(Prediction, Tweet)
        .join(Tweet, Prediction.tweet_id == Tweet.tweet_id)
        .filter(
            Tweet.tweet_date >= start_date,
            Tweet.tweet_date <= end_date,
            Tweet.location.isnot(None),
            Tweet.location != 'All Lagos',
        )
        .all()
    )

    location_data = {}
    for pred, tweet in rows:
        loc = tweet.location
        if loc not in location_data:
            location_data[loc] = {'total': 0, 'negative': 0}
        location_data[loc]['total'] += 1
        if pred.label == 'Negative':
            location_data[loc]['negative'] += 1

    locations = []
    for loc, data in location_data.items():
        if data['total'] > 0:
            negative_pct = round(data['negative'] / data['total'] * 100, 1)
            locations.append({
                'name': loc,
                'negative_pct': negative_pct,
                'total': data['total'],
            })

    locations.sort(key=lambda x: x['negative_pct'], reverse=True)
    return jsonify({'locations': locations}), 200

@dashboard_bp.route('/stream-status', methods=['GET'])
@jwt_required()
def stream_status():
    """
    Returns current streaming status and recent tweets.
    Optional ?model= param re-classifies tweets with a specific model.
    """
    model = request.args.get('model', None)

    try:
        recent_rows = db.session.execute(
            db.select(Tweet, Prediction)
            .join(Prediction, Prediction.tweet_id == Tweet.tweet_id)
            .order_by(Tweet.created_at.desc())
            .limit(10)
        ).all()

        tweets_data = []

        if model and nlp_service:
            raw_texts  = [t.raw_text          for t, p in recent_rows]
            locations  = [t.location           for t, p in recent_rows]
            timestamps = [t.created_at.isoformat() for t, p in recent_rows]

            if model == 'ensemble':
                batch = nlp_service.predict_batch_ensemble(raw_texts)
            else:
                batch = nlp_service.predict_batch_with_model(raw_texts, model)

            for i, result in enumerate(batch):
                item = {
                    'text':       raw_texts[i][:100],
                    'label':      result['label'],
                    'confidence': result['confidence'],
                    'created_at': timestamps[i],
                    'location':   locations[i],
                }
                if result.get('ensemble'):
                    item['ensemble']   = True
                    item['individual'] = result['individual']
                    item['votes']      = result['votes']
                    item['total']      = result['total']
                tweets_data.append(item)
        else:
            for tweet, pred in recent_rows:
                tweets_data.append({
                    'text':       tweet.raw_text[:100],
                    'label':      pred.label,
                    'confidence': pred.confidence,
                    'created_at': tweet.created_at.isoformat(),
                    'location':   tweet.location,
                })

        return jsonify({
            'status': 'streaming',
            'recent_tweets': tweets_data,
            'total_streamed': Tweet.query.filter_by(source='simulated').count()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500