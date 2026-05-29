import pandas as pd
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from extensions import db
from models import Tweet, Prediction
from services.nlp_service import NLPService
from services.summary_service import generate_daily_summary

tweets_bp = Blueprint('tweets', __name__)

# Global services
nlp_service = None
# No global service needed
# summary_service = None

def init_nlp_service(service):
    global nlp_service
    nlp_service = service

def init_summary_service(service):
    pass # Kept for compatibility

@tweets_bp.route('/upload', methods=['POST'])
@jwt_required()
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be CSV'}), 400

    # Read CSV
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400

    # Find tweet column
    tweet_col = None
    for col in ['tweet_text', 'text', 'tweet', 'content']:
        if col in df.columns:
            tweet_col = col
            break

    if not tweet_col:
        return jsonify({'error': 'CSV must have a column named tweet_text, text, tweet, or content'}), 400

    tweets = df[tweet_col].dropna().tolist()
    if not tweets:
        return jsonify({'error': 'No valid tweets found in CSV'}), 400

    model = request.form.get('model', 'bert')

    # Classify tweets with the chosen model
    if model == 'ensemble':
        results = nlp_service.predict_batch_ensemble(tweets)
    elif model != 'bert':
        results = nlp_service.predict_batch_with_model(tweets, model)
    else:
        results = nlp_service.predict_batch(tweets)

    # Save to database
    total    = len(results)
    negative = sum(1 for r in results if r['label'] == 'Negative')
    neutral  = sum(1 for r in results if r['label'] == 'Neutral')
    positive = sum(1 for r in results if r['label'] == 'Positive')

    for result in results:
        tweet_obj = Tweet(
            source='upload',
            raw_text=result['tweet'],
            location=None,
            tweet_date=datetime.now().date()
        )
        db.session.add(tweet_obj)
        db.session.flush()

        prediction = Prediction(
            tweet_id=tweet_obj.tweet_id,
            label=result['label'],
            confidence=result['confidence'],
            model_used=result['model_used']
        )
        db.session.add(prediction)

    db.session.commit()

    negative_pct = (negative / total * 100) if total > 0 else 0

    response = {
        'total': total,
        'negative': negative,
        'neutral': neutral,
        'positive': positive,
        'negative_pct': round(negative_pct, 1),
        'model_used': results[0]['model_used'] if results else 'Unknown',
        'message': f'{total} tweets classified successfully',
    }

    # For ensemble, add per-model vote breakdown
    if model == 'ensemble' and results and results[0].get('ensemble'):
        model_names = [ind['model'] for ind in results[0]['individual']]
        breakdown = {m: {'Negative': 0, 'Neutral': 0, 'Positive': 0} for m in model_names}
        for r in results:
            for ind in r.get('individual', []):
                breakdown[ind['model']][ind['label']] += 1
        response['ensemble_breakdown'] = breakdown

    return jsonify(response), 200

@tweets_bp.route('/search', methods=['POST'])
@jwt_required()
def search_tweets():
    data = request.get_json()
    keywords = data.get('keywords', [])
    location = data.get('location', 'Lagos')

    if not keywords:
        return jsonify({'error': 'Keywords are required'}), 400

    # Simulate search by filtering cleaned_tweets.csv
    df = pd.read_csv('data/cleaned_tweets.csv')
    
    # Filter by keywords
    mask = df['cleaned_text'].str.contains('|'.join(keywords), case=False, na=False)
    if location and location != 'All Lagos':
        mask &= df['location'].str.contains(location, case=False, na=False)
    
    filtered_df = df[mask]
    tweets = filtered_df['cleaned_text'].tolist()[:100]  # Limit to 100

    if not tweets:
        return jsonify({
            'total': 0, 'negative': 0, 'neutral': 0, 'positive': 0, 'results': []
        }), 200

    model = data.get('model', 'bert')

    # Classify with chosen model
    if model == 'ensemble':
        results = nlp_service.predict_batch_ensemble(tweets)
    elif model != 'bert':
        results = nlp_service.predict_batch_with_model(tweets, model)
    else:
        results = nlp_service.predict_batch(tweets)

    # Save to database
    total    = len(results)
    negative = sum(1 for r in results if r['label'] == 'Negative')
    neutral  = sum(1 for r in results if r['label'] == 'Neutral')
    positive = sum(1 for r in results if r['label'] == 'Positive')

    saved_results = []
    for result in results:
        tweet_obj = Tweet(
            source='search',
            raw_text=result['tweet'],
            location=location,
            tweet_date=datetime.now().date()
        )
        db.session.add(tweet_obj)
        db.session.flush()

        prediction = Prediction(
            tweet_id=tweet_obj.tweet_id,
            label=result['label'],
            confidence=result['confidence'],
            model_used=result['model_used']
        )
        db.session.add(prediction)

        row = {
            'tweet':      result['tweet'],
            'label':      result['label'],
            'confidence': result['confidence'],
            'location':   location,
            'model_used': result.get('model_used', 'BERT'),
        }
        if result.get('ensemble'):
            row['ensemble']   = True
            row['individual'] = result['individual']
            row['votes']      = result['votes']
            row['total']      = result['total']
        saved_results.append(row)

    db.session.commit()

    from flask import current_app
    generate_daily_summary(current_app)

    return jsonify({
        'total':      total,
        'negative':   negative,
        'neutral':    neutral,
        'positive':   positive,
        'model_used': results[0]['model_used'] if results else 'BERT',
        'results':    saved_results,
    }), 200