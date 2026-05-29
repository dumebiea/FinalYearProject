from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from services.nlp_service import NLPService

predict_bp = Blueprint('predict', __name__)

# Global NLP service instance
nlp_service = None

def init_nlp_service(service):
    global nlp_service
    nlp_service = service

def _label_message(label):
    if label == 'Negative':
        return "This tweet shows signs of food-related distress"
    if label == 'Neutral':
        return "This tweet does not show strong sentiment signals"
    return "This tweet shows a positive food security signal"


@predict_bp.route('', methods=['POST'])
@jwt_required()
def predict():
    data = request.get_json()
    tweet = data.get('tweet')
    model = data.get('model', 'bert')

    if not tweet or not tweet.strip():
        return jsonify({'error': 'Tweet text is required'}), 400

    if not nlp_service:
        return jsonify({'error': 'NLP service not initialized'}), 500

    try:
        if model == 'ensemble':
            result = nlp_service.predict_ensemble(tweet)
        else:
            result = nlp_service.predict_with_model(tweet, model)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    result['message'] = _label_message(result['label'])
    return jsonify(result), 200