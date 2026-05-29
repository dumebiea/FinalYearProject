from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db
from models import User

auth_bp = Blueprint('auth', __name__)
from flask import Blueprint
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 1. Define the Blueprint (if not already there)
auth_bp = Blueprint('auth', __name__)

# 2. Setup the "Real" NLP Tool
def init_nlp_service(app=None):
    """
    Initializes the NLTK VADER sentiment analyzer.
    This allows the app to score text as positive, negative, or neutral.
    """
    try:
        # Downloads the sentiment rules if they aren't already on your PC
        nltk.download('vader_lexicon', quiet=True)
        
        # We attach the analyzer to the app object so it's accessible everywhere
        app.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        print("[OK] NLP Sentiment Service: VADER initialized and ready.")
        return app.sentiment_analyzer
    except Exception as e:
        print(f"[ERROR] Failed to initialize NLP Service: {e}")
        return None

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'viewer')

    if not all([username, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Check if admin exists
    admin_exists = User.query.filter_by(role='admin').first() is not None
    if role == 'admin' and admin_exists:
        role = 'viewer'  # Default to viewer if admin already exists

    # Check if email exists
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 400

    # Create user
    user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password),
        role=role
    )
    db.session.add(user)
    db.session.commit()

    return jsonify({
        'message': 'User registered successfully',
        'user_id': user.user_id
    }), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({'error': 'Missing email or password'}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid credentials'}), 401

    # Update last login
    user.last_login = db.func.now()
    db.session.commit()

    access_token = create_access_token(identity=user.user_id)
    return jsonify({
        'access_token': access_token,
        'user': {
            'user_id': user.user_id,
            'username': user.username,
            'role': user.role
        }
    }), 200

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    user_id = get_jwt_identity()
    user = User.query.get_or_404(user_id)
    return jsonify({
        'user_id': user.user_id,
        'username': user.username,
        'email': user.email,
        'role': user.role
    }), 200