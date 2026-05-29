from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import desc
from extensions import db
from models import Alert, User
from services.alert_service import acknowledge_alert as acknowledge_alert_service, get_active_alerts as get_active_alerts_service

alerts_bp = Blueprint('alerts', __name__)

def init_alert_service(service):
    pass

@alerts_bp.route('', methods=['GET'])
@jwt_required()
def get_alerts():
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))

    query = Alert.query.order_by(desc(Alert.created_at))
    total = query.count()
    alerts = query.offset((page - 1) * limit).limit(limit).all()

    pages = (total + limit - 1) // limit

    alert_list = []
    for alert in alerts:
        alert_list.append({
            'alert_id': alert.alert_id,
            'alert_type': alert.alert_type,
            'negative_pct': alert.negative_pct,
            'location': alert.location,
            'alert_date': alert.alert_date.isoformat(),
            'is_emailed': alert.is_emailed,
            'acknowledged': alert.acknowledged,
            'created_at': alert.created_at.isoformat()
        })

    return jsonify({
        'alerts': alert_list,
        'total': total,
        'pages': pages,
        'current_page': page
    }), 200

@alerts_bp.route('/<alert_id>/acknowledge', methods=['PATCH'])
@jwt_required()
def acknowledge(alert_id):
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    try:
        alert = acknowledge_alert_service(alert_id, user)
        return jsonify({
            'message': 'Alert acknowledged',
            'alert_id': alert.alert_id
        }), 200
    except PermissionError as e:
        return jsonify({'error': str(e)}), 403
    except Exception as e:
        return jsonify({'error': 'Alert not found'}), 404

@alerts_bp.route('/active', methods=['GET'])
@jwt_required()
def get_active_alerts():
    alerts = get_active_alerts_service()
    alert_list = []
    for alert in alerts:
        alert_list.append({
            'alert_id': alert.alert_id,
            'alert_type': alert.alert_type,
            'negative_pct': alert.negative_pct,
            'location': alert.location,
            'alert_date': alert.alert_date.isoformat(),
            'is_emailed': alert.is_emailed,
            'acknowledged': alert.acknowledged,
            'created_at': alert.created_at.isoformat()
        })

    return jsonify({
        'alerts': alert_list,
        'count': len(alert_list)
    }), 200