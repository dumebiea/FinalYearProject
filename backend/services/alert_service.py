import json
import logging
from flask_mail import Message
from extensions import db, mail
from config import Config

def send_alert_email(alert, summary):
    """
    Send an email notification for the alert using Flask-Mail.
    """
    try:
        # Email subject definition
        if alert.alert_type == 'crisis':
            subject = f"🚨 FOOD CRISIS ALERT — Lagos {alert.location} {alert.alert_date}"
        else:
            subject = f"⚠️ Food Security Warning — Lagos {alert.location} {alert.alert_date}"

        # Extract keywords for the email body
        keywords = []
        if summary.top_keywords:
            try:
                keywords = json.loads(summary.top_keywords)
            except json.JSONDecodeError:
                keywords = []
        
        keywords_str = ", ".join(keywords) if keywords else "None detected"

        # Email body (plain text) exactly as requested
        body = f"""FOOD CRISIS DETECTION SYSTEM — ALERT NOTIFICATION

Alert Type : {alert.alert_type.upper()}
Location   : {alert.location}
Date       : {alert.alert_date}
Negative % : {alert.negative_pct}%
Total Tweets Analysed : {summary.total_tweets}

This alert was generated because negative sentiment in food-related
discussions exceeded the {"50%" if alert.alert_type == 'crisis' else "35%"} threshold.

Top Keywords: {keywords_str}

Please log in to the Food Crisis Detection dashboard to review
the full sentiment analysis and acknowledge this alert.

— Food Crisis Detection System, Covenant University 2025
"""

        msg = Message(
            subject=subject,
            recipients=[Config.ALERT_RECIPIENT_EMAIL],
            body=body,
            sender=Config.MAIL_DEFAULT_SENDER or Config.MAIL_USERNAME
        )

        mail.send(msg)
        
        # Update alert.is_emailed = True if email sent successfully
        alert.is_emailed = True
        db.session.commit()
        print(f"Email sent successfully for alert {alert.alert_id}")

    except Exception as e:
        # If email fails (SMTP error): log the error but do not crash the app
        logging.error(f"SMTP Error while sending alert email: {str(e)}")
        print(f"SMTP Error: {str(e)}")
        # We don't re-raise to avoid crashing the background task

def acknowledge_alert(alert_id, user):
    """
    Acknowledge an alert (admin only).
    """
    from models import Alert
    alert = Alert.query.get_or_404(alert_id)
    if user.role != 'admin':
        raise PermissionError("Only admins can acknowledge alerts")

    alert.acknowledged = True
    db.session.commit()
    return alert

def get_active_alerts():
    """
    Get all unacknowledged alerts.
    """
    from models import Alert
    return Alert.query.filter_by(acknowledged=False).order_by(Alert.created_at.desc()).all()