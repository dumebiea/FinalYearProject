#!/usr/bin/env python
"""Test seed_data function directly"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import create_app

app = create_app()
print("✓ App created and database seeded")

# Check database contents
with app.app_context():
    from models import User, Tweet, Prediction, SentimentSummary
    user_count = User.query.count()
    tweet_count = Tweet.query.count()
    pred_count = Prediction.query.count()
    summary_count = SentimentSummary.query.count()
    
    print(f"\n📊 Database Contents:")
    print(f"  Users: {user_count}")
    print(f"  Tweets: {tweet_count}")
    print(f"  Predictions: {pred_count}")
    print(f"  Summaries: {summary_count}")
    
    if summary_count > 0:
        print("\n✅ Seeding successful - charts should now display!")
    else:
        print("\n❌ No summaries found - seeding may have failed")
