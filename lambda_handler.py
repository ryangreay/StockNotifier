import os
import json
import pytz
from datetime import datetime
import redis
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database import get_db, engine
from src import models

# Initialize Redis connection
REDIS_URL = os.getenv('REDIS_URL')
redis_client = redis.from_url(REDIS_URL)

def get_market_event_type():
    """Determine the current market event type based on time."""
    eastern = pytz.timezone('America/New_York')
    now = datetime.now(eastern)
    
    # Market hours in ET
    market_open = now.replace(hour=9, minute=30)
    midday = now.replace(hour=12, minute=0)
    market_close = now.replace(hour=16, minute=0)
    
    if abs((now - market_open).total_seconds()) < 300:  # Within 5 minutes of market open
        return 'market_open'
    elif abs((now - midday).total_seconds()) < 300:  # Within 5 minutes of midday
        return 'midday'
    elif abs((now - market_close).total_seconds()) < 300:  # Within 5 minutes of market close
        return 'market_close'
    else:
        return None

def get_users_for_notification(db: Session, event_type: str):
    """Get users who should be notified for the current market event."""
    weekday = datetime.now().weekday()  # 0 = Monday, 6 = Sunday
    
    # Query users based on their notification preferences
    query = db.query(models.User, models.UserSettings, models.UserStock).join(
        models.UserSettings,
        models.User.id == models.UserSettings.user_id
    ).join(
        models.UserStock,
        models.User.id == models.UserStock.user_id
    ).filter(
        models.User.is_active == True,
        models.UserStock.enabled == True
    )
    
    # Add event-specific filters
    if event_type == 'market_open':
        query = query.filter(models.UserSettings.notify_market_open == True)
    elif event_type == 'midday':
        query = query.filter(models.UserSettings.notify_midday == True)
    elif event_type == 'market_close':
        query = query.filter(models.UserSettings.notify_market_close == True)
    
    # Check notification days using bit operations
    query = query.filter(
        func.get_bit(models.UserSettings.notification_days, weekday) == 1
    )

    return query.all()

def publish_to_redis(users_data):
    """Publish user data to Redis pub/sub channel."""
    channel = 'market_notifications'
    print(f"Starting Redis publish process. Redis URL: {REDIS_URL.replace(REDIS_URL.split('@')[0], '***')}")
    
    try:
        # Group stocks by user
        user_stocks = {}
        for user, settings, stock in users_data:
            print(f"Processing user {user.id} with stock {stock.symbol}")
            if user.id not in user_stocks:
                user_stocks[user.id] = {
                    'user_id': user.id,
                    'stocks': [],
                    'settings': {
                        'prediction_threshold': settings.prediction_threshold,
                        'significant_movement_threshold': settings.significant_movement_threshold,
                        'prediction_window': settings.prediction_window,
                        'training_timeframe': settings.training_timeframe
                    }
                }
            user_stocks[user.id]['stocks'].append(stock.symbol)
        
        print(f"Processed user stocks data: {json.dumps(user_stocks, indent=2)}")
        
        # Publish each user's data
        for user_id, user_data in user_stocks.items():
            message = json.dumps(user_data)
            print(f"Publishing to channel '{channel}' for user {user_id}: {message}")
            result = redis_client.publish(channel, message)
            print(f"Publish result for user {user_id}: {result} subscribers received the message")
            
        print("Redis publish process completed successfully")
        
    except redis.RedisError as e:
        print(f"Redis error occurred: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error in publish_to_redis: {str(e)}")
        raise

def lambda_handler(event, context):
    """AWS Lambda handler function."""
    db = None
    try:
        # Get current market event type
        event_type = get_market_event_type()
        print(f"Current market event type: {event_type}")
        
        if not event_type:
            return {
                'statusCode': 200,
                'body': json.dumps('No market event at this time')
            }
        
        # Get DB session
        db = next(get_db())
        
        # Get users to notify
        users_data = get_users_for_notification(db, event_type)
        print(f"Found {len(users_data)} users to notify")
        
        if not users_data:
            print("No users found for notification")
            return {
                'statusCode': 200,
                'body': json.dumps('No users to notify')
            }
        
        # Publish to Redis
        publish_to_redis(users_data)
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully processed {event_type} event')
        }
        
    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
    finally:
        if db is not None:
            db.close() 