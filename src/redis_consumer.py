import os
import json
import redis
import asyncio
import aiohttp
from datetime import datetime, timedelta
from jose import jwt

# Initialize Redis connection
REDIS_URL = os.getenv('REDIS_URL')
API_BASE_URL = os.getenv('API_BASE_URL')
JWT_SECRET = os.getenv('JWT_SECRET')
JWT_ALGORITHM = 'HS256'

redis_client = redis.from_url(REDIS_URL)

def create_system_jwt(user_id: int) -> str:
    """Create a system JWT token for API calls."""
    expires_delta = timedelta(minutes=5)  # Short-lived token
    expire = datetime.utcnow() + expires_delta
    
    to_encode = {
        "sub": str(user_id),
        "exp": expire,
        "type": "system"  # Indicate this is a system-generated token
    }
    
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def call_predict_api(session, user_id: int, symbol: str, notify: bool = True):
    """Call the prediction API endpoint."""
    headers = {
        "Authorization": f"Bearer {create_system_jwt(user_id)}",
        "Content-Type": "application/json"
    }
    
    async with session.post(
        f"{API_BASE_URL}/predict",
        headers=headers,
        json={"symbol": symbol, "notify": notify}
    ) as response:
        return await response.json()

async def process_prediction(user_data, session):
    """Process predictions for a user's stocks."""
    user_id = user_data['user_id']
    settings = user_data['settings']
    
    for symbol in user_data['stocks']:
        try:
            # Call prediction API
            await call_predict_api(
                session=session,
                user_id=user_id,
                symbol=symbol,
                notify=True  # The API will handle notification thresholds
            )
        except Exception as e:
            print(f"Error processing prediction for user {user_id}, symbol {symbol}: {str(e)}")

async def process_message(message, session):
    """Process a message from Redis pub/sub."""
    try:
        # Parse message data
        user_data = json.loads(message['data'])
        
        # Process predictions for user
        await process_prediction(user_data, session)
            
    except Exception as e:
        print(f"Error processing message: {str(e)}")

async def start_consumer():
    """Start the Redis pub/sub consumer."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe('market_notifications')
    
    print("Starting Redis consumer...")
    
    # Create aiohttp session for API calls
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                message = pubsub.get_message()
                if message and message['type'] == 'message':
                    await process_message(message, session)
                await asyncio.sleep(0.1)  # Prevent busy-waiting
            except Exception as e:
                print(f"Consumer error: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying

if __name__ == "__main__":
    asyncio.run(start_consumer()) 