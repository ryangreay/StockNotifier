import os
import json
import redis
import asyncio
import aiohttp
from datetime import datetime, timedelta
from jose import jwt

# Initialize Redis connection
REDIS_URL = os.getenv('REDIS_URL')
API_BASE_URL = os.getenv('API_BASE_URL', '').rstrip('/')

# Ensure HTTPS
if not API_BASE_URL.startswith('http'):
    API_BASE_URL = f"https://{API_BASE_URL}"

JWT_SECRET = os.getenv('JWT_SECRET')
JWT_ALGORITHM = 'HS256'

print(f"Initializing consumer with API URL: {API_BASE_URL}")

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

async def call_predict_api(session: aiohttp.ClientSession, user_id: int, symbol: str, notify: bool = True):
    """Call the prediction API endpoint."""
    headers = {
        "Authorization": f"Bearer {create_system_jwt(user_id)}",
        "Content-Type": "application/json"
    }
    
    url = f"{API_BASE_URL}/predict"
    print(f"Calling API endpoint: {url}")
    
    try:
        async with session.post(
            url,
            headers=headers,
            json={"symbol": symbol, "notify": notify},
            timeout=30
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"API error (status {response.status}): {error_text}")
                return {"error": error_text}
            return await response.json()
    except aiohttp.ClientError as e:
        print(f"Network error calling API: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error calling API: {str(e)}")
        raise

async def process_prediction(user_data: dict, session: aiohttp.ClientSession):
    """Process predictions for a user's stocks."""
    user_id = user_data['user_id']
    
    for symbol in user_data['stocks']:
        try:
            print(f"Processing prediction for user {user_id}, symbol {symbol}")
            response = await call_predict_api(
                session=session,
                user_id=user_id,
                symbol=symbol,
                notify=True  # The API will handle notification thresholds
            )
            print(f"Prediction response for {symbol}: {json.dumps(response, indent=2)}")
        except Exception as e:
            print(f"Error processing prediction for user {user_id}, symbol {symbol}: {str(e)}")
            continue  # Continue with next symbol even if one fails

async def process_message(message: dict):
    """Process a message from Redis pub/sub."""
    try:
        # Parse message data
        user_data = json.loads(message['data'])
        print(f"Processing message for user {user_data['user_id']}")
        
        # Create a single session for all API calls
        connector = aiohttp.TCPConnector(
            force_close=True,
            enable_cleanup_closed=True,
            ttl_dns_cache=300
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            await process_prediction(user_data, session)
            
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        print(f"Message data: {message.get('data', 'No data')}")

async def start_consumer():
    """Start the Redis pub/sub consumer."""
    pubsub = redis_client.pubsub()
    pubsub.subscribe('market_notifications')
    
    print(f"Starting Redis consumer, listening on channel: market_notifications")
    print(f"API endpoint: {API_BASE_URL}")
    
    while True:
        try:
            message = pubsub.get_message()
            if message and message['type'] == 'message':
                print("Received message from Redis")
                await process_message(message)
            await asyncio.sleep(0.1)  # Prevent busy-waiting
        except Exception as e:
            print(f"Consumer error: {str(e)}")
            await asyncio.sleep(1)  # Wait before retrying

if __name__ == "__main__":
    asyncio.run(start_consumer()) 