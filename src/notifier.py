import asyncio
from telegram import Bot
from src.config import TELEGRAM_BOT_TOKEN
from src.models import UserTelegramConnection, UserSettings

class StockNotifier:
    def __init__(self):
        if not TELEGRAM_BOT_TOKEN:
            raise ValueError("Missing Telegram credentials in environment variables")
        
        self.bot = Bot(TELEGRAM_BOT_TOKEN)
    
    async def send_message(self, chat_id, text):
        """Send message via Telegram."""
        await self.bot.send_message(chat_id=chat_id, text=text, parse_mode='HTML')
    
    async def send_notification(self, user_id, symbol, prediction, probability, current_price, db):
        """Send notification about predicted stock movement."""
        # Get user's telegram connection
        telegram_conn = db.query(UserTelegramConnection).filter(
            UserTelegramConnection.user_id == user_id,
            UserTelegramConnection.is_active == True
        ).first()
        
        if not telegram_conn:
            return False, "No active Telegram connection found"
            
        # Get user's settings for prediction window
        user_settings = db.query(UserSettings).filter(
            UserSettings.user_id == user_id
        ).first()
        
        if not user_settings:
            return False, "User settings not found"
        
        # Format the time window based on the prediction window
        if user_settings.training_timeframe == '1H':
            time_text = f"{user_settings.prediction_window} hours"
        elif user_settings.training_timeframe == '1D':
            time_text = f"{user_settings.prediction_window // 24} days"
        elif user_settings.training_timeframe == '1W':
            time_text = f"{user_settings.prediction_window // (24 * 7)} weeks"
        elif user_settings.training_timeframe == '1M':
            time_text = f"{user_settings.prediction_window // (24 * 30)} months"
        else:
            time_text = f"{user_settings.prediction_window} hours"  # fallback
        
        movement_type = "upward" if prediction == 1 else "downward"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        message = (
            f"🚨 <b>Stock Alert: {symbol}</b> 📈\n\n"
            f"Significant {movement_type} movement predicted in the next {time_text}\n"
            f"Confidence: {confidence:.2%}\n"
            f"Current Price: ${current_price:.2f}"
        )
        
        try:
            await self.send_message(telegram_conn.telegram_chat_id, message)
            return True, "Message sent"
        except Exception as e:
            return False, str(e)
    
    async def send_error_notification(self, user_id, symbol, error_message, db):
        """Send notification about system errors."""
        # Get user's telegram connection
        telegram_conn = db.query(UserTelegramConnection).filter(
            UserTelegramConnection.user_id == user_id,
            UserTelegramConnection.is_active == True
        ).first()
        
        if not telegram_conn:
            return False, "No active Telegram connection found"
            
        message = (
            f"⚠️ <b>System Alert: {symbol}</b>\n\n"
            f"Error monitoring stock:\n{error_message}"
        )
        
        try:
            await self.send_message(telegram_conn.telegram_chat_id, message)
            return True, "Error notification sent"
        except Exception as e:
            return False, str(e) 