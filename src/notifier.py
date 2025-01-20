import asyncio
from telegram import Bot
from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class StockNotifier:
    def __init__(self):
        if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
            raise ValueError("Missing Telegram credentials in environment variables")
        
        self.bot = Bot(TELEGRAM_BOT_TOKEN)
        self.chat_id = TELEGRAM_CHAT_ID
    
    async def send_message(self, text):
        """Send message via Telegram."""
        await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode='HTML')
    
    async def send_notification(self, symbol, prediction, probability, current_price):
        """Send notification about predicted stock movement."""
        movement_type = "upward" if prediction == 1 else "downward"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        message = (
            f"🚨 <b>Stock Alert: {symbol}</b> 📈\n\n"
            f"Significant {movement_type} movement predicted in the next 24 hours\n"
            f"Confidence: {confidence:.2%}\n"
            f"Current Price: ${current_price:.2f}"
        )
        
        try:
            await self.send_message(message)
            return True, "Message sent"
        except Exception as e:
            return False, str(e)
    
    async def send_error_notification(self, symbol, error_message):
        """Send notification about system errors."""
        message = (
            f"⚠️ <b>System Alert: {symbol}</b>\n\n"
            f"Error monitoring stock:\n{error_message}"
        )
        
        try:
            await self.send_message(message)
            return True, "Error notification sent"
        except Exception as e:
            return False, str(e) 