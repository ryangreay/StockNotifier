from telegram import Bot, Update
from telegram.ext import CallbackContext
from datetime import datetime, timedelta
from uuid import uuid4
from src.config import TELEGRAM_BOT_TOKEN

class TelegramBot:
    def __init__(self):
        self.bot = Bot(TELEGRAM_BOT_TOKEN)
        self.pending_connections = {}  # Store temporary tokens

    async def handle_start(self, update: Update, context: CallbackContext):
        """Handle /start command - generate connection token."""
        # Generate unique token
        token = str(uuid4())[:8]  # Short, unique token
        chat_id = update.effective_chat.id
        
        # Store token with expiry (15 minutes)
        self.pending_connections[token] = {
            'chat_id': chat_id,
            'expires_at': datetime.now() + timedelta(minutes=15)
        }
        
        await update.message.reply_text(
            f"Your connection code is: {token}\n"
            f"Enter this code on the website to connect your account.\n"
            "This code will expire in 15 minutes."
        )