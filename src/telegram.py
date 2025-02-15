from telegram import Update
from telegram.ext import ContextTypes
from datetime import datetime, timedelta
import uuid
from src.database import SessionLocal
from src.models import PendingTelegramConnection
from sqlalchemy import delete

class TelegramBot:
    def __init__(self):
        """Initialize the TelegramBot."""
        self.db = SessionLocal()
        
        # Clean up expired pending connections
        self._cleanup_expired_connections()
    
    def _cleanup_expired_connections(self):
        """Remove expired pending connections."""
        try:
            self.db.execute(
                delete(PendingTelegramConnection).where(
                    PendingTelegramConnection.expires_at < datetime.now()
                )
            )
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(f"Error cleaning up expired connections: {str(e)}")
    
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        try:
            # Generate a short token from UUID
            token = str(uuid.uuid4())[:8]
            
            # Store the connection in the database
            pending_conn = PendingTelegramConnection(
                token=token,
                telegram_chat_id=str(update.effective_chat.id),
                expires_at=datetime.now() + timedelta(minutes=15)  # Token expires in 15 minutes
            )
            
            self.db.add(pending_conn)
            self.db.commit()
            
            # Send welcome message with connection token
            await update.message.reply_text(
                f"Welcome to the Stock Movement Predictor Bot! 📈\n\n"
                f"Your connection code is: <code>{token}</code>\n\n"
                f"Enter this code on the website to connect your account.\n"
                f"This code will expire in 15 minutes.",
                parse_mode='HTML'
            )
            
        except Exception as e:
            self.db.rollback()
            print(f"Error in handle_start: {str(e)}")
            await update.message.reply_text(
                "Sorry, there was an error processing your request. Please try again later."
            )
    
    def __del__(self):
        """Cleanup when the bot is destroyed."""
        self.db.close()