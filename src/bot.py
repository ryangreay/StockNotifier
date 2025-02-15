from telegram.ext import Application, CommandHandler
from telegram import Update
import asyncio
from src.telegram import TelegramBot
from src.config import TELEGRAM_BOT_TOKEN
import logging
import signal

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def main():
    """Run the bot."""
    try:
        # Create the Application
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Create TelegramBot instance
        bot = TelegramBot()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", bot.handle_start))
        
        logger.info("Starting Telegram bot...")
        
        # Run the bot until stopped
        application.run_polling(drop_pending_updates=True)
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {str(e)}")

if __name__ == "__main__":
    main() 