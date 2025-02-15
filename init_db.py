from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.exc import SQLAlchemyError
from src.models import UserTelegramConnection, PendingTelegramConnection
from src.config import DATABASE_URL

def init_telegram_tables():
    """Initialize only the Telegram-related tables."""
    try:
        # Print database URL (with password masked)
        db_url_parts = DATABASE_URL.split('@')
        if len(db_url_parts) > 1:
            masked_url = f"{db_url_parts[0].split(':')[0]}:***@{db_url_parts[1]}"
        else:
            masked_url = DATABASE_URL
        print(f"Using database URL: {masked_url}")
        
        print(f"Connecting to database...")
        engine = create_engine(DATABASE_URL)
        
        # Test connection
        with engine.connect() as conn:
            print("Database connection successful!")
        
        # Create MetaData instance
        metadata = MetaData()
        
        # Create only the Telegram tables
        print("\nCreating Telegram tables...")
        UserTelegramConnection.__table__.create(bind=engine, checkfirst=True)
        PendingTelegramConnection.__table__.create(bind=engine, checkfirst=True)
        
        # Verify tables were created
        inspector = inspect(engine)
        table_names = ['user_telegram_connections', 'pending_telegram_connections']
        print("\nCreated tables:")
        for table in table_names:
            if table in inspector.get_table_names():
                print(f"\n  Table: {table}")
                # Print columns for each table
                columns = inspector.get_columns(table)
                for column in columns:
                    print(f"    - {column['name']}: {column['type']}")
                
                # Print foreign keys
                foreign_keys = inspector.get_foreign_keys(table)
                if foreign_keys:
                    print(f"    Foreign Keys:")
                    for fk in foreign_keys:
                        print(f"      - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
            else:
                print(f"\n  Warning: Table {table} was not created!")
        
        print("\nTelegram tables initialization complete!")
        
    except SQLAlchemyError as e:
        print(f"\nDatabase error: {str(e)}")
        raise
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting Telegram tables initialization...")
    init_telegram_tables() 