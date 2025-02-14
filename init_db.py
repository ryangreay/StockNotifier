from sqlalchemy import create_engine, inspect, text, MetaData
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import DropTable
from sqlalchemy.ext.compiler import compiles
from src.models import Base, User, UserSettings, UserStock, RefreshToken, UserTelegramConnection
from src.config import DATABASE_URL

@compiles(DropTable, "postgresql")
def _compile_drop_table(element, compiler, **kwargs):
    return compiler.visit_drop_table(element) + " CASCADE"

def drop_tables(engine):
    """Drop all tables in the correct order."""
    print("Dropping existing tables...")
    
    # Create MetaData instance
    metadata = MetaData()
    metadata.reflect(bind=engine)
    
    # Drop all tables
    metadata.drop_all(bind=engine)
    print("All tables dropped successfully!")

def init_db():
    """Initialize database tables."""
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
        
        # Drop existing tables
        drop_tables(engine)
        
        # Create all tables
        print("\nCreating tables...")
        Base.metadata.create_all(bind=engine)
        
        # Verify tables were created
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        print("\nCreated tables:")
        for table in table_names:
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
        
        print("\nDatabase initialization complete!")
        
    except SQLAlchemyError as e:
        print(f"\nDatabase error: {str(e)}")
        raise
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting database initialization...")
    init_db() 