from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class Database:
    client: MongoClient = None
    db = None

db = Database()

def connect_to_mongo():
    """Connect to MongoDB (synchronous)"""
    try:
        db.client = MongoClient(settings.MONGODB_URL)
        db.db = db.client[settings.DATABASE_NAME]
        # Verify connection
        db.client.admin.command('ping')
        logger.info("Connected to MongoDB successfully")
        return True
    except ConnectionFailure as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        return False

def close_mongo_connection():
    """Close MongoDB connection (synchronous)"""
    if db.client is not None:
        db.client.close()
        logger.info("MongoDB connection closed")

def get_database():
    return db.db

def get_users_collection():
    if db.db is not None:
        return db.db.users
    return None

def get_chats_collection():
    if db.db is not None:
        return db.db.chats
    return None
