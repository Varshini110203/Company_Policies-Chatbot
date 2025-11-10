from app.core.database import get_users_collection
from app.core.security import verify_password, get_password_hash, create_access_token
from app.models.user import UserCreate, User
from bson import ObjectId
import logging
from datetime import datetime, timedelta
from app.core.config import settings

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self):
        self.users_collection = get_users_collection()

    def register_user(self, user_data: UserCreate) -> User:
        """Register a new user (synchronous)"""
        if self.users_collection is None:
            raise ValueError("Database not connected")
            
        # Check if user already exists
        if self.users_collection.find_one({"email": user_data.email}):
            raise ValueError("User with this email already exists")
        
        if self.users_collection.find_one({"username": user_data.username}):
            raise ValueError("User with this username already exists")
        
        # Create new user
        user_dict = {
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": get_password_hash(user_data.password),
            "created_at": ObjectId().generation_time
        }
        
        result = self.users_collection.insert_one(user_dict)
        
        return User(
            id=str(result.inserted_id),
            username=user_data.username,
            email=user_data.email,
            created_at=user_dict["created_at"]
        )

    def authenticate_user(self, username: str, password: str):
        """Authenticate user (synchronous)"""
        if self.users_collection is None:
            return None
            
        # Try to find user by username OR email
        user_data = self.users_collection.find_one({
            "$or": [
                {"username": username},
                {"email": username}
            ]
        })
        if not user_data:
            return None
        
        if not verify_password(password, user_data["password_hash"]):
            return None
        
        return User(
            id=str(user_data["_id"]),
            username=user_data["username"],
            email=user_data["email"],
            created_at=user_data["created_at"]
        )

    def get_user_by_id(self, user_id: str):
        """Get user by ID (synchronous)"""
        if self.users_collection is None:
            return None
            
        try:
            user_data = self.users_collection.find_one({"_id": ObjectId(user_id)})
            if user_data:
                return User(
                    id=str(user_data["_id"]),
                    username=user_data["username"],
                    email=user_data["email"],
                    created_at=user_data["created_at"]
                )
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
        return None

    def get_user_by_email(self, email: str):
        """Get user by email (synchronous)"""
        if self.users_collection is None:
            return None
            
        try:
            user_data = self.users_collection.find_one({"email": email})
            if user_data:
                return User(
                    id=str(user_data["_id"]),
                    username=user_data["username"],
                    email=user_data["email"],
                    created_at=user_data["created_at"]
                )
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            return None
        return None

    def get_user_by_username(self, username: str):
        """Get user by username (synchronous)"""
        if self.users_collection is None:
            return None
            
        try:
            user_data = self.users_collection.find_one({"username": username})
            if user_data:
                return User(
                    id=str(user_data["_id"]),
                    username=user_data["username"],
                    email=user_data["email"],
                    created_at=user_data["created_at"]
                )
        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None
        return None

    def initiate_password_reset(self, email: str) -> bool:
        """Initiate password reset process"""
        try:
            # Check if user exists
            user = self.get_user_by_email(email)
            if user:
                # In development, just log the reset request
                logger.info(f"Password reset requested for: {email}")
                # For now, we'll just return True without sending actual email
                # In production, you would:
                # 1. Generate a reset token
                # 2. Store it in database with expiration
                # 3. Send email with reset link
                print(f"DEBUG: Password reset would be sent to: {email}")
            
            # Always return True for security (don't reveal if email exists)
            return True
        except Exception as e:
            logger.error(f"Error initiating password reset: {str(e)}")
            # Still return True for security
            return True

    def reset_password(self, token: str, new_password: str) -> bool:
        """Reset password using token"""
        try:
            # For development, accept any token that starts with "dev_"
            if token.startswith("dev_"):
                # Extract email from dev token (format: "dev_reset_token_email@example.com")
                email = token.replace("dev_reset_token_", "")
                
                user = self.get_user_by_email(email)
                if not user:
                    return False
                
                # Update user's password
                if self.users_collection is None:
                    return False
                
                result = self.users_collection.update_one(
                    {"email": email},
                    {"$set": {"password_hash": get_password_hash(new_password)}}
                )
                
                if result.modified_count > 0:
                    logger.info(f"Password reset successfully for: {email}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error resetting password: {str(e)}")
            return False