from app.core.database import get_chats_collection
from app.services.document_processor import DocumentProcessor
from app.services.llm_service import LLMService
from app.models.chat import QueryResponse
from bson import ObjectId
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)

_document_processor = None
_llm_service = None

def get_document_processor():
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
        _document_processor.initialize_vector_store()
    return _document_processor

def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(document_processor=get_document_processor())
    return _llm_service

def is_valid_object_id(id_string):
    """Check if string is a valid MongoDB ObjectId"""
    if not id_string:
        return False
    try:
        ObjectId(id_string)
        return True
    except:
        return False

class ChatService:
    def __init__(self):
        self.chats_collection = get_chats_collection()
        self.document_processor = get_document_processor()
        self.llm_service = get_llm_service()

    def process_query(self, user_id: str, chat_id: str, query: str) -> QueryResponse:
        """Process query within a continuous chat context with version awareness"""
        if not self.document_processor or not self.document_processor.is_initialized():
            return QueryResponse(
                response="System not ready. Please try again shortly.",
                source_document="system",
                confidence=0.0,
                chat_id=chat_id
            )

        try:
            # Search for similar documents with metadata
            search_results, search_metadata = self.document_processor.search_similar(query)
            
            # Get version context
            version_context = self.document_processor.get_version_context()
            
            # Generate response with version awareness
            response_text = self.llm_service.generate_response(
                query=query,
                context=search_results,
                version_context=version_context,
                search_results_metadata=search_metadata
            )

            # Calculate confidence score (convert to Python float)
            confidence = 0.0
            source_document = "No relevant documents"
            
            if search_results and len(search_results) > 0:
                # Use the highest confidence score from search results
                confidence = float(search_results[0][1]) if len(search_results[0]) > 1 else 0.0
                
                # Get the primary source document from metadata
                if search_metadata and len(search_metadata) > 0:
                    source_document = search_metadata[0].get('document_name', 'Multiple documents')
                else:
                    source_document = "Multiple documents"

            # Save message to existing chat or create new one
            message_data = {
                "role": "user",
                "content": query,
                "timestamp": datetime.utcnow()
            }
            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.utcnow(),
                "source_document": source_document,
                "confidence": confidence
            }

            # Handle chat ID - support both ObjectId and custom string IDs
            if chat_id and chat_id != "null" and chat_id != "undefined":
                try:
                    # Try to find existing chat with the given ID
                    if is_valid_object_id(chat_id):
                        # It's a MongoDB ObjectId
                        query_filter = {"_id": ObjectId(chat_id), "user_id": ObjectId(user_id)}
                    else:
                        # It's a custom string ID (like 'chat-1762410876054')
                        query_filter = {"_id": chat_id, "user_id": ObjectId(user_id)}
                    
                    # Append to existing chat
                    result = self.chats_collection.update_one(
                        query_filter,
                        {
                            "$push": {"messages": {"$each": [message_data, assistant_message]}},
                            "$set": {"updated_at": datetime.utcnow()}
                        }
                    )
                    
                    if result.matched_count == 0:
                        # Chat not found, create new one with the provided chat_id
                        logger.info(f"Chat {chat_id} not found, creating new chat with this ID")
                        chat_id = self._create_new_chat(user_id, chat_id, query, message_data, assistant_message)
                    else:
                        logger.info(f"Successfully updated existing chat: {chat_id}")
                        
                except Exception as e:
                    logger.warning(f"Error updating existing chat {chat_id}, creating new one: {e}")
                    # Create new chat but don't reuse the problematic chat_id
                    chat_id = self._create_new_chat(user_id, None, query, message_data, assistant_message)
            else:
                # Create a new chat
                chat_id = self._create_new_chat(user_id, None, query, message_data, assistant_message)

            return QueryResponse(
                response=response_text,
                source_document=source_document,
                confidence=confidence,
                chat_id=str(chat_id)
            )
            
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            return QueryResponse(
                response="I'm having trouble responding. Please try again later.",
                source_document="system",
                confidence=0.0,
                chat_id=chat_id
            )

    def _create_new_chat(self, user_id: str, chat_id: str, query: str, user_message: dict, assistant_message: dict) -> str:
        """Helper method to create a new chat"""
        chat_data = {
            "user_id": ObjectId(user_id),
            "title": query[:40] + ("..." if len(query) > 40 else ""),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "messages": [user_message, assistant_message]
        }
        
        # If a custom chat_id is provided, use it as _id
        if chat_id and not is_valid_object_id(chat_id):
            chat_data["_id"] = chat_id
            result = self.chats_collection.insert_one(chat_data)
            return chat_id
        else:
            # Let MongoDB generate an ObjectId
            result = self.chats_collection.insert_one(chat_data)
            return str(result.inserted_id)

    def get_chat_history(self, user_id: str):
        """Fetch all conversations for user"""
        try:
            chats = self.chats_collection.find({"user_id": ObjectId(user_id)}).sort("updated_at", -1)
            return [
                {
                    "_id": str(chat["_id"]),  # Convert both ObjectId and string IDs to string
                    "title": chat.get("title", "Untitled Chat"),
                    "created_at": chat["created_at"],
                    "updated_at": chat["updated_at"],
                    "messages": chat.get("messages", [])
                }
                for chat in chats
            ]
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
            return []

    def get_chat(self, user_id: str, chat_id: str):
        """Get specific chat by ID"""
        try:
            # Build query based on ID type
            if is_valid_object_id(chat_id):
                query = {"_id": ObjectId(chat_id), "user_id": ObjectId(user_id)}
            else:
                query = {"_id": chat_id, "user_id": ObjectId(user_id)}
                
            chat = self.chats_collection.find_one(query)
            if chat:
                return {
                    "_id": str(chat["_id"]),
                    "title": chat.get("title", "Untitled Chat"),
                    "created_at": chat["created_at"],
                    "updated_at": chat["updated_at"],
                    "messages": chat.get("messages", [])
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching chat: {e}")
            return None

    def delete_chat(self, user_id: str, chat_id: str):
        """Delete specific chat"""
        try:
            # Build query based on ID type
            if is_valid_object_id(chat_id):
                query = {"_id": ObjectId(chat_id), "user_id": ObjectId(user_id)}
            else:
                query = {"_id": chat_id, "user_id": ObjectId(user_id)}
                
            result = self.chats_collection.delete_one(query)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting chat: {e}")
            return False

    def clear_all_chats(self, user_id: str):
        """Clear all user chats"""
        try:
            result = self.chats_collection.delete_many({"user_id": ObjectId(user_id)})
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error clearing chats: {e}")
            return 0

    def get_system_status(self):
        """Get system status information"""
        if not self.document_processor:
            return {
                "status": "Document processor not initialized",
                "initialized": False,
                "document_count": 0,
                "chunk_count": 0
            }
        
        status = self.document_processor.get_status()
        return {
            "status": "Operational" if status["initialized"] else "Initializing",
            "initialized": status["initialized"],
            "document_count": status["document_count"],
            "chunk_count": status["documents_loaded"]
        }