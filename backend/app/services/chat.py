from app.core.database import get_chats_collection
from app.services.document_processor import DocumentProcessor
from app.services.llm_service import LLMService
from app.models.chat import QueryResponse
from bson import ObjectId
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

_loaded_processors = {}
_loaded_llms = {}

# ---------------- REGION DETECTOR ----------------
def detect_region(query: str) -> str:
    q = query.lower()
    if "us" in q or "america" in q or "usa" in q or "united states" in q:
        return "us"
    elif "india" in q or "indian" in q:
        return "india"
    return "india"

# ---------------- GLOBAL CACHE ----------------
def get_document_processor(region: str):
    global _loaded_processors
    if region not in _loaded_processors:
        logger.info(f"Initializing document processor for region: {region.upper()}")
        dp = DocumentProcessor(region=region)
        dp.initialize_vector_store()
        _loaded_processors[region] = dp
    return _loaded_processors[region]

def get_llm_service(region: str):
    global _loaded_llms
    if region not in _loaded_llms:
        logger.info(f"Initializing LLM service for region: {region.upper()}")
        _loaded_llms[region] = LLMService(document_processor=get_document_processor(region))
    return _loaded_llms[region]

# ---------------- UTIL ----------------
def is_valid_object_id(id_string):
    if not id_string:
        return False
    try:
        ObjectId(id_string)
        return True
    except:
        return False

# ---------------- CHAT SERVICE ----------------
class ChatService:
    def __init__(self):
        self.chats_collection = get_chats_collection()

    def process_query(self, user_id: str, chat_id: str, query: str) -> QueryResponse:
        try:
            region = detect_region(query)
            logger.info(f"Detected region for query: {region.upper()}")

            document_processor = get_document_processor(region)
            llm_service = get_llm_service(region)

            if not document_processor or not document_processor.is_initialized():
                return QueryResponse(
                    response=f"{region.upper()} system is initializing. Please try again shortly.",
                    source_document="system",
                    confidence=0.0,
                    chat_id=chat_id
                )

            # ---------------- SEARCH ----------------
            search_results, search_metadata = document_processor.search_similar(query)
            version_context = document_processor.get_version_context()

            response_text = llm_service.generate_response(
                query=query,
                context=search_results,
                version_context=version_context,
                search_results_metadata=search_metadata
            )

            confidence, source_document = 0.0, "No relevant documents"
            if search_results:
                confidence = float(search_results[0][1])
                source_document = (
                    search_metadata[0].get("document_name", "Multiple documents")
                    if search_metadata else "Multiple documents"
                )

            message_data = {
                "role": "user",
                "content": query,
                "timestamp": datetime.utcnow(),
                "region": region
            }
            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.utcnow(),
                "source_document": source_document,
                "confidence": confidence,
                "region": region
            }

            # ---------------- SAVE TO CHAT ----------------
            if chat_id and chat_id not in ["null", "undefined"]:
                try:
                    query_filter = (
                        {"_id": ObjectId(chat_id), "user_id": ObjectId(user_id)}
                        if is_valid_object_id(chat_id)
                        else {"_id": chat_id, "user_id": ObjectId(user_id)}
                    )
                    result = self.chats_collection.update_one(
                        query_filter,
                        {
                            "$push": {"messages": {"$each": [message_data, assistant_message]}},
                            "$set": {"updated_at": datetime.utcnow(), "region": region}
                        }
                    )
                    if result.matched_count == 0:
                        chat_id = self._create_new_chat(user_id, chat_id, query, message_data, assistant_message)
                except Exception as e:
                    logger.warning(f"Error updating existing chat {chat_id}: {e}")
                    chat_id = self._create_new_chat(user_id, None, query, message_data, assistant_message)
            else:
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

    # ---------------- HELPERS ----------------
    def _create_new_chat(self, user_id: str, chat_id: str, query: str, user_message: dict, assistant_message: dict) -> str:
        chat_data = {
            "user_id": ObjectId(user_id),
            "title": query[:40] + ("..." if len(query) > 40 else ""),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "messages": [user_message, assistant_message],
            "region": user_message.get("region", "india")
        }
        if chat_id and not is_valid_object_id(chat_id):
            chat_data["_id"] = chat_id
            self.chats_collection.insert_one(chat_data)
            return chat_id
        else:
            result = self.chats_collection.insert_one(chat_data)
            return str(result.inserted_id)

    def get_chat_history(self, user_id: str):
        try:
            chats = self.chats_collection.find({"user_id": ObjectId(user_id)}).sort("updated_at", -1)
            return [
                {
                    "_id": str(chat["_id"]),
                    "title": chat.get("title", "Untitled Chat"),
                    "region": chat.get("region", "india"),
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
        try:
            query = (
                {"_id": ObjectId(chat_id), "user_id": ObjectId(user_id)}
                if is_valid_object_id(chat_id)
                else {"_id": chat_id, "user_id": ObjectId(user_id)}
            )
            chat = self.chats_collection.find_one(query)
            if chat:
                return {
                    "_id": str(chat["_id"]),
                    "title": chat.get("title", "Untitled Chat"),
                    "region": chat.get("region", "india"),
                    "created_at": chat["created_at"],
                    "updated_at": chat["updated_at"],
                    "messages": chat.get("messages", [])
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching chat: {e}")
            return None

    def delete_chat(self, user_id: str, chat_id: str):
        try:
            query = (
                {"_id": ObjectId(chat_id), "user_id": ObjectId(user_id)}
                if is_valid_object_id(chat_id)
                else {"_id": chat_id, "user_id": ObjectId(user_id)}
            )
            result = self.chats_collection.delete_one(query)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting chat: {e}")
            return False

    def clear_all_chats(self, user_id: str):
        try:
            result = self.chats_collection.delete_many({"user_id": ObjectId(user_id)})
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error clearing chats: {e}")
            return 0

    def get_system_status(self):
        india_dp = _loaded_processors.get("india")
        us_dp = _loaded_processors.get("us")

        india_status = india_dp.get_status() if india_dp else {"initialized": False}
        us_status = us_dp.get_status() if us_dp else {"initialized": False}

        return {
            "india": india_status,
            "us": us_status,
            "overall": "Operational" if (india_status["initialized"] or us_status["initialized"]) else "Initializing"
        }
