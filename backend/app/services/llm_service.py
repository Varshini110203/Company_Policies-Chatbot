import logging
from typing import List, Tuple, Dict, Any
import requests
import warnings
import urllib3
from app.core.config import settings

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, document_processor=None):
        logger.info("Initializing Groq LLM Service")
        self.api_url = settings.GROQ_API_URL
        self.api_key = settings.GROQ_API_KEY
        self.model = (
            settings.GROQ_MODEL
            if hasattr(settings, "GROQ_MODEL")
            else "llama-3.1-8b-instant"
        )
        self.processor = document_processor

    def _extract_text_from_context(self, context_item) -> str:
        """Extract text content from context item"""
        if isinstance(context_item, str):
            return context_item
        elif isinstance(context_item, dict):
            for key in ["content", "text", "document", "page_content"]:
                if key in context_item:
                    text = context_item[key]
                    if isinstance(text, str):
                        return text
            return str(context_item)
        else:
            return str(context_item)

    def _build_prompt(
        self,
        query: str,
        context: List[Tuple],
        version_context: str = "",
        search_results_metadata: List[Dict] = None,
    ) -> str:
        """Build optimized prompt with size limits"""
        
        # --- More aggressive limits ---
        max_chunks = 3  # Reduced from 8 to 3
        max_chunk_length = 800  # Reduced from 1000 to 800
        max_total_chars = 12000  # More conservative limit

        context_texts = []
        version_mentions = []

        # Sort by relevance and take top chunks
        sorted_context = sorted(context, key=lambda x: x[1] if len(x) > 1 else 0, reverse=True)
        
        for i, doc in enumerate(sorted_context[:max_chunks]):
            if isinstance(doc, tuple) and len(doc) > 0:
                content = self._extract_text_from_context(doc[0])
                
                # More aggressive truncation
                if len(content) > max_chunk_length:
                    # Try to keep the most relevant part
                    sentences = content.split('.')
                    compressed_content = []
                    current_length = 0
                    for sentence in sentences:
                        if current_length + len(sentence) < max_chunk_length:
                            compressed_content.append(sentence)
                            current_length += len(sentence)
                        else:
                            break
                    content = '. '.join(compressed_content) + '.'
                
                similarity = doc[1] if len(doc) > 1 else 0.0

                source_info = ""
                if search_results_metadata and i < len(search_results_metadata):
                    metadata = search_results_metadata[i]
                    doc_name = metadata.get("document_name", "Unknown")
                    modified_date = metadata.get("modified_date", "Unknown date")
                    is_recent = metadata.get("is_most_recent", False)

                    if is_recent:
                        source_info = f"\n[SOURCE: {doc_name} - LATEST - {modified_date}]"
                    else:
                        source_info = f"\n[SOURCE: {doc_name} - {modified_date}]"

                    version_mentions.append(f"- {doc_name} (Modified: {modified_date})")

                context_texts.append(
                    f"Content {i+1} (Relevance: {similarity:.2f}):{source_info}\n{content}"
                )

        context_str = "\n\n" + "\n\n".join(context_texts)

        # Simplify version summary
        version_summary = ""
        if version_mentions:
            version_summary = "Relevant documents:\n" + "\n".join(version_mentions[:3])  # Limit to 3

        # More concise prompt template
        prompt = f"""Answer the question using the document context below.

    QUESTION: {query}

    {version_summary}

    DOCUMENT CONTEXT:{context_str}

    INSTRUCTIONS:
    - Use only the provided document context
    - Be concise and accurate
    - If information is missing, say so clearly
    """

        # --- Smart truncation instead of blind cutting ---
        if len(prompt) > max_total_chars:
            logger.warning(f"⚠️ Prompt too large ({len(prompt)} chars). Optimizing...")
            
            # Remove less important sections first
            if len(context_str) > 6000:
                # Truncate least relevant chunks first
                context_str = self._truncate_context_smartly(context_str, max_total_chars - 2000)
            
            # Rebuild prompt with truncated context
            prompt = f"""Answer the question using the document context below.

    QUESTION: {query}

    DOCUMENT CONTEXT:{context_str}

    INSTRUCTIONS:
    - Use only the provided document context
    - Be concise and accurate
    - If information is missing, say so clearly
    """

        return prompt

    def _truncate_context_smartly(self, context_str: str, max_length: int) -> str:
        """Smart context truncation preserving most relevant parts"""
        chunks = context_str.split('\n\n')
        
        # Keep most relevant chunks (those with higher relevance scores)
        kept_chunks = []
        current_length = 0
        
        for chunk in chunks:
            chunk_length = len(chunk)
            if current_length + chunk_length <= max_length:
                kept_chunks.append(chunk)
                current_length += chunk_length
            else:
                # Try to truncate this chunk
                lines = chunk.split('\n')
                kept_lines = []
                for line in lines:
                    if current_length + len(line) <= max_length:
                        kept_lines.append(line)
                        current_length += len(line)
                    else:
                        break
                if kept_lines:
                    kept_chunks.append('\n'.join(kept_lines))
                break
        
        return '\n\n'.join(kept_chunks)
    def generate_response(
        self,
        query: str,
        context: List[Tuple],
        version_context: str = "",
        search_results_metadata: List[Dict] = None,
    ) -> str:
        """Generate response with version awareness"""

        if not context:
            return (
                "I don't have specific information about this in our HR documents. "
                "Please contact HR for detailed assistance."
            )

        prompt = self._build_prompt(query, context, version_context, search_results_metadata)

        response = self._call_groq_api(prompt)

        logger.info("Generated version-aware response")
        return response

    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API to generate response"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False,
            }

            logger.info(f"Calling Groq API with prompt: {len(prompt)} chars")

            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=60, verify=False
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"].strip()
                return response_text
            else:
                logger.error(f"Groq API error: {response.status_code}")
                return "I'm having trouble processing your request. Please try again."

        except requests.exceptions.Timeout:
            logger.error("Groq API timeout")
            return "Request timeout. Please try again."
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return "Service temporarily unavailable. Please try again later."

    def classify_query(self, query: str) -> str:
        """Enhanced query classification with version awareness"""
        query_lower = query.lower()

        version_keywords = [
            "version",
            "previous",
            "old",
            "earlier",
            "past",
            "current",
            "latest",
            "new",
            "change",
            "difference",
        ]
        if any(keyword in query_lower for keyword in version_keywords):
            if "what version" in query_lower or "available version" in query_lower:
                return "list_versions"
            elif any(k in query_lower for k in ["previous", "old", "earlier"]):
                return "previous_version"
            elif any(k in query_lower for k in ["current", "latest", "new"]):
                return "current_version"
            elif any(k in query_lower for k in ["compare", "difference", "change"]):
                return "version_comparison"

        categories = {
            "leave": ["leave", "vacation", "holiday", "sick", "time off", "pto"],
            "policy": ["policy", "rule", "guideline", "procedure"],
            "remote work": ["remote", "work from home", "wfh", "telecommute"],
            "benefits": ["benefit", "insurance", "health", "retirement", "401k"],
            "compensation": ["salary", "pay", "compensation", "bonus"],
            "onboarding": ["onboard", "training", "new employee", "orientation"],
            "dress code": ["dress", "code", "attire", "clothing", "uniform"],
        }

        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category

        return "general"
