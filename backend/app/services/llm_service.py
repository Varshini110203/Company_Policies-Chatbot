import logging
from typing import List, Tuple, Dict, Any
import requests
import warnings
import urllib3
from app.core.config import settings

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, document_processor=None):
        logger.info("Initializing Groq LLM Service")
        self.api_url = settings.GROQ_API_URL
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL if hasattr(settings, 'GROQ_MODEL') else "llama-3.1-8b-instant"
        self.processor = document_processor
        
    def _extract_text_from_context(self, context_item) -> str:
        """Extract text content from context item"""
        if isinstance(context_item, str):
            return context_item
        elif isinstance(context_item, dict):
            for key in ['content', 'text', 'document', 'page_content']:
                if key in context_item:
                    text = context_item[key]
                    if isinstance(text, str):
                        return text
            return str(context_item)
        else:
            return str(context_item)
        
    def _build_prompt(self, query: str, context: List[Tuple], version_context: str = "", search_results_metadata: List[Dict] = None) -> str:
        """Build prompt emphasizing document versions and recency"""
        
        # Extract context content with version info
        context_texts = []
        version_mentions = []
        
        for i, doc in enumerate(context[:5]):
            if isinstance(doc, tuple) and len(doc) > 0:
                content = self._extract_text_from_context(doc[0])
                similarity = doc[1] if len(doc) > 1 else 0.0
                
                # Add version source information if available
                source_info = ""
                if search_results_metadata and i < len(search_results_metadata):
                    metadata = search_results_metadata[i]
                    doc_name = metadata.get('document_name', 'Unknown')
                    modified_date = metadata.get('modified_date', 'Unknown date')
                    is_recent = metadata.get('is_most_recent', False)
                    
                    if is_recent:
                        source_info = f"\n[SOURCE: {doc_name} - LATEST VERSION - Modified: {modified_date}]"
                    else:
                        source_info = f"\n[SOURCE: {doc_name} - OLDER VERSION - Modified: {modified_date}]"
                    
                    version_mentions.append(f"- {doc_name} (Modified: {modified_date})")
                
                context_texts.append(f"Content {i+1} (Relevance: {similarity:.2f}):{source_info}\n{content}")
            else:
                content = self._extract_text_from_context(doc)
                context_texts.append(f"Content: {content}")
        
        context_str = "\n\n" + "\n\n".join(context_texts)
        
        # Build version summary
        version_summary = ""
        if version_mentions:
            version_summary = "Documents found in search:\n" + "\n".join(version_mentions)
        
        prompt = f"""You are a helpful policy assistant. Based EXCLUSIVELY on the following policy context, please answer the user's question in a clear, concise manner.

USER QUESTION: {query}

{version_summary}

RELEVANT CONTENT FROM DOCUMENTS:{context_str}

VERSION CONTEXT: {version_context}

CRITICAL INSTRUCTIONS:
1. FIRST identify if the user is asking about:
   - Current/latest policies (use most recent documents)
   - Previous/older versions (use older documents)
   - Version comparison (compare across documents)
   - Available versions (list all document versions)

2. FOR VERSION-SPECIFIC QUERIES:
   - If asking "what are the versions available for X", list ALL versions found in the documents with their dates
   - If asking "current/latest X policy", use ONLY the most recent document
   - If asking "previous version of X", use OLDER documents and specify they are from previous versions
   - If asking about changes/differences, compare across versions and highlight what changed

3. Instructions
   - Start with direct answer to the question
   - Specify which version(s) you're referencing
   - Provide ONE clear, comprehensive answer that best addresses the question from the latest and most relevant documents
   - Use natural, conversational language - bullet points or numbered lists
   - Do NOT mention sources, page numbers, or use [SOURCE] markers in your response
   - If multiple versions exist, mention the differences
   - If information conflicts between versions, note this and prioritize recent versions
   - Be transparent about version sources
   - Do not give relevance scores and confidence levels in the final answer

4. ANSWER FORMAT:
- Start with direct answer to the question
- Brief context if needed.
- If information conflicts between versions, note this and prioritize recent versions

5. If information is missing from documents, say so and specify which aspects are unavailable."""

        return prompt

    def generate_response(self, query: str, context: List[Tuple], version_context: str = "", search_results_metadata: List[Dict] = None) -> str:
        """Generate response with version awareness"""
        
        if not context:
            return "I don't have specific information about this in our HR documents. Please contact HR for detailed assistance."
        
        # Build enhanced prompt with version info
        prompt = self._build_prompt(query, context, version_context, search_results_metadata)
        
        # Call Groq API
        response = self._call_groq_api(prompt)
        
        logger.info("Generated version-aware response")
        return response

    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API to generate response"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False
            }
            
            logger.info(f"Calling Groq API with prompt: {len(prompt)} chars")
            
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload, 
                timeout=60,
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content'].strip()
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
        
        # Check for version-specific queries first
        version_keywords = ['version', 'previous', 'old', 'earlier', 'past', 'current', 'latest', 'new', 'change', 'difference']
        if any(keyword in query_lower for keyword in version_keywords):
            if 'what version' in query_lower or 'available version' in query_lower:
                return "list_versions"
            elif 'previous' in query_lower or 'old' in query_lower or 'earlier' in query_lower:
                return "previous_version"
            elif 'current' in query_lower or 'latest' in query_lower or 'new' in query_lower:
                return "current_version"
            elif 'compare' in query_lower or 'difference' in query_lower or 'change' in query_lower:
                return "version_comparison"
        
        categories = {
            "leave": ['leave', 'vacation', 'holiday', 'sick', 'time off', 'pto'],
            "policy": ['policy', 'rule', 'guideline', 'procedure'],
            "remote work": ['remote', 'work from home', 'wfh', 'telecommute'],
            "benefits": ['benefit', 'insurance', 'health', 'retirement', '401k'],
            "compensation": ['salary', 'pay', 'compensation', 'bonus'],
            "onboarding": ['onboard', 'training', 'new employee', 'orientation'],
            "dress code": ['dress', 'code', 'attire', 'clothing', 'uniform']
        }
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return "general"