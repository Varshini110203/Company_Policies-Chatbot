import os
import faiss
import numpy as np
import pickle
import hashlib
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None
        self.documents = []
        self.chunk_sources = []  # Track which document each chunk comes from
        self.document_versions = {}
        self._initialized = False

    def extract_document_content(self, file_path: str) -> Dict[str, Any]:
        """Extract content and metadata from a PDF file."""
        try:
            with open(file_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                content_hash = hashlib.md5(text.encode()).hexdigest()
                doc_title = os.path.splitext(os.path.basename(file_path))[0]
                modified_time = os.path.getmtime(file_path)
                created_time = os.path.getctime(file_path)

                return {
                    "file_path": file_path,
                    "filename": os.path.basename(file_path),
                    "title": doc_title,
                    "content": text,
                    "content_hash": content_hash,
                    "page_count": len(pdf.pages),
                    "file_size": os.path.getsize(file_path),
                    "modified_time": modified_time,
                    "created_time": created_time,
                    "modified_date": datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d"),
                    "created_date": datetime.fromtimestamp(created_time).strftime("%Y-%m-%d"),
                }
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return None

    def _vector_store_is_up_to_date(self) -> bool:
        """Check if existing vector store matches current document folder state."""
        index_path = os.path.join(settings.VECTOR_STORE_PATH, "faiss.index")
        versions_path = os.path.join(settings.VECTOR_STORE_PATH, "versions.pkl")
        chunk_sources_path = os.path.join(settings.VECTOR_STORE_PATH, "chunk_sources.pkl")
        documents_path = settings.DOCUMENTS_PATH

        if not os.path.exists(index_path) or not os.path.exists(versions_path) or not os.path.exists(chunk_sources_path):
            return False

        try:
            with open(versions_path, "rb") as f:
                stored_versions = pickle.load(f)

            # Current folder PDFs
            current_files = {
                f: os.path.getmtime(os.path.join(documents_path, f))
                for f in os.listdir(documents_path)
                if f.endswith(".pdf")
            }

            if not current_files:
                return False

            stored_files = {
                info["filename"]: info["modified_time"] for info in stored_versions.values()
            }

            # Added or removed PDFs
            if set(current_files.keys()) != set(stored_files.keys()):
                return False

            # Modified PDFs
            for fname, mtime in current_files.items():
                if abs(mtime - stored_files.get(fname, 0)) > 1:
                    return False

            return True

        except Exception as e:
            logger.warning(f"Vector store freshness check failed: {str(e)}")
            return False

    def load_documents(self) -> List[str]:
        """Load and split PDF documents from the documents directory."""
        documents_path = settings.DOCUMENTS_PATH
        all_texts = []

        os.makedirs(documents_path, exist_ok=True)
        pdf_files = [f for f in os.listdir(documents_path) if f.endswith(".pdf")]

        if not pdf_files:
            logger.warning(f"No PDF files found in {documents_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")

        all_documents = []
        for filename in pdf_files:
            file_path = os.path.join(documents_path, filename)
            doc_data = self.extract_document_content(file_path)
            if doc_data:
                all_documents.append(doc_data)
                logger.info(f"Extracted {filename} ({doc_data['page_count']} pages)")

        if not all_documents:
            return []

        # Save version info
        self.document_versions = {
            doc["filename"]: doc for doc in all_documents
        }

        # Split documents and track sources
        for doc in all_documents:
            chunks = self.text_splitter.split_text(doc["content"])
            all_texts.extend(chunks)
            # Track which document each chunk comes from
            for chunk in chunks:
                self.chunk_sources.append(doc["filename"])
            logger.info(f"Split {doc['filename']} into {len(chunks)} chunks")

        logger.info(f"Total {len(all_texts)} text chunks from {len(all_documents)} documents")
        return all_texts

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            logger.warning("No texts to create embeddings for")
            return np.array([])

        logger.info(f"Creating embeddings for {len(texts)} chunks")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        logger.info(f"Embeddings created: {embeddings.shape}")
        return embeddings

    def build_vector_store(self, texts: List[str], embeddings: np.ndarray):
        if len(texts) == 0 or embeddings.shape[0] == 0:
            logger.warning("No texts or embeddings to build vector store")
            return

        dim = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatL2(dim)
        self.vector_store.add(embeddings.astype(np.float32))
        self.documents = texts

        # Store chunk sources
        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
        faiss.write_index(self.vector_store, os.path.join(settings.VECTOR_STORE_PATH, "faiss.index"))

        with open(os.path.join(settings.VECTOR_STORE_PATH, "documents.pkl"), "wb") as f:
            pickle.dump(texts, f)
            
        with open(os.path.join(settings.VECTOR_STORE_PATH, "chunk_sources.pkl"), "wb") as f:
            pickle.dump(self.chunk_sources, f)

        with open(os.path.join(settings.VECTOR_STORE_PATH, "versions.pkl"), "wb") as f:
            pickle.dump(self.document_versions, f)

        self._initialized = True
        logger.info(f"✅ Vector store built with {len(texts)} chunks")

    def load_vector_store(self) -> bool:
        index_path = os.path.join(settings.VECTOR_STORE_PATH, "faiss.index")
        docs_path = os.path.join(settings.VECTOR_STORE_PATH, "documents.pkl")
        chunk_sources_path = os.path.join(settings.VECTOR_STORE_PATH, "chunk_sources.pkl")
        versions_path = os.path.join(settings.VECTOR_STORE_PATH, "versions.pkl")

        if not (os.path.exists(index_path) and os.path.exists(docs_path) and os.path.exists(chunk_sources_path)):
            return False

        try:
            self.vector_store = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            with open(chunk_sources_path, "rb") as f:
                self.chunk_sources = pickle.load(f)
            if os.path.exists(versions_path):
                with open(versions_path, "rb") as f:
                    self.document_versions = pickle.load(f)
            self._initialized = True
            logger.info("✅ Loaded existing FAISS vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return False

    def initialize_vector_store(self):
        logger.info("Initializing vector store...")

        if self._vector_store_is_up_to_date():
            if self.load_vector_store():
                logger.info("✅ Vector store is up-to-date — using existing files.")
                return

        logger.info("⚙️ Rebuilding vector store due to new or modified documents...")
        texts = self.load_documents()
        if texts:
            embeddings = self.create_embeddings(texts)
            if embeddings.size > 0:
                self.build_vector_store(texts, embeddings)
                logger.info("✅ Vector store rebuilt successfully")
            else:
                logger.error("Embedding generation failed.")
        else:
            logger.warning("No documents found for vector store rebuild.")
            self._initialized = False

    def search_similar(self, query: str, k: int = 5) -> Tuple[List[Tuple[str, float]], List[Dict]]:
        """
        Search for similar documents with metadata about source documents
        Returns: (results, metadata) where results are (content, confidence_score) tuples
        """
        if not self._initialized or self.vector_store is None or not self.documents:
            logger.error("Vector store not ready for search.")
            return [], []

        try:
            query_vec = self.embedding_model.encode([query])
            distances, indices = self.vector_store.search(query_vec.astype(np.float32), k*2)
            
            # Get the most recent document
            recent_doc_name = None
            if self.document_versions:
                recent_doc = max(self.document_versions.values(), key=lambda x: x["modified_time"])
                recent_doc_name = recent_doc["filename"]
            
            # Process all results
            all_results = []
            all_metadata = []
            
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.documents):
                    # Convert numpy float to Python float
                    similarity = float(1.0 / (1.0 + distances[0][i]))
                    content = self.documents[idx]
                    all_results.append((content, similarity))
                    
                    # Get document metadata for this chunk
                    doc_name = self.chunk_sources[idx] if idx < len(self.chunk_sources) else "Unknown"
                    doc_info = self.document_versions.get(doc_name, {})
                    
                    metadata = {
                        'document_name': doc_name,
                        'modified_date': doc_info.get('modified_date', 'Unknown'),
                        'is_most_recent': doc_name == recent_doc_name,
                        'similarity_score': similarity  # Already Python float
                    }
                    all_metadata.append(metadata)
            
            # Separate recent and older results
            recent_results = []
            recent_metadata = []
            other_results = []
            other_metadata = []
            
            for result, metadata in zip(all_results, all_metadata):
                if metadata['is_most_recent']:
                    recent_results.append(result)
                    recent_metadata.append(metadata)
                else:
                    other_results.append(result)
                    other_metadata.append(metadata)
            
            # Prioritize recent document results but include some older ones for comparison
            final_results = []
            final_metadata = []
            
            # Add recent results first
            if recent_results:
                final_results.extend(recent_results[:min(3, len(recent_results))])
                final_metadata.extend(recent_metadata[:min(3, len(recent_metadata))])
            
            # Add older results for context
            remaining_slots = k - len(final_results)
            if remaining_slots > 0 and other_results:
                final_results.extend(other_results[:remaining_slots])
                final_metadata.extend(other_metadata[:remaining_slots])
            
            return final_results, final_metadata
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return [], []

    def get_status(self) -> dict:
        return {
            "initialized": self._initialized,
            "documents_loaded": len(self.documents),
            "vector_store_ready": self.vector_store is not None,
            "document_count": len(self.document_versions),
        }

    def is_initialized(self) -> bool:
        return self._initialized

    def get_version_context(self) -> str:
        """Get a detailed description of document versions for the LLM"""
        if not self.document_versions:
            return "No documents found in the system."

        # Sort documents by modification time (newest first)
        sorted_docs = sorted(self.document_versions.values(), 
                           key=lambda x: x["modified_time"], 
                           reverse=True)
        
        version_info = []
        for i, doc in enumerate(sorted_docs):
            recency_indicator = " (MOST RECENT)" if i == 0 else " (OLDER VERSION)"
            version_info.append(
                f"• {doc['filename']}{recency_indicator}\n"
                f"  - Modified: {doc.get('modified_date', 'Unknown')}\n"
                f"  - Pages: {doc.get('page_count', 'Unknown')}\n"
                f"  - Size: {doc.get('file_size', 0) / 1024:.1f} KB"
            )
        
        summary = "\n\n".join(version_info)
        
        return (
            f"DOCUMENT VERSION CONTEXT:\n"
            f"Total documents: {len(sorted_docs)}\n"
            f"Documents sorted by modification date (newest first):\n\n"
            f"{summary}"
        )

    def get_all_versions_for_topic(self, topic: str) -> List[Dict]:
        """Get all document versions that contain information about a specific topic"""
        if not self._initialized:
            return []
        
        # Search for the topic across all documents
        results, metadata = self.search_similar(topic, k=10)
        
        # Group results by document
        doc_results = {}
        for result, meta in zip(results, metadata):
            doc_name = meta['document_name']
            if doc_name not in doc_results:
                doc_info = self.document_versions.get(doc_name, {})
                doc_results[doc_name] = {
                    'document_name': doc_name,
                    'modified_date': doc_info.get('modified_date', 'Unknown'),
                    'is_most_recent': meta['is_most_recent'],
                    'content_snippets': [],
                    'max_similarity': 0.0
                }
            
            content, similarity = result
            doc_results[doc_name]['content_snippets'].append(content)
            doc_results[doc_name]['max_similarity'] = max(doc_results[doc_name]['max_similarity'], similarity)
        
        return list(doc_results.values())