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
    def __init__(self, region: str = "india"):
        self.region = region.lower()
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None
        self.documents = []
        self.chunk_sources = []
        self.document_versions = {}
        self._initialized = False

    # ---------------- PATHS ----------------
    def _get_paths(self):
        base_docs = os.path.join(settings.DOCUMENTS_PATH, self.region)
        base_vector = os.path.join(settings.VECTOR_STORE_PATH, self.region)
        os.makedirs(base_vector, exist_ok=True)
        return base_docs, base_vector

    # ---------------- PDF SCAN ----------------
    def _get_all_pdfs(self, root_path: str) -> List[str]:
        pdf_files = []
        for dirpath, _, filenames in os.walk(root_path):
            for f in filenames:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(dirpath, f))
        return pdf_files

    # ---------------- PDF READER ----------------
    def extract_document_content(self, file_path: str) -> Dict[str, Any]:
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

    # ---------------- LOAD + SPLIT ----------------
    def load_documents(self) -> List[str]:
        documents_path, _ = self._get_paths()
        all_texts = []

        pdf_files = self._get_all_pdfs(documents_path)
        if not pdf_files:
            logger.warning(f"No PDFs found under {documents_path}")
            return []

        all_documents = []
        for file_path in pdf_files:
            doc_data = self.extract_document_content(file_path)
            if doc_data:
                all_documents.append(doc_data)
                logger.info(f"Extracted {doc_data['filename']} ({doc_data['page_count']} pages)")

        self.document_versions = {doc["filename"]: doc for doc in all_documents}

        for doc in all_documents:
            chunks = self.text_splitter.split_text(doc["content"])
            all_texts.extend(chunks)
            for chunk in chunks:
                self.chunk_sources.append(doc["filename"])
            logger.info(f"Split {doc['filename']} into {len(chunks)} chunks")

        logger.info(f"✅ Total {len(all_texts)} text chunks from {len(all_documents)} {self.region.upper()} documents")
        return all_texts

    # ---------------- EMBEDDINGS ----------------
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            logger.warning("No text to embed.")
            return np.array([])
        logger.info(f"Creating embeddings for {len(texts)} chunks ({self.region.upper()})")
        return self.embedding_model.encode(texts, show_progress_bar=False)

    # ---------------- BUILD VECTOR STORE ----------------
    def build_vector_store(self, texts: List[str], embeddings: np.ndarray):
        _, vector_path = self._get_paths()
        dim = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatL2(dim)
        self.vector_store.add(embeddings.astype(np.float32))
        self.documents = texts

        # Save FAISS + metadata
        faiss.write_index(self.vector_store, os.path.join(vector_path, "faiss.index"))
        with open(os.path.join(vector_path, "documents.pkl"), "wb") as f:
            pickle.dump(texts, f)
        with open(os.path.join(vector_path, "chunk_sources.pkl"), "wb") as f:
            pickle.dump(self.chunk_sources, f)
        with open(os.path.join(vector_path, "versions.pkl"), "wb") as f:
            pickle.dump(self.document_versions, f)

        self._initialized = True
        logger.info(f"✅ Built FAISS vector store for {self.region.upper()} with {len(texts)} chunks")

    # ---------------- LOAD EXISTING ----------------
    def load_vector_store(self) -> bool:
        _, vector_path = self._get_paths()
        index_path = os.path.join(vector_path, "faiss.index")
        docs_path = os.path.join(vector_path, "documents.pkl")
        chunk_sources_path = os.path.join(vector_path, "chunk_sources.pkl")
        versions_path = os.path.join(vector_path, "versions.pkl")

        if not (os.path.exists(index_path) and os.path.exists(docs_path)):
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
            logger.info(f"✅ Loaded existing FAISS vector store for {self.region.upper()}")
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS for {self.region.upper()}: {e}")
            return False

    # ---------------- INITIALIZE ----------------
    def initialize_vector_store(self):
        if self.load_vector_store():
            return
        texts = self.load_documents()
        if texts:
            embeddings = self.create_embeddings(texts)
            if embeddings.size > 0:
                self.build_vector_store(texts, embeddings)

    # ---------------- STATUS + CONTEXT ----------------
    def is_initialized(self) -> bool:
        return getattr(self, "_initialized", False)

    def get_status(self) -> dict:
        return {
            "initialized": getattr(self, "_initialized", False),
            "document_count": len(self.document_versions),
            "documents_loaded": len(self.documents)
        }

    def get_version_context(self) -> str:
        if not self.document_versions:
            return "No version context available."
        return "\n".join(
            f"{name} (Modified: {info.get('modified_date', 'Unknown')})"
            for name, info in self.document_versions.items()
        )

    # ---------------- SEARCH ----------------
    def search_similar(self, query: str, top_k: int = 5):
        """Search similar chunks using FAISS."""
        if not self._initialized or self.vector_store is None:
            raise ValueError(f"FAISS index not initialized for {self.region.upper()} region.")

        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.vector_store.search(query_embedding, top_k)
        distances, indices = distances[0], indices[0]

        results, metadata = [], []
        for idx, dist in zip(indices, distances):
            if 0 <= idx < len(self.documents):
                content = self.documents[idx]
                similarity = float(np.exp(-dist))  # Convert L2 distance to similarity
                results.append((content, similarity))
                filename = self.chunk_sources[idx] if idx < len(self.chunk_sources) else "Unknown"
                info = self.document_versions.get(filename, {})
                metadata.append({
                    "document_name": filename,
                    "modified_date": info.get("modified_date", "Unknown"),
                    "is_most_recent": True
                })
        return results, metadata
