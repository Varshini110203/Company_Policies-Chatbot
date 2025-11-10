from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api.routes import api_router
from app.services.document_processor import DocumentProcessor
from app.core.database import connect_to_mongo, close_mongo_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global document processor instance
doc_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up HR Assistant API...")
    
    # Connect to MongoDB (synchronous)
    if connect_to_mongo():
        logger.info("✅ MongoDB connected successfully")
    else:
        logger.error("❌ MongoDB connection failed")
    
    # Initialize document processor (synchronous)
    global doc_processor
    doc_processor = DocumentProcessor()
    
    try:
        doc_processor.initialize_vector_store()  # This is synchronous now
        if doc_processor.is_initialized():
            logger.info("✅ Document processor initialized successfully")
            status = doc_processor.get_status()
            logger.info(f"📊 Vector store status: {status['documents_loaded']} documents loaded")
        else:
            logger.error("❌ Document processor failed to initialize")
    except Exception as e:
        logger.error(f"❌ Error initializing document processor: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HR Assistant API...")
    close_mongo_connection()

app = FastAPI(
    title="HR Assistant API",
    description="AI-powered HR Assistant Chat Application",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    # allow_origins=settings.CORS_ORIGINS,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "HR Assistant API is running"}

@app.get("/health")
async def health_check():
    global doc_processor
    status = "healthy" if doc_processor and doc_processor.is_initialized() else "degraded"
    return {
        "status": status,
        "vector_store_initialized": doc_processor.is_initialized() if doc_processor else False,
        "documents_loaded": len(doc_processor.documents) if doc_processor and doc_processor.is_initialized() else 0
    }

@app.get("/status")
async def status_check():
    global doc_processor
    if doc_processor:
        processor_status = doc_processor.get_status()
        return {
            "api": "running",
            "database": "connected", 
            "vector_store": processor_status
        }
    else:
        return {
            "api": "running",
            "database": "connected",
            "vector_store": "not_initialized"
        }