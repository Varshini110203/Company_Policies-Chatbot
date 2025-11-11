from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api.routes import api_router
from app.services.document_processor import DocumentProcessor
from app.core.database import connect_to_mongo, close_mongo_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store all region processors globally
region_processors = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up HR Assistant API...")

    # ---------------- DATABASE CONNECTION ----------------
    if connect_to_mongo():
        logger.info("✅ MongoDB connected successfully")
    else:
        logger.error("❌ MongoDB connection failed")

    # ---------------- VECTOR STORE INITIALIZATION ----------------
    global region_processors
    region_processors = {}

    for region in ["india", "us"]:
        try:
            dp = DocumentProcessor(region=region)
            dp.initialize_vector_store()
            region_processors[region] = dp

            if dp.is_initialized():
                status = dp.get_status()
                logger.info(
                    f"✅ {region.upper()} vector store ready "
                    f"({status['document_count']} documents, {status['documents_loaded']} chunks)"
                )
            else:
                logger.warning(f"⚠️ {region.upper()} vector store not fully initialized.")
        except Exception as e:
            logger.error(f"❌ Error initializing {region.upper()} vector store: {str(e)}")

    logger.info("📊 Vector store initialization complete for all regions.")

    yield

    # ---------------- SHUTDOWN ----------------
    logger.info("🧹 Shutting down HR Assistant API...")
    close_mongo_connection()
    logger.info("✅ MongoDB connection closed.")


# ---------------- FASTAPI APP INSTANCE ----------------
app = FastAPI(
    title="HR Assistant API",
    description="AI-powered HR Assistant Chat Application",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------- CORS MIDDLEWARE ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ROUTES ----------------
app.include_router(api_router, prefix="/api/v1")


# ---------------- HEALTH ENDPOINTS ----------------
@app.get("/")
async def root():
    return {"message": "HR Assistant API is running"}


@app.get("/health")
async def health_check():
    statuses = {}
    for region, dp in region_processors.items():
        statuses[region] = dp.get_status() if dp else {"initialized": False}
    return {
        "status": "healthy"
        if any(dp.is_initialized() for dp in region_processors.values())
        else "degraded",
        "regions": statuses,
    }


@app.get("/status")
async def status_check():
    region_status = {
        region: dp.get_status() if dp else {"initialized": False}
        for region, dp in region_processors.items()
    }

    return {
        "api": "running",
        "database": "connected",
        "vector_stores": region_status,
    }
