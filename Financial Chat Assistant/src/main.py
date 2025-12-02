"""FastAPI application for Financial Chat Assistant"""

import logging, sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from llama_index.core import StorageContext, load_index_from_storage
import uvicorn

from .app_state import initialize_config, get_app_state, set_query_engine

from src import Config, DocumentProcessor, DocumentUtils, IndexBuilder
from src import QueryEngineBuilder, QueryRequest, QueryResponse, ReportOutput

from .routers import (
    health_check_router,
    document_router,
    query_router
)

# Setup logging
# Remove all existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
     handlers=[
        logging.StreamHandler(sys.stdout)  # Output to console
    ])

logger = logging.getLogger(__name__)
# Test it immediately
logger.info("=" * 60)
print("🔧 Logging configured successfully! - from print statement")
logger.info("🔧 Logging configured successfully!")
logger.info("=" * 60)

# Get reference to Global state
app_state = get_app_state()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI app
    Initializes resources on startup and cleans up on shutdown
    """
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 Starting Financial Chat Assistant API")
    logger.info("=" * 60)
    
    try:
        # ✅ Initialize config using app_state module
        config = initialize_config()
        
        # Check if index exists
        if config.storage_dir.exists():
            logger.info("📚 Loading existing index...")
            
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(config.storage_dir)
                )
                index = load_index_from_storage(
                    storage_context,
                    index_id="summary_index"
                )
                
                # Create query components
                query_builder = QueryEngineBuilder(config)
                query_engine = query_builder.create_query_engine(index)
                index_builder = IndexBuilder(config)
                
                # ✅ Store using helper function
                set_query_engine(query_engine, index_builder, query_builder)
                
                logger.info("✓ Index loaded successfully")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to load index: {e}")
        else:
            logger.warning("⚠️  No index found")
            logger.info("→ Use /document/process to process a document")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}", exc_info=True)
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("👋 Shutting down")
    logger.info("=" * 60)

app = FastAPI(
    title="Financial Chat Assistant API",
    description="Multimodal RAG system for querying financial documents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #  Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_check_router)
app.include_router(document_router)
app.include_router(query_router)

logger.info("✓ Routers registered")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with welcome page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Chat Assistant API</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; margin-top: 30px; }
            .endpoint { 
                background: #f8f9fa; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }
            .method { 
                color: #27ae60; 
                font-weight: bold;
                font-family: monospace;
            }
            code { 
                background: #e8e8e8; 
                padding: 2px 8px; 
                border-radius: 3px;
                font-family: monospace;
            }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .status { color: #27ae60; font-weight: bold; }
            .section { margin-bottom: 30px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏦 Financial Chat Assistant API</h1>
            <p class="status">✅ API is running!</p>
            <p>Multimodal RAG system for querying financial documents</p>
            
            <div class="section">
                <h2>🏥 Health & Status Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/health</code>
                    <p>Check API health status and configuration</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/config-info</code>
                    <p>Get current configuration information</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/status</code>
                    <p>Get detailed system status</p>
                </div>
            </div>
            
            <div class="section">
                <h2>📄 Document Processor Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/processor/hello</code>
                    <p>Hello World from Document Processor 🚀</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/processor/config-info</code>
                    <p>Get configuration info from processor</p>
                </div>
            </div>
            
            <div class="section">
                <h2>📚 Documentation</h2>
                <p>🔗 <a href="/docs">Interactive API Documentation (Swagger UI)</a></p>
                <p>🔗 <a href="/redoc">Alternative Documentation (ReDoc)</a></p>
            </div>
            
            <div class="section">
                <h2>🧪 Quick Test</h2>
                <p>Try these endpoints:</p>
                <ul>
                    <li><a href="/health">/health</a> - Basic health check</li>
                    <li><a href="/status">/status</a> - Detailed system status</li>
                    <li><a href="/processor/hello">/processor/hello</a> - Hello World</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    start_server(reload=True)