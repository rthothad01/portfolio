"""FastAPI application for Financial Chat Assistant"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from .app_state import initialize_config

from .health_check import router as health_check_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global configuration
app_config = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI app
    Loads configuration on startup
    """
    global app_config
    
    # Startup: Load configuration
    logger.info("🚀 Starting Financial Chat Assistant API")
    
    try:
        initialize_config()
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Financial Chat Assistant API")

app = FastAPI(
    title="Financial Chat Assistant API",
    description="Multimodal RAG system for querying financial documents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_check_router)      # ← NEW: Health check endpoints


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

