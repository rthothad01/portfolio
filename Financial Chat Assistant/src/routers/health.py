"""Health check router"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from src.app_state import get_app_state

from pydantic import BaseModel
from datetime import datetime, timezone

import logging

logger = logging.getLogger(__name__)

# Create a router for document processor endpoints
router = APIRouter(
            prefix="",  # No prefix, keep /health as-is
            tags=["Health, Status & Config"])

@router.get("/ping")
async def ping():
    """
    Basic health check
    
    Returns server status and current time
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "financial-chat-assistant"
    }

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns API status and configuration state
    """
    try:
        app_state = get_app_state()
        config = app_state.get("config")
        query_engine = app_state.get("query_engine")
        
        return {
            "status": "healthy",
            "index_loaded": query_engine is not None,
            "config_loaded": config is not None,
            "model": config.llm_model if config else None,
            "embedding_model": config.embedding_model,
            "directories_setup": config.data_dir.exists(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except RuntimeError as e:
        # Config not initialized
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e),
            "config_loaded": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Unexpected error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config-info")
async def get_config_info():
    """
    Get configuration information
    
    Returns the current configuration summary
    """
    
    try:
        app_state = get_app_state()
        config = app_state.get("config")
        
        if not config:
            raise HTTPException(
            status_code=500,
            detail="Configuration not loaded"
        )

        summary = config.get_summary()
        
        return {
                "status": "success",
                "config": summary,
                "api_keys_loaded": bool(config.openai_api_key and config.llamaparse_api_key),
                "directories": {
                    "data_dir": {
                        "path": str(config.data_dir),
                        "exists": config.data_dir.exists()
                    },
                    "images_dir": {
                        "path": str(config.images_dir),
                        "exists": config.images_dir.exists()
                    },
                    "storage_dir": {
                        "path": str(config.storage_dir),
                        "exists": config.storage_dir.exists()
                    },
                    "cache_dir": {
                        "path": str(config.cache_dir),
                        "exists": config.cache_dir.exists()
                    }
                },
                "model_settings": {
                    "llm_model": config.llm_model,
                    "embedding_model": config.embedding_model,
                    "temperature": config.temperature,
                    "similarity_top_k": config.similarity_top_k,
                    "response_mode": config.response_mode,
                },
                "llamaparse_settings": {
                    "result_type": config.result_type,
                    "gpt4o_mode": config.gpt4o_mode,
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Error getting config info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    



    