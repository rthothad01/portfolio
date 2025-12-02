"""Application state management

This module manages the global application configuration.
Separated from main.py to avoid circular imports.
"""

import logging
from typing import Optional
from .config import Config
from .indexing import IndexBuilder
from .query_engine import QueryEngineBuilder

logger = logging.getLogger(__name__)

# Global state
_app_state = {
    "config": None,
    "query_engine": None,
    "index_builder": None,
    "query_builder": None,
}

def initialize_config() -> Config:
    """
    Initialize the application configuration
    
    Called once during application startup.
    
    Returns:
        Config: The initialized configuration
    """
    logger.info("📋 Loading configuration...")
    
    config = Config()
    config.setup_directories()

     # Store in global state
    _app_state["config"] = config
    
    logger.info("✅ Configuration loaded successfully!")
    logger.info(f"   - LLM Model: {config.llm_model}")
    logger.info(f"   - Embedding Model: {config.embedding_model}")
    logger.info(f"   - Data Directory: {config.data_dir}")
    logger.info(f"   - OpenAI API Key: {'*' * 20}{config.openai_api_key[-4:]}")
    logger.info(f"   - LlamaParse API Key: {'*' * 20}{config.llamaparse_api_key[-4:]}")
    
    return config


def get_config() -> Config:
    """
    Get the global configuration instance
    
    Returns:
        Config: The application configuration
        
    Raises:
        RuntimeError: If configuration not initialized
    """
    config = _app_state.get("config")
    if config is None:
        raise RuntimeError("Configuration not initialized")
    return config

def get_app_state():
    """
    Get the global application state dictionary
    
    Returns:
        dict: Application state
    """
    return _app_state

def set_query_engine(query_engine, index_builder, query_builder):
    """
    Set query engine and related components in global state
    
    Args:
        query_engine: Query engine instance
        index_builder: Index builder instance
        query_builder: Query builder instance
    """
    _app_state["query_engine"] = query_engine
    _app_state["index_builder"] = index_builder
    _app_state["query_builder"] = query_builder
    
    logger.info("✓ Query engine components stored in global state")