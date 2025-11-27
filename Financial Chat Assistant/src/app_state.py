"""Application state management

This module manages the global application configuration.
Separated from main.py to avoid circular imports.
"""

import logging
from typing import Optional
from .config import Config

logger = logging.getLogger(__name__)

# Global configuration instance
_app_config: Optional[Config] = None


def initialize_config() -> Config:
    """
    Initialize the application configuration
    
    Called once during application startup.
    
    Returns:
        Config: The initialized configuration
    """
    global _app_config
    
    logger.info("📋 Loading configuration...")
    
    _app_config = Config()
    _app_config.setup_directories()
    
    logger.info("✅ Configuration loaded successfully!")
    logger.info(f"   - LLM Model: {_app_config.llm_model}")
    logger.info(f"   - Embedding Model: {_app_config.embedding_model}")
    logger.info(f"   - Data Directory: {_app_config.data_dir}")
    logger.info(f"   - OpenAI API Key: {'*' * 20}{_app_config.openai_api_key[-4:]}")
    logger.info(f"   - LlamaParse API Key: {'*' * 20}{_app_config.llamaparse_api_key[-4:]}")
    
    return _app_config


def get_config() -> Config:
    """
    Get the global configuration instance
    
    Returns:
        Config: The application configuration
        
    Raises:
        RuntimeError: If configuration not initialized
    """
    global _app_config
    
    if _app_config is None:
        raise RuntimeError(
            "Configuration not initialized. "
            "App may not have started properly."
        )
    
    return _app_config


def is_config_initialized() -> bool:
    """
    Check if configuration has been initialized
    
    Returns:
        bool: True if config is initialized, False otherwise
    """
    return _app_config is not None