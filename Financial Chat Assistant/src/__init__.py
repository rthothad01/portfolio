"""Financial Chat Assistant - Source Package"""

#expose config module at the package level
from .config import Config
from .models import (
    QueryRequest,
    QueryResponse,
    ReportOutput,
    TextBlock,
    ImageBlock,
    SYSTEM_PROMPT
)
from .document_processor import DocumentProcessor
from .indexing import DocumentUtils, IndexBuilder
from .query_engine import QueryEngineBuilder

__version__ = "1.0.0"
__all__ = [
            "Config"
            "QueryRequest",
            "QueryResponse",
            "ReportOutput",
            "TextBlock",
            "ImageBlock",
            "SYSTEM_PROMPT",
            "DocumentProcessor",
            "DocumentUtils",
            "IndexBuilder",
            "QueryEngineBuilder"
            ]