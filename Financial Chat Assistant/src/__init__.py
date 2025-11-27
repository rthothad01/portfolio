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

__version__ = "1.0.0"
__all__ = [
            "Config"
            "QueryRequest",
            "QueryResponse",
            "ReportOutput",
            "TextBlock",
            "ImageBlock",
            "SYSTEM_PROMPT"
            ]