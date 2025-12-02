from src.routers.health import router as health_check_router
from src.routers.document import router as document_router
from src.routers.query import router as query_router

__all__ = [
    "health_check_router",
]