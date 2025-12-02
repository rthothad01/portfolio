"""Query router"""

import logging
import time

from fastapi import APIRouter, HTTPException

# ✅ Import from app_state
from src.app_state import get_config, get_app_state
from src.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["Query"]
)


@router.post("", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the processed document
    """
    try:
        # ✅ Get config and state
        config = get_config()
        app_state = get_app_state()
        
        query_builder = app_state.get("query_builder")
        
        if not query_builder:
            raise HTTPException(
                status_code=400,
                detail="No document processed. Use /document/process endpoint first."
            )
        
        # Execute query
        start_time = time.time()
        
        # Update query engine if parameters changed
        if request.similarity_top_k != config.similarity_top_k:
            logger.info(f"Updating similarity_top_k to {request.similarity_top_k}")
            index = app_state["index_builder"].get_index()
            if index is None:
                raise HTTPException(
                    status_code=400,
                    detail="No document processed. Use /document/process endpoint first."
                )
            query_builder.create_query_engine(
                index,
                similarity_top_k=request.similarity_top_k
            )
        
        response = query_builder.query(
                        request.query, 
                        verbose=request.verbose if hasattr(request, 'verbose') else True,
                        include_images=request.include_images,
                        max_image_size_mb=request.max_image_size_mb if hasattr(request, 'max_image_size_mb') else 5.0
                    )
        
        if not response:
            raise HTTPException(
                status_code=500,
                detail="Query execution failed"
            )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            query=request.query,
            report=response,
            source_pages=[],
            response_time_ms=response_time_ms,
            model_used=config.llm_model,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))