"""Document processing router"""

import logging, time
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File

# ✅ Import from app_state
from src.app_state import get_config, get_app_state, set_query_engine
from src.document_processor import DocumentProcessor
from src.indexing import DocumentUtils, IndexBuilder
from src.query_engine import QueryEngineBuilder
from src.models import DocumentProcessRequest, DocumentProcessResponse


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/document",
    tags=["Document Processing"]
)


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = None,  # 
    force_rebuild: bool = False,
    extract_images: bool = True
):
    """
    Process a PDF document
    
    Either:
    1. Upload a file (multipart/form-data)
    2. Provide a URL in JSON body (application/json)
    
    Args:
        request: JSON request with URL and options
        file: PDF file upload (alternative to URL)
        
    Returns:
        Processing status and statistics
    """
    try:
        start_time = time.time()
        
        # ✅ Get config from app_state module
        config = get_config()
        print(f"download_url is {url}")
        print(f"images_dir is {config.images_dir}")
        
        # Method 1: JSON body with URL
        if url:
            logger.info(f"Processing document from URL: {url}")
            
            # Determine filename from URL
            from urllib.parse import urlparse
            from pathlib import Path
            url_path = urlparse(url).path
            filename = Path(url_path).name or "downloaded.pdf"
            file_path = config.data_dir / filename
            
            # Process with URL
            processor = DocumentProcessor(config)
            
            result = processor.process_document(
                file_path=file_path,
                download_url=url,
                images_dir=config.images_dir
            )
            
            force_rebuild = force_rebuild
        # Method 2: File upload
        elif file:
            logger.info(f"Processing uploaded file: {file.filename}")
            
            if not file.filename.endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are supported"
                )
            
            # Save uploaded file
            file_path = config.data_dir / file.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            logger.info(f"Saved uploaded file to {file_path}")
            
            # Process uploaded file
            processor = DocumentProcessor(config)
            result = processor.process_document(
                file_path=file_path,
                download_url=None,
                images_dir=config.images_dir
            )
        # Neither provided
        else:
            raise HTTPException(
                status_code=400,
                detail="Either file upload or URL must be provided"
            )

        # Check if processing succeeded
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Failed to process document"
            )
        
        pages, images_dir = result
        
        # Create text nodes
        text_nodes = DocumentUtils.create_text_nodes(pages, images_dir)
        
        # Build index
        index_builder = IndexBuilder(config)
        index = index_builder.build_index(text_nodes, force_rebuild=force_rebuild)
        
        # Create query engine
        query_builder = QueryEngineBuilder(config)
        query_engine = query_builder.create_query_engine(index)
        
        # Update global state
        set_query_engine(query_engine, index_builder, query_builder)
        
        # Get statistics
        summary = processor.get_page_summary(pages)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return DocumentProcessResponse(
            status="success",
            message="Document processed successfully",
            summary=summary,
            nodes_created=len(text_nodes),
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))