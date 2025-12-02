from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Union, Optional, Literal
import logging, base64

logger = logging.getLogger(__name__)

class TextBlock(BaseModel):
    """
    Text block containing analytical content
    
    Used for textual analysis, insights, and explanations in reports.
    """
    type: Literal["text"] = Field(
        default="text",
        description="Block type identifier"
    )
    content: str = Field(
        ..., 
        description="The text content for this block",
        min_length=1
    )
    
    def __str__(self) -> str:
        return self.content[:100] + ("..." if len(self.content) > 100 else "")
    
class ImageBlock(BaseModel):
    """
    Image content block with base64 encoding support
    
    Represents an image that supports the query response, typically a chart,
    table, or other visual element from the source document.
    """
    type: Literal["image"] = Field(
        default="image",
        description="Block type identifier"
    )
    file_path: str = Field(
        ..., 
        description="File path to the image (relative or absolute)",
        examples=["data_images/page-5.jpg", "data_images/chart-revenue-page-12.jpg"]
    )
    base64_data: Optional[str] = Field(
        default=None,
        description="Base64 encoded image data for inline display",
        examples=["iVBORw0KGgoAAAANSUhEUgAAAAUA..."]
    )
    mime_type: Optional[str] = Field(
        default="image/jpeg",
        description="MIME type of the image",
        examples=["image/jpeg", "image/png", "image/gif"]
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number in the source document where this image was found",
        ge=1,
        examples=[5, 12, 23]
    )
    
    def __str__(self) -> str:
        return f"Image({self.file_path})"
    
    def exists(self) -> bool:
        """Check if the image file exists"""
        return Path(self.file_path).exists()
    
    def load_base64(self) -> bool:
        """
        Load image from file system and encode as base64
        
        This method reads the image file and populates the base64_data field.
        It automatically detects the MIME type based on file extension.
        
        Returns:
            bool: True if successful, False if file not found or encoding failed
        """
        try:
            file_path = Path(self.file_path)
            resolved = None
            try:
                resolved = file_path.resolve()
            except Exception:
                resolved = file_path

            logger.debug(f"Attempting to load image. path={file_path}, resolved={resolved}")

            if not file_path.exists():
                logger.warning(f"Image file not found: {resolved}")
                return False

            # Check for zero-byte files early
            try:
                size = file_path.stat().st_size
            except Exception as e:
                logger.warning(f"Unable to stat image file {resolved}: {e}")
                size = None

            if size == 0:
                logger.warning(f"Image file is zero bytes: {resolved}")
                return False
            
            # Read image file
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            
            # Encode to base64
            self.base64_data = base64.b64encode(image_bytes).decode("utf-8")
            
            # Set MIME type based on extension
            ext = file_path.suffix.lower()
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".svg": "image/svg+xml"
            }
            self.mime_type = mime_types.get(ext, "image/jpeg")
            
            # Extract page number from filename if present
            if self.page_number is None:
                import re
                match = re.search(r'page-?(\d+)', file_path.stem, re.IGNORECASE)
                if match:
                    self.page_number = int(match.group(1))
            
            logger.info(f"Loaded image as base64: {resolved} ({len(self.base64_data)} chars)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load image {self.file_path}: {e}", exc_info=True)
            return False
    
    def get_data_url(self) -> Optional[str]:
        """
        Get data URL for embedding in HTML/JSON
        
        Creates a data URL that can be used directly in HTML img tags or
        displayed in web interfaces.
        
        Returns:
            str: Data URL in format "data:image/jpeg;base64,..." or None if loading failed
            
        Example:
            >>> block = ImageBlock(file_path="data_images/chart.jpg")
            >>> url = block.get_data_url()
            >>> # Use in HTML: <img src="{url}" />
        """
        if not self.base64_data:
            if not self.load_base64():
                return None
        
        return f"data:{self.mime_type};base64,{self.base64_data}"

    def get_file_size(self) -> Optional[int]:
        """
        Get file size in bytes
        
        Returns:
            int: File size in bytes, or None if file doesn't exist
        """
        try:
            return Path(self.file_path).stat().st_size
        except Exception:
            return None

class ReportOutput(BaseModel):
    """
    Report output containing text and image blocks
    
    Represents the complete response from a query, containing a mix of
    text explanations and supporting images (charts, tables, etc.)
    """
    blocks: List[Union[TextBlock, ImageBlock]] = Field(
        default_factory=list,
        description="List of content blocks (text and images) in the report",
        examples=[[
            {"type": "text", "content": "Revenue increased by 20%"},
            {"type": "image", "file_path": "data_images/page-5.jpg"}
        ]]
    )

    def validate_blocks(self) -> bool:
        """Validate that report has at least one image block"""
        has_image = any(isinstance(b, ImageBlock) for b in self.blocks)
        if not has_image:
            logger.warning("Report has no image blocks")
        return has_image
    
    def get_stats(self) -> dict:
        """Get statistics about the report"""
        text_blocks = [b for b in self.blocks if isinstance(b, TextBlock)]
        image_blocks = [b for b in self.blocks if isinstance(b, ImageBlock)]
        
        return {
            "total_blocks": len(self.blocks),
            "text_blocks": len(text_blocks),
            "image_blocks": len(image_blocks),
            "has_images": len(image_blocks) > 0,
            "total_text_length": sum(len(b.content) for b in text_blocks),
            "image_paths": [b.file_path for b in image_blocks]
        }
    
    def load_all_images(self, max_size_mb: float = 5.0) -> dict:
        """
        Load all images as base64
        
        Args:
            max_size_mb: Maximum size per image in MB (default: 5MB)
            
        Returns:
            dict: Summary with counts of successful and failed loads
        """
        max_size_bytes = int(max_size_mb * 1024 * 1024)
        successful = 0
        failed = 0
        skipped = 0
        
        for block in self.blocks:
            if isinstance(block, ImageBlock):
                # Check file size before loading
                file_size = block.get_file_size()
                if file_size and file_size > max_size_bytes:
                    logger.warning(
                        f"Skipping large image {block.file_path} "
                        f"({file_size / 1024 / 1024:.2f}MB > {max_size_mb}MB)"
                    )
                    skipped += 1
                    continue
                
                if block.load_base64():
                    successful += 1
                else:
                    failed += 1
        
        return {
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "total": len([b for b in self.blocks if isinstance(b, ImageBlock)])
        }
    
    def render_notebook(self) -> None:
        """
        Render report in Jupyter notebook
        
        Displays text as Markdown and images inline.
        """
        try:
            from IPython.display import display, Markdown, Image
            
            for block in self.blocks:
                if isinstance(block, TextBlock):
                    display(Markdown(block.text))
                elif isinstance(block, ImageBlock):
                    if block.exists():
                        display(Image(filename=block.file_path))
                    else:
                        logger.warning(f"Image not found: {block.file_path}")
                        display(Markdown(f"⚠️ *Image not found: {block.file_path}*"))
        except ImportError:
            logger.error("IPython not available. Cannot render in notebook.")

    def to_markdown(self) -> str:
        """
        Convert report to markdown string
        
        Returns:
            Markdown representation with image references
        """
        parts = []
        for i, block in enumerate(self.blocks, 1):
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ImageBlock):
                parts.append(f"![Image {i}]({block.file_path})")
            parts.append("")  # Add spacing
        
        return "\n".join(parts)

    def to_html(self, max_image_width: int = 800) -> str:
        """
        Convert report to HTML string
        
        Args:
            max_image_width: Maximum width for images in pixels
            
        Returns:
            HTML string representation
        """
        import base64
        
        html_parts = ['<div class="report-container">']
        
        for block in self.blocks:
            if isinstance(block, TextBlock):
                # Simple markdown to HTML conversion
                html_text = block.text.replace('\n', '<br>')
                html_parts.append(f'<div class="text-block">{html_text}</div>')
            
            elif isinstance(block, ImageBlock):
                if block.exists():
                    try:
                        with open(block.file_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        html_parts.append(
                            f'<div class="image-block" style="text-align: center;">'
                            f'<img src="data:image/jpeg;base64,{img_data}" '
                            f'style="max-width: {max_image_width}px;" />'
                            f'</div>'
                        )
                    except Exception as e:
                        logger.error(f"Failed to load image: {e}")
                        html_parts.append(f'<p>Image not available: {block.file_path}</p>')
                else:
                    html_parts.append(f'<p>Image not found: {block.file_path}</p>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary with data URLs
        
        Returns:
            dict: Dictionary representation with embedded image data URLs
        """
        return {
            "blocks": [
                {
                    "type": block.type,
                    "content": block.content if isinstance(block, TextBlock) else None,
                    "file_path": block.file_path if isinstance(block, ImageBlock) else None,
                    "data_url": block.get_data_url() if isinstance(block, ImageBlock) else None,
                    "page_number": getattr(block, 'page_number', None) if isinstance(block, ImageBlock) else None,
                }
                for block in self.blocks
            ],
            "stats": self.get_stats()
        }

    def get_text_summary(self, max_length: int = 500) -> str:
        """
        Get a text summary of all text blocks
        
        Args:
            max_length: Maximum length of summary
            
        Returns:
            str: Concatenated text from all text blocks
        """
        text_parts = [
            block.content 
            for block in self.blocks 
            if isinstance(block, TextBlock)
        ]
        full_text = " ".join(text_parts)
        
        if len(full_text) > max_length:
            return full_text[:max_length] + "..."
        return full_text
    
    def get_image_blocks(self) -> List[ImageBlock]:
        """
        Get all image blocks
        
        Returns:
            List[ImageBlock]: List of all image blocks in the report
        """
        return [b for b in self.blocks if isinstance(b, ImageBlock)]
    
    def get_text_blocks(self) -> List[TextBlock]:
        """
        Get all text blocks
        
        Returns:
            List[TextBlock]: List of all text blocks in the report
        """
        return [b for b in self.blocks if isinstance(b, TextBlock)]
class QueryRequest(BaseModel):
    """
    Query request model
    
    Represents a request to query the processed document with configurable parameters.
    """
    query: str = Field(
        ...,
        description="Natural language question to ask about the document",
        min_length=1,
        max_length=1000,
        examples=[
            "What was the total revenue in Q4 2024?",
            "Show me the profit margins by region",
            "What are the key financial highlights?",
            "Compare revenue growth year-over-year"
        ]
    )
    similarity_top_k: int = Field(
        default=10,
        description="Number of most similar document chunks to retrieve for context",
        ge=1,
        le=50,
        examples=[5, 10, 20]
    )
    include_images: bool = Field(
        default=True,
        description="Whether to include images (charts, tables) in the response as base64 encoded data",
        examples=[True, False]
    )
    max_image_size_mb: float = Field(
        default=5.0,
        description="Maximum size per image in megabytes (images larger than this will be skipped)",
        ge=0.1,
        le=50.0,
        examples=[1.0, 5.0, 10.0]
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging for debugging query execution",
        examples=[True, False]
    )

class DocumentProcessRequest(BaseModel):
    """
    Document processing request model
    
    Request to process a PDF document, either via file upload or URL download.
    """
    url: Optional[str] = Field(
        default=None,
        description="URL to download PDF from (alternative to file upload)",
        examples=[
            "https://example.com/financial-report-q4-2024.pdf",
            "https://sec.gov/archives/edgar/data/0001318605/000156459024040122/tsla-10q_20240630.pdf"
        ]
    )
    force_rebuild: bool = Field(
        default=False,
        description="Force rebuild of the index even if it already exists",
        examples=[True, False]
    )
    extract_images: bool = Field(
        default=True,
        description="Whether to extract images from the PDF during processing",
        examples=[True, False]
    )

class DocumentProcessResponse(BaseModel):
    """
    Document processing response model
    
    Response after successfully processing a document.
    """
    status: str = Field(
        ...,
        description="Processing status",
        examples=["success", "failed", "partial"]
    )
    message: str = Field(
        ...,
        description="Human-readable message about the processing result",
        examples=[
            "Document processed successfully",
            "Failed to parse PDF",
            "Document processed with warnings"
        ]
    )
    summary: dict = Field(
        default_factory=dict,
        description="Summary statistics about the processed document",
        examples=[{
            "total_pages": 50,
            "images_extracted": 25,
            "text_length": 45000,
            "file_size_mb": 2.5
        }]
    )
    nodes_created: int = Field(
        default=0,
        description="Number of text nodes created in the index",
        ge=0,
        examples=[50, 127, 200]
    )
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken to process the document in milliseconds",
        ge=0,
        examples=[5000.0, 12500.0, 8750.0]
    )
class QueryResponse(BaseModel):
    """
    Query response model
    
    Represents the complete response to a query, including the answer,
    supporting images, and metadata about the query execution.
    """
    query: str = Field(
        ...,
        description="The original query that was asked",
        examples=["What was the total revenue?"]
    )
    report: ReportOutput = Field(
        ...,
        description="Report output containing text blocks (answers) and image blocks (supporting visuals)"
    )
    source_pages: List[int] = Field(
        default_factory=list,
        description="List of page numbers from the source document that were referenced",
        examples=[[5, 12, 15], [3], []]
    )
    response_time_ms: float = Field(
        ...,
        description="Time taken to process the query in milliseconds",
        ge=0,
        examples=[1234.56, 2500.0, 850.25]
    )
    model_used: str = Field(
        ...,
        description="Name of the language model used to generate the response",
        examples=["gpt-4o", "gpt-4-turbo", "claude-3-opus"]
    )
    similarity_top_k: int = Field(
        default=10,
        description="Number of similar chunks that were retrieved",
        ge=1,
        examples=[10, 15, 20]
    )
    chunks_retrieved: Optional[int] = Field(
        default=None,
        description="Actual number of chunks retrieved (may be less than similarity_top_k if document is small)",
        ge=0,
        examples=[10, 7, 15]
    )
    
    def get_summary(self) -> dict:
        """
        Get a summary of the query response
        
        Returns:
            dict: Summary including query, answer preview, and statistics
        """
        return {
            "query": self.query,
            "answer_preview": self.report.get_text_summary(200),
            "total_blocks": len(self.report.blocks),
            "has_images": len(self.report.get_image_blocks()) > 0,
            "source_pages": self.source_pages,
            "response_time_ms": self.response_time_ms,
            "model": self.model_used
        }


# System prompt for LLM
SYSTEM_PROMPT = """\
You are a financial document analysis assistant specializing in earnings reports and investor presentations.

**Your Task:**
Generate comprehensive, well-structured reports that combine textual analysis with relevant visual evidence.

**Input Format:**
You will receive parsed content from financial documents containing:
- Markdown-formatted text (tables, headings, bullet points)
- Metadata with image file paths for each page

**Output Requirements:**
1. **Structure**: Create a report with interleaving TextBlock and ImageBlock elements
2. **Image Selection**: Include images ONLY when they contain:
   - Financial tables with numerical data
   - Charts/graphs showing trends
   - Key performance metrics visualizations
   - Comparative data presentations
3. **Minimum Images**: MUST include at least one ImageBlock
4. **Text Quality**: Provide clear, analytical text that:
   - Synthesizes information from multiple pages
   - Highlights key trends and patterns
   - Includes specific numbers and percentages
   - Cites sources using page numbers

**Critical Rules:**
- Output MUST be a structured tool call (ReportOutput format)
- Do NOT return plain text responses
- Prioritize images with dense tabular or graphical content
- Ensure text blocks provide context for adjacent images

**Example Structure:**
1. TextBlock: Executive summary with key findings
2. ImageBlock: Chart showing quarterly trends
3. TextBlock: Detailed analysis of the chart
4. ImageBlock: Table with specific metrics
5. TextBlock: Comparative analysis and conclusions
"""