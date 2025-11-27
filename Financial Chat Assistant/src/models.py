from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class TextBlock(BaseModel):
    """
    Text block containing analytical content
    
    Used for textual analysis, insights, and explanations in reports.
    """
    text: str = Field(
        ..., 
        description="The text content for this block",
        min_length=1
    )
    
    def __str__(self) -> str:
        return self.text[:100] + ("..." if len(self.text) > 100 else "")
    
class ImageBlock(BaseModel):
    """
    Image block containing visual data
    
    References an image file path for charts, tables, or visualizations.
    """
    file_path: str = Field(
        ..., 
        description="File path to the image (relative or absolute)"
    )
    
    def exists(self) -> bool:
        """Check if the image file exists"""
        return Path(self.file_path).exists()
    
    def __str__(self) -> str:
        return f"Image({self.file_path})"

class ReportOutput(BaseModel):
    """
    Complete report with interleaved text and image blocks
    
    Represents a multimodal report that combines textual analysis
    with supporting visual evidence. Must contain at least one image block.
    """
    
    blocks: List[Union[TextBlock, ImageBlock]] = Field(
        ..., 
        description="Ordered list of text and image blocks",
        min_length=1
    )

    def validate_blocks(self) -> bool:
        """Validate that report has at least one image block"""
        has_image = any(isinstance(b, ImageBlock) for b in self.blocks)
        if not has_image:
            logger.warning("Report has no image blocks")
        return has_image
    
    def get_stats(self) -> dict:
        """Get statistics about the report"""
        return {
            "total_blocks": len(self.blocks),
            "text_blocks": sum(1 for b in self.blocks if isinstance(b, TextBlock)),
            "image_blocks": sum(1 for b in self.blocks if isinstance(b, ImageBlock)),
            "has_images": self.validate_blocks()
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
        """Convert report to dictionary for JSON serialization"""
        return {
            'blocks': [
                {
                    'type': 'text' if isinstance(b, TextBlock) else 'image',
                    'content': b.text if isinstance(b, TextBlock) else b.file_path
                }
                for b in self.blocks
            ],
            'stats': self.get_stats()
        }

class QueryRequest(BaseModel):
    """Request model for query API endpoint"""
    query: str = Field(..., description="The question to ask", min_length=1)
    similarity_top_k: Optional[int] = Field(
        10, 
        description="Number of similar chunks to retrieve",
        ge=1,
        le=20
    )
    include_images: bool = Field(
        True,
        description="Whether to include images in response"
    )


class QueryResponse(BaseModel):
    """Response model for query API endpoint"""
    query: str = Field(..., description="The original query")
    report: ReportOutput = Field(..., description="The generated report")
    source_pages: List[int] = Field(
        default_factory=list,
        description="Page numbers of source documents"
    )
    response_time_ms: float = Field(
        ..., 
        description="Response time in milliseconds"
    )
    model_used: str = Field(..., description="LLM model used for generation")


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