from src import Config
import logging, re
from pathlib import Path
from typing import List, Optional
from llama_index.core.schema import TextNode
from llama_index.core import (
                SummaryIndex, StorageContext, 
                load_index_from_storage
                )
import shutil

logger = logging.getLogger(__name__)

class DocumentUtils:
    """Utility functions for document processing"""
    
    @staticmethod
    def extract_page_number(filename: str) -> int:
        """
        Extract page number from filename with flexible pattern matching
        
        Supports patterns like:
        - document-page-5.jpg
        - page-5.jpg
        - page_5.jpg
        - rjf1q25-page-5.jpg
        
        Args:
            filename: Image filename (e.g., 'document-page-5.jpg')
            
        Returns:
            Page number or float('inf') if not found
        """
          # Try multiple patterns
        patterns = [
            r'-page-(\d+)\.jpg$',      # document-page-5.jpg
            r'page-(\d+)\.jpg$',       # page-5.jpg
            r'page_(\d+)\.jpg$',       # page_5.jpg
            r'_(\d+)\.jpg$',           # document_5.jpg
        ]

        for pattern in patterns:
            match = re.search(pattern, str(filename), re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # If no pattern matches, log warning and return infinity
        logger.warning(f"Could not extract page number from filename: {filename}")
        return float('inf')
    
    @staticmethod
    def get_sorted_image_files(image_dir: Path) -> List[Path]:
        """
        Get sorted list of image files by page number
        
        Args:
            image_dir: Directory containing images
            
        Returns:
            Sorted list of image Path objects
        """
        if not image_dir.exists():
            logger.warning(f"Image directory not found: {image_dir}")
            return []
        
        files = [
            f for f in image_dir.iterdir() 
            if f.is_file() and f.suffix.lower() == '.jpg'
        ]
        return sorted(files, key=lambda f: DocumentUtils.extract_page_number(f.name))
    
    @staticmethod
    def create_text_nodes(json_pages: List[dict], image_dir: Path) -> List[TextNode]:
        """
        Create TextNode objects from parsed JSON pages
        
        Each node contains:
        - Page number
        - Parsed markdown text
        - Path to corresponding page image
        
        Args:
            json_pages: List of page dictionaries from LlamaParse
            image_dir: Directory containing page images
            
        Returns:
            List of TextNode objects with metadata
        """
        nodes = []
        sorted_images = DocumentUtils.get_sorted_image_files(image_dir)
        
        # Log image filenames for debugging
        logger.info(f"Found {len(sorted_images)} images in {image_dir}")
        for i, img in enumerate(sorted_images):
            page_num = DocumentUtils.extract_page_number(img.name)
            logger.info(f"  Image {i}: {img.name} -> page {page_num}")
        
        logger.info(f"Creating nodes for {len(json_pages)} pages")
        for idx, page_data in enumerate(json_pages):
            page_num = idx + 1
            metadata = {
                'page_num': idx + 1,
                'parsed_text_markdown': page_data.get('md', ''),
            }
            
            # Add image path if available
            if idx < len(sorted_images):
                metadata['image_path'] = str(sorted_images[idx])
                logger.debug(f"Page {page_num} matched to image: {sorted_images[idx].name}")
            else:
                logger.warning(f"No image found for page {page_num} (index {idx})")
                logger.warning(f"  Total images: {len(sorted_images)}, Total pages: {len(json_pages)}")
            
            # Create node with empty text (metadata contains the actual content)
            node = TextNode(text="", metadata=metadata)
            nodes.append(node)
        
        logger.info(f"Created {len(nodes)} text nodes")
        logger.info(f"  Nodes with images: {sum(1 for n in nodes if 'image_path' in n.metadata)}")
        logger.info(f"  Nodes without images: {sum(1 for n in nodes if 'image_path' not in n.metadata)}")
        return nodes
    
class IndexBuilder:
    """
    Builds and manages LlamaIndex indexes
    
    Handles:
    - Index creation from text nodes
    - Index persistence to disk
    - Index loading from storage
    """
    
    def __init__(self, config: Config):
        """
        Initialize index builder
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.index: Optional[SummaryIndex] = None
        logger.info("IndexBuilder initialized")

    def build_index(self, text_nodes: List[TextNode], force_rebuild: bool = False) -> SummaryIndex:
        """
        Build or load summary index from text nodes
        
        Args:
            text_nodes: List of TextNode objects with metadata
            force_rebuild: If True, rebuild even if storage exists
            
        Returns:
            SummaryIndex instance
        """
        try:
            storage_path = self.config.storage_dir
            
            # Load from storage if exists and not forcing rebuild
            if storage_path.exists() and not force_rebuild:
                logger.info("Loading existing index from storage")
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(storage_path)
                )
                self.index = load_index_from_storage(
                    storage_context, 
                    index_id="summary_index"
                )
                logger.info("Index loaded successfully")
            
            else:
                # Build new index
                logger.info("Building new index from text nodes")
                self.index = SummaryIndex(text_nodes)
                
                # Persist to storage
                self.index.set_index_id("summary_index")
                storage_path.mkdir(parents=True, exist_ok=True)
                self.index.storage_context.persist(str(storage_path))
                logger.info(f"Index built and persisted to {storage_path}")
            
            return self.index
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}", exc_info=True)
            raise
    
    def get_index(self) -> Optional[SummaryIndex]:
        """Get the current index instance"""
        return self.index
    
    def clear_storage(self) -> bool:
        """
        Clear persisted index storage
        
        Returns:
            True if successful, False otherwise
        """
        try:
            storage_path = self.config.storage_dir
            
            if storage_path.exists():
                shutil.rmtree(storage_path)
                logger.info(f"Cleared storage: {storage_path}")
                return True
            else:
                logger.info("No storage to clear")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear storage: {e}")
            return False

