from src import Config
from typing import List, Dict, Optional, Tuple
from llama_parse import LlamaParse
import logging, requests, shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles PDF document parsing and image extraction using LlamaParse
    
    This class manages:
    - PDF downloading (if needed)
    - Document parsing with GPT-4o mode
    - Image extraction from parsed pages
    - Error handling and logging
    """
    
    def __init__(self, config: Config):
        """
        Initialize document processor
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.parser = LlamaParse(
            api_key=config.llamaparse_api_key,
            result_type=config.result_type,
            verbose=config.verbose,
            gpt4o_mode=config.gpt4o_mode
        )
        logger.info("DocumentProcessor initialized")

    def download_pdf(self, url: str, save_path: Path, force: bool = False) -> bool:
        """
        Download PDF from URL
        
        Args:
            url: URL to download from
            save_path: Local path to save the PDF
            force: If True, re-download even if file exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if save_path.exists() and not force:
                logger.info(f"File already exists: {save_path}")
                return True
            
            logger.info(f"Downloading PDF from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Ensure parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded to {save_path}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF: {e}")
            return False
        except IOError as e:
            logger.error(f"Failed to save PDF: {e}")
            return False
        
    def parse_document(self, file_path: Path) -> Optional[List[Dict]]:
        """
        Parse PDF document and extract structured data
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of page dictionaries or None if parsing fails
            
        Each page dictionary contains:
            - 'md': Markdown text
            - 'images': List of image metadata
            - 'page': Page number
        """
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            logger.info(f"Parsing document: {file_path}")
            md_json_obj = self.parser.get_json_result(file_path=str(file_path))
            
            if not md_json_obj or len(md_json_obj) == 0:
                logger.error("Parser returned empty result")
                return None
            
            # Extract pages
            md_json_list = md_json_obj[0].get("pages", [])
            
            # Log metadata
            metadata = md_json_obj[0].get('job_metadata', {})
            logger.info(f"Successfully parsed {len(md_json_list)} pages")
            
            if self.config.verbose and metadata:
                logger.info(f"Credits used: {metadata.get('credits_used', 'N/A')}")
                logger.info(f"Job pages: {metadata.get('job_pages', 'N/A')}")
            
            # Store the full object for image extraction
            self._last_parsed_obj = md_json_obj
            
            return md_json_list
            
        except Exception as e:
            logger.error(f"Failed to parse document: {e}", exc_info=True)
            return None
        
    def extract_images(self, download_path: Path, clear_existing: bool = True) -> bool:
        """
        Extract and download page images from last parsed document
        
        Args:
            download_path: Directory to save images
            clear_existing: Whether to clear existing images
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not hasattr(self, '_last_parsed_obj'):
                logger.error("No parsed document available. Call parse_document() first.")
                return False
            
            # Clear existing images if requested
            if clear_existing and download_path.exists():
                logger.info(f"Clearing existing images in {download_path}")
                shutil.rmtree(download_path)
            
            # Ensure directory exists
            download_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Extracting images to {download_path}")
            self.parser.get_images(
                self._last_parsed_obj, 
                download_path=download_path
            )
            
            # Verify images were downloaded
            image_count = len(list(download_path.glob("*.jpg")))
            logger.info(f"Successfully extracted {image_count} images")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract images: {e}", exc_info=True)
            return False
        
    def process_document(self, file_path: Optional[Path] = None, download_url: Optional[str] = None, \
                         images_dir: Optional[Path] = None
                        ) -> Optional[Tuple[List[Dict], Path]]:
        """
        Complete document processing pipeline
        
        Downloads (if needed), parses, and extracts images from a PDF.
        
        Args:
            file_path: Path to PDF file (uses config default if None)
            download_url: Optional URL to download PDF from
            images_dir: Directory for images (uses config default if None)
            
        Returns:
            Tuple of (parsed_pages, images_directory) or None if processing fails
        """
        logger.info("Document URL is {download_url}")

        try:
            # Use defaults from config
            file_path = file_path or self.config.pdf_path
            images_dir = images_dir or self.config.images_dir

            logger.info("File path: %s", file_path)
            logger.info("Images folder: %s", images_dir)
            logger.info("Download URL: %s", download_url)
            
            # Download if URL provided and file doesn't exist
            if download_url and not file_path.exists():
                if not self.download_pdf(download_url, file_path):
                    return None
            
            # Parse document
            parsed_pages = self.parse_document(file_path)
            if parsed_pages is None:
                return None
            
            # Extract images
            if not self.extract_images(images_dir):
                logger.warning("Image extraction failed, continuing with text only")
            
            return parsed_pages, images_dir
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            return None
        
    def get_page_summary(self, parsed_pages: List[Dict]) -> Dict:
        """
        Get summary statistics about parsed pages
        
        Args:
            parsed_pages: List of parsed page dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_pages': len(parsed_pages),
            'pages_with_images': 0,
            'pages_with_tables': 0,
            'total_images': 0,
            'average_text_length': 0
        }
        
        text_lengths = []
        for page in parsed_pages:
            # Check for images
            images = page.get('images', [])
            if images:
                summary['pages_with_images'] += 1
                summary['total_images'] += len(images)
            
            # Check for tables (heuristic: contains table markdown)
            md_text = page.get('md', '')
            if '|' in md_text and '---' in md_text:
                summary['pages_with_tables'] += 1
            
            text_lengths.append(len(md_text))
        
        if text_lengths:
            summary['average_text_length'] = sum(text_lengths) / len(text_lengths)
        
        return summary