from src import Config
from typing import List, Dict, Optional, Tuple
from llama_parse import LlamaParse
import logging, requests, shutil
from pathlib import Path
import nest_asyncio
nest_asyncio.apply()

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

    def download_pdf(self, url: str, save_path: Path, force: bool = False, 
                 timeout: int = 120, max_retries: int = 3) -> bool:
        """
        Download PDF from URL with retry logic
        
        Args:
            url: URL to download from
            save_path: Local path to save the PDF
            force: If True, re-download even if file exists
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if save_path.exists() and not force:
                logger.info(f"File already exists: {save_path}")
                return True
            
            # Ensure parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try multiple times
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Downloading PDF from {url} (attempt {attempt}/{max_retries})")
                    
                    # Download with streaming to handle large files
                    response = requests.get(
                        url, 
                        timeout=timeout,
                        stream=True,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        },
                        allow_redirects=True
                    )
                    response.raise_for_status()
                    
                    # Get file size if available
                    total_size = int(response.headers.get('content-length', 0))
                    if total_size > 0:
                        logger.info(f"File size: {total_size / (1024*1024):.2f} MB")
                    
                    # Download in chunks with progress
                    chunk_size = 8192
                    downloaded = 0
                    last_progress = 0
                    
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # Log progress every 10%
                                if total_size > 0:
                                    progress = int((downloaded / total_size) * 100)
                                    if progress >= last_progress + 10:
                                        logger.info(f"Download progress: {progress}%")
                                        last_progress = progress
                    
                    # Verify download
                    if save_path.exists() and save_path.stat().st_size > 0:
                        logger.info(
                            f"✓ Successfully downloaded to {save_path} "
                            f"({downloaded / (1024*1024):.2f} MB)"
                        )
                        return True
                    else:
                        raise IOError("Downloaded file is empty or doesn't exist")
                        
                except requests.Timeout as e:
                    logger.warning(f"Attempt {attempt} timed out after {timeout} seconds")
                    if attempt < max_retries:
                        logger.info(f"Retrying in 5 seconds...")
                        import time
                        time.sleep(5)
                    else:
                        logger.error(f"All {max_retries} download attempts failed due to timeout")
                        raise
                        
                except requests.RequestException as e:
                    logger.warning(f"Attempt {attempt} failed: {e}")
                    if attempt < max_retries:
                        logger.info(f"Retrying in 5 seconds...")
                        import time
                        time.sleep(5)
                    else:
                        logger.error(f"All {max_retries} download attempts failed")
                        raise
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}", exc_info=True)
            # Clean up partial download
            if save_path.exists():
                try:
                    save_path.unlink()
                    logger.info("Cleaned up partial download")
                except:
                    pass
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
        
    # def extract_images(self, download_path: Path, clear_existing: bool = True) -> bool:
    #     """
    #     Extract and download page images from last parsed document
        
    #     Args:
    #         download_path: Directory to save images
    #         clear_existing: Whether to clear existing images
            
    #     Returns:
    #         True if successful, False otherwise
    #     """
    #     try:
    #         if not hasattr(self, '_last_parsed_obj'):
    #             logger.error("No parsed document available. Call parse_document() first.")
    #             return False
            
    #         # Clear existing images if requested
    #         if clear_existing and download_path.exists():
    #             logger.info(f"Clearing existing images in {download_path}")
    #             shutil.rmtree(download_path)
            
    #         # Ensure directory exists
    #         download_path.mkdir(parents=True, exist_ok=True)
            
    #         logger.info(f"Extracting images to {download_path}")
    #         self.parser.get_images(
    #             self._last_parsed_obj, 
    #             download_path=download_path
    #         )
            
    #         # Verify images were downloaded
    #         image_files = list(download_path.glob("*.jpg"))
    #         image_count = len(image_files)
    #         logger.info(f"Successfully extracted {image_count} images")

    #         # Diagnostic logging: list images with sizes and flag zero-byte files
    #         for img in sorted(image_files):
    #             try:
    #                 resolved = img.resolve()
    #             except Exception:
    #                 resolved = img
    #             try:
    #                 size = img.stat().st_size
    #             except Exception as e:
    #                 size = None
    #                 logger.warning(f"Unable to stat image {resolved}: {e}")

    #             logger.info(f"Image: {resolved} size={size}")
    #             if size == 0:
    #                 logger.error(f"Zero-byte image detected: {resolved}")
            
    #         # Verify we have images for all pages
    #         if hasattr(self, '_last_parsed_obj') and self._last_parsed_obj:
    #             pages = self._last_parsed_obj[0].get("pages", [])
    #             if image_count != len(pages):
    #                 logger.warning(
    #                     f"Image count mismatch: {image_count} images extracted "
    #                     f"but document has {len(pages)} pages"
    #                 )
    #                 # List actual image filenames for debugging
    #                 logger.info("Extracted image files:")
    #                 for img_file in sorted(image_files):
    #                     logger.info(f"  - {img_file.name}")
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"Failed to extract images: {e}", exc_info=True)
    #         return False
    def extract_images(self, download_path: Path, clear_existing: bool = True) -> bool:
        """
        Extract and download page images from last parsed document
        
        This version manually extracts images from the parsed JSON data,
        completely avoiding LlamaParse's get_images() method which has
        event loop issues.
        
        Args:
            download_path: Directory to save images
            clear_existing: Whether to clear existing images
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.debug_parsed_data()

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
            
            # Extract images manually from parsed data
            # This completely avoids the get_images() async issue caused by the version above
            
            pages = self._last_parsed_obj[0].get("pages", [])
            total_images_downloaded = 0
            
            for page_idx, page_data in enumerate(pages):
                # Get images array from page data
                images = page_data.get("images", [])
                
                if not images:
                    logger.debug(f"No images found in page {page_idx + 1}")
                    continue
                
                logger.info(f"Page {page_idx + 1} has {len(images)} image(s)")
                
                for img_idx, image_info in enumerate(images):
                    try:
                        # Extract image URL or path from the image info
                        # LlamaParse stores images with URLs we can download
                        img_url = None
                        
                        # Try different possible keys for the image URL
                        if isinstance(image_info, dict):
                            img_url = (
                                image_info.get("url") or 
                                image_info.get("image_url") or
                                image_info.get("path") or
                                image_info.get("image_path")
                            )
                        elif isinstance(image_info, str):
                            # Sometimes it's just a URL string
                            img_url = image_info
                        
                        if not img_url:
                            logger.warning(
                                f"No URL found for image {img_idx} on page {page_idx + 1}"
                            )
                            continue
                        
                        # Download the image
                        import requests
                        
                        logger.debug(f"Downloading image from: {img_url}")
                        
                        response = requests.get(img_url, timeout=30)
                        response.raise_for_status()
                        
                        # Determine filename
                        # Use page number for consistent naming
                        if len(images) == 1:
                            # Single image per page - use simple naming
                            img_filename = f"page-{page_idx + 1}.jpg"
                        else:
                            # Multiple images per page - include image index
                            img_filename = f"page-{page_idx + 1}-img-{img_idx + 1}.jpg"
                        
                        img_path = download_path / img_filename
                        
                        # Save image
                        with open(img_path, 'wb') as f:
                            f.write(response.content)
                        
                        file_size = len(response.content)
                        logger.info(
                            f"Downloaded: {img_filename} ({file_size} bytes)"
                        )
                        total_images_downloaded += 1
                        
                    except Exception as e:
                        logger.warning(
                            f"Failed to download image {img_idx} from page {page_idx + 1}: {e}"
                        )
                        continue
            
            # Verify images were downloaded
            image_files = list(download_path.glob("*.jpg"))
            image_count = len(image_files)
            
            logger.info(f"Successfully downloaded {total_images_downloaded} images")
            logger.info(f"Found {image_count} .jpg files in {download_path}")
            
            # Log each image file
            for img in sorted(image_files):
                try:
                    size = img.stat().st_size
                    logger.info(f"Image file: {img.name} size={size} bytes")
                    if size == 0:
                        logger.error(f"Zero-byte image detected: {img}")
                except Exception as e:
                    logger.warning(f"Unable to stat image {img}: {e}")
            
            # Verify we have images for all pages
            if pages:
                if image_count != len(pages):
                    logger.warning(
                        f"Image count mismatch: {image_count} images extracted "
                        f"but document has {len(pages)} pages"
                    )
                    logger.info("This might be normal if some pages don't have images")
            
            return total_images_downloaded > 0
            
        except Exception as e:
            logger.error(f"Failed to extract images: {e}", exc_info=True)
            return False

    def debug_parsed_data(self):
        """
        Debug method to see what LlamaParse is returning
        
        This will show you:
        - If images are in the parsed data
        - What format they're in
        - Where the image URLs are (if any)
        """
    
        if not hasattr(self, '_last_parsed_obj'):
            logger.error("❌ No parsed object found!")
            return
        
        logger.info("=" * 70)
        logger.info("🔍 DEBUGGING PARSED DATA STRUCTURE")
        logger.info("=" * 70)
        
        parsed = self._last_parsed_obj
        logger.info(f"Parsed data type: {type(parsed)}")
        logger.info(f"Parsed data length: {len(parsed)}")
        
        if parsed and len(parsed) > 0:
            first_item = parsed[0]
            logger.info(f"First item keys: {list(first_item.keys())}")
            
            # Check pages
            pages = first_item.get("pages", [])
            logger.info(f"📄 Total pages: {len(pages)}")
            
            # Examine first 3 pages in detail
            for i, page in enumerate(pages[:3]):
                logger.info(f"")
                logger.info(f"--- 📄 Page {i + 1} ---")
                logger.info(f"Page keys: {list(page.keys())}")
                
                # Check for images (plural)
                images = page.get("images", [])
                logger.info(f"🖼️  Images array length: {len(images)}")
                
                if images:
                    logger.info(f"✅ IMAGES FOUND in page {i + 1}!")
                    logger.info(f"First image type: {type(images[0])}")
                    
                    if isinstance(images[0], dict):
                        logger.info(f"Image keys: {list(images[0].keys())}")
                        
                        # Try to find URL in various possible keys
                        img = images[0]
                        possible_url_keys = ['url', 'image_url', 'path', 'image_path', 'src', 'href']
                        
                        for key in possible_url_keys:
                            if key in img:
                                value = img[key]
                                logger.info(f"  ✅ Found '{key}': {str(value)[:100]}...")
                        
                        # Print all key-value pairs (truncated)
                        logger.info("  All image data:")
                        for key, value in img.items():
                            if isinstance(value, str) and len(value) > 100:
                                logger.info(f"    {key}: <string, {len(value)} chars>")
                            else:
                                logger.info(f"    {key}: {value}")
                                
                    elif isinstance(images[0], str):
                        logger.info(f"Image is a string: {images[0][:100]}...")
                    else:
                        logger.info(f"Image value: {images[0]}")
                else:
                    logger.warning(f"⚠️  No 'images' array in page {i + 1}")
                    
                    # Check for singular 'image' key
                    if 'image' in page:
                        logger.info(f"Found singular 'image' key: {type(page['image'])}")
                    
                    # Check for other possible image keys
                    possible_keys = ['img', 'pictures', 'figures', 'graphics']
                    for key in possible_keys:
                        if key in page:
                            logger.info(f"Found '{key}' key: {type(page[key])}")
            
            # Summary
            logger.info("")
            logger.info("=" * 70)
            logger.info("📊 SUMMARY")
            logger.info("=" * 70)
            
            total_pages_with_images = sum(1 for p in pages if p.get("images"))
            total_images = sum(len(p.get("images", [])) for p in pages)
            
            logger.info(f"Pages with images: {total_pages_with_images}/{len(pages)}")
            logger.info(f"Total images found: {total_images}")
            
            if total_images == 0:
                logger.warning("⚠️  NO IMAGES FOUND IN PARSED DATA!")
                logger.warning("Possible reasons:")
                logger.warning("  1. PDF has no extractable images")
                logger.warning("  2. Images are embedded as page backgrounds")
                logger.warning("  3. PDF is scanned (images are part of scan)")
                logger.warning("  4. LlamaParse API isn't returning image URLs")
                logger.warning("  5. Try setting gpt4o_mode=False")
            else:
                logger.info(f"✅ Images are present in parsed data")
        
        logger.info("=" * 70)

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