from pathlib import Path
import sys, logging

# Add the project root to sys.path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
print(f"Project root - {project_root} added to sys.path:")

from src import Config, DocumentProcessor, DocumentUtils, IndexBuilder

# Setup logging
# Remove all existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
     handlers=[
        logging.StreamHandler(sys.stdout)  # Output to console
    ])

logger = logging.getLogger(__name__)
# Test it immediately
logger.info("=" * 60)
print("🔧 Logging configured successfully! - from print statement")
logger.info("🔧 Logging configured successfully!")
logger.info("=" * 60)

# Process document
config = Config()
processor = DocumentProcessor(config)
print("✓ DocumentProcessor initialized successfully")

print(f"Current working directory is {Path.cwd()}")
project_root = config.project_root
data_dir = project_root / config.data_dir 
pdf_file="RJF20250129 1Q Presentation.pdf"
file_path = data_dir.joinpath(pdf_file)
result = processor.process_document(file_path=file_path)

if result:
    pages, images_dir = result
    summary = processor.get_page_summary(pages)
    print("✓ Document processing successful")
    print(f"Summary: {summary}")

    pages, images_dir = result
    
    # Create text nodes
    text_nodes = DocumentUtils.create_text_nodes(pages, images_dir)
    print(f"✓ Created {len(text_nodes)} text nodes")
    
    # Build index
    index_builder = IndexBuilder(config)
    index = index_builder.build_index(text_nodes)
    print(f"✓ Built index with {len(text_nodes)} nodes")
else:
    print("✗ Processing failed")

