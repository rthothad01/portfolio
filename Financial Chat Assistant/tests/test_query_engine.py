from pathlib import Path
import sys, logging

# Add the project root to sys.path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
print(f"Project root - {project_root} added to sys.path:")

from src import Config, DocumentProcessor, DocumentUtils, IndexBuilder, QueryEngineBuilder
print("✓ Imports work correctly")

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

# Test with sample URL
print("Processing document from sample URL...")
# result = processor.process_document(
#     download_url="https://www.raymondjames.com/-/media/rj/dotcom/files/our-company/news-and-media/2025-press-releases/rjf20250129-1q-presentation.pdf"
# )
print(f"Current working directory is {Path.cwd()}")
project_root = config.project_root
data_dir = project_root / config.data_dir 
pdf_file="RJF20250129 1Q Presentation.pdf"
file_path = data_dir.joinpath(pdf_file)
logger.info(f"Processing file at {file_path}")

result = processor.process_document(file_path=file_path)

if result:
    pages, images_dir = result
    print("✓ Document processing successful")
    
    # Create text nodes
    text_nodes = DocumentUtils.create_text_nodes(pages, images_dir)
    print(f"✓ Created {len(text_nodes)} text nodes")
    
    # Build index
    index_builder = IndexBuilder(config)
    index = index_builder.build_index(text_nodes)
    print(f"✓ Built index with {len(text_nodes)} nodes")

    # Create query engine
    qe_builder = QueryEngineBuilder(config)
    query_engine = qe_builder.create_query_engine(index)
    print("✓ Query engine created successfully")

    # Test query
    query="What is the Net Interest Income for Q1 2025?"
    response = qe_builder.query(query)
    if response:
        print("✓ Query executed successfully")
        print(f"Query: {query}")
        print(f"Blocks: {response.get_stats()}")
    else:
        print("✗ Query execution failed")