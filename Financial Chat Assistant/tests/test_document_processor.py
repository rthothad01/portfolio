from pathlib import Path
import sys

# Add the project root to sys.path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
print(f"Project root - {project_root} added to sys.path:")

from src import Config, DocumentProcessor

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
result = processor.process_document(file_path=file_path)

if result:
    pages, images_dir = result
    summary = processor.get_page_summary(pages)
    print("✓ Document processing successful")
    print(f"Summary: {summary}")
else:
    print("✗ Document processing failed")
