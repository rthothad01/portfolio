from pathlib import Path
import sys, logging

# Add the project root to sys.path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
print(f"Project root - {project_root} added to sys.path:")

from src import TextBlock, ImageBlock, ReportOutput, Config

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


# Create sample report
text = TextBlock(text="This is a sample text block for testing.")

# images_dir = Config().images_dir

# image = ImageBlock(file_path="path/to/image.png")
report = ReportOutput(blocks=[text])

print(report.get_stats())
print("✓ Models work correctly")