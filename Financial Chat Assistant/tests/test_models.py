from pathlib import Path
import sys

# Add the project root to sys.path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
print(f"Project root - {project_root} added to sys.path:")

from src import TextBlock, ImageBlock, ReportOutput, Config

# Create sample report
text = TextBlock(text="This is a sample text block for testing.")

# images_dir = Config().images_dir

# image = ImageBlock(file_path="path/to/image.png")
report = ReportOutput(blocks=[text])

print(report.get_stats())