import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

logger.info("✓ If you see this, basicConfig works")
logger.warning("✓ Warning message")
logger.error("✓ Error message")