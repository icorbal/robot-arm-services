"""Root conftest — ensures src is importable and sets asyncio mode."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
