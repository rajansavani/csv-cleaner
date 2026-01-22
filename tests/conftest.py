import sys
from pathlib import Path

# add repo root so `import src...` works in ci and locally
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))