import sys
from pathlib import Path

# Make ``sample_pkg`` importable without installing the example.
sys.path.insert(0, str(Path(__file__).parent))
