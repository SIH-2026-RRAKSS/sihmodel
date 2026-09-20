import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.terminal_prediction import compute_7_component_terminal_score
print("Imported successfully!")
