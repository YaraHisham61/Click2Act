from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 

MODELS_PATH = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data"
# Click2Act canonical palette
C2A_PALETTE = {
    "primary"   : "#2E86AB",
    "secondary" : "#A23B72",
    "tertiary"  : "#F18F01",
    "neutral"   : "#6C757D",
    "success"   : "#3BB273",
    "warning"   : "#E84855",
    "bg"        : "#F8F9FA",
    "text"      : "#212529",
}