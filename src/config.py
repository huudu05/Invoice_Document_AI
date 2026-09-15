import shutil
from pathlib import Path

# If pdftoppm is already in the PATH (usually true on Linux/macOS),
# no need to set anything — leave it as None.
# If not, enter the path here manually (common on Windows).
POPPLER_PATH: str | None = None if shutil.which("pdftoppm") else r"C:\Program Files\poppler-26.02.0\Library\bin"

#input_proccesing
MAX_FILE_SIZE_MB = 20
MIN_IMAGE_WIDTH = 300
MIN_IMAGE_HEIGHT = 300
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

#knowledge_base
INFERENCE_DIR = Path("outputs/inference")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "invoices"

#training
DEFAULT_MODEL_DIR = Path(
    "models/layoutlmv3/best_model"
)

DEFAULT_OUTPUT_DIR = Path(
    "models/layoutlmv3/evaluation/test"
)


DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 0


SEED = 42
MODEL_NAME = "microsoft/layoutlmv3-base"
VAL_RATIO = 0.10
EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CLIP_MAX_NORM = 1.0
NUM_WORKERS = 0
EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 1e-4
OUTPUT_DIR = Path(
    "models/layoutlmv3"
)

#inference
DEFAULT_OUTPUT_DIR_INFERENCE = Path("outputs/inference")

#pipeline
DEFAULT_TOP_K = 5
MIN_RELEVANCE_SCORE = 0.80

