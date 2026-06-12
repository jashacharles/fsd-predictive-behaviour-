# controls the training env
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "training"
EVAL_DATA_DIR = DATA_DIR / "evaluation"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_driving_transformer.pth"
STATS_PATH = OUTPUT_DIR / "normalization_stats.npz"

SEQ_LEN = 30
PREDICT_HORIZON = 15  # frames ahead to predict (~0.5 seconds at 30 FPS)
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 3

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"