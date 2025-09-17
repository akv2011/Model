
from pathlib import Path
import os

OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', './output')).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_JSON = Path(os.environ.get('DATASET_JSON', OUTPUT_DIR / 'dataset_0.json')).resolve()

CACHE_NUM_CT_TRAIN = int(os.environ.get('CACHE_NUM_CT_TRAIN', 150))
CACHE_NUM_MRI_TRAIN = int(os.environ.get('CACHE_NUM_MRI_TRAIN', 100))
CACHE_NUM_CT_VAL = int(os.environ.get('CACHE_NUM_CT_VAL', 40))
CACHE_NUM_MRI_VAL = int(os.environ.get('CACHE_NUM_MRI_VAL', 45))

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 1))
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 0))
PIN_MEMORY = bool(int(os.environ.get('PIN_MEMORY', 1)))

OUTPUT_CLASSES = int(os.environ.get('OUTPUT_CLASSES', 3))
CHANNELS = tuple(int(v) for v in os.environ.get('CHANNELS', '12,24,48,96,192').split(','))
STRIDES = tuple(int(v) for v in os.environ.get('STRIDES', '2,2,2,2').split(','))

NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 50))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 1e-5))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', 1e-5))


ROI_SIZE = tuple(int(v) for v in os.environ.get('ROI_SIZE', '256,256,16').split(','))
SW_BATCH_SIZE = int(os.environ.get('SW_BATCH_SIZE', 4))
SLIDING_WINDOW_OVERLAP = float(os.environ.get('SLIDING_WINDOW_OVERLAP', 0.8))
