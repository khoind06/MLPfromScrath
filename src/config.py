import os
# 1. PATHS & SYSTEM
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "day_0.gz")
SPLIT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "split")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_FILE = "train.csv"
VAL_FILE   = "val.csv"
TEST_FILE  = "test.csv"
SEED = 42

# 2. DATASET DEFINITION
NUM_SAMPLES = 300_000   
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

LABEL_COLUMN = "label"
NUMERIC_FEATURES = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_FEATURES = [f"C{i}" for i in range(1, 27)]

# 3. FEATURE ENGINEERING
HASHING_DIM = 2**17      
CROSS_HASH_DIM = 2**14   
NUM_BINS = 10           
MIN_CATEGORY_FREQ = 5   

# Các cặp feature tương tác mạnh nhất
CROSS_FEATURES = [
    ("C1", "C2"), ("C1", "C3"), ("C2", "C3"),
    ("C6", "C9"), ("C10", "C11"), ("C14", "C15"), 
    ("C20", "C21"), ("I5", "C20"), ("I7", "C26")
]

# 4. MODEL HYPERPARAMETERS
# 4.1. Kiến trúc Mạnh (Large Capacity)
HIDDEN_DIMS = [512, 256, 128]  
PROJ_DIM = 32           

# 4.2. Tốc độ học 
BATCH_SIZE = 1024        
NUM_EPOCHS = 30          
LEARNING_RATE = 0.001  
GRADIENT_CLIP_VALUE = 5.0

# 4.3. Regularization nhẹ (Low Constraints)
DROPOUT_RATE = 0.3     
WEIGHT_DECAY = 1e-5     
USE_BATCH_NORM = True

# 4.4. Scheduler & Early Stop
WARMUP_EPOCHS = 2      
LR_DECAY_RATE = 0.85
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4