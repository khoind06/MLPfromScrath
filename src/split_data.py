import gzip
import csv
import os
import random
import time

from config import (
    RAW_DATA_PATH,
    SPLIT_DATA_DIR,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    NUM_SAMPLES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
)

def split_data():
    print("SPLITTING DATA")
    print(f"Total samples: {NUM_SAMPLES:,}")
    print(f"Train ratio: {TRAIN_RATIO:.1%} (~{int(NUM_SAMPLES * TRAIN_RATIO):,})")
    print(f"Val ratio:   {VAL_RATIO:.1%} (~{int(NUM_SAMPLES * VAL_RATIO):,})")
    print(f"Test ratio:  {TEST_RATIO:.1%} (~{int(NUM_SAMPLES * TEST_RATIO):,})")    
    # Set random seed
    random.seed(SEED)
    
    # Create output directory
    os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
    
    # Define output paths
    train_path = os.path.join(SPLIT_DATA_DIR, TRAIN_FILE)
    val_path   = os.path.join(SPLIT_DATA_DIR, VAL_FILE)
    test_path  = os.path.join(SPLIT_DATA_DIR, TEST_FILE)
    
    # Initialize counters
    counts = {'train': 0, 'val': 0, 'test': 0}
    start_time = time.time()
    
    # Open output CSV files
    with open(train_path, "w", newline="", encoding="utf-8") as train_f, \
         open(val_path,   "w", newline="", encoding="utf-8") as val_f, \
         open(test_path,  "w", newline="", encoding="utf-8") as test_f:
        
        train_writer = csv.writer(train_f)
        val_writer   = csv.writer(val_f)
        test_writer  = csv.writer(test_f)
        
        # Read raw gz file (streaming)
        print("\nReading data...")
        with gzip.open(RAW_DATA_PATH, "rt", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            
            for count, row in enumerate(reader):
                if count >= NUM_SAMPLES:
                    break
                
                # Random split
                r = random.random()
                
                if r < TRAIN_RATIO:
                    train_writer.writerow(row)
                    counts['train'] += 1
                elif r < TRAIN_RATIO + VAL_RATIO:
                    val_writer.writerow(row)
                    counts['val'] += 1
                else:
                    test_writer.writerow(row)
                    counts['test'] += 1
                
                # Progress reporting
                if (count + 1) % 10_000 == 0:
                    elapsed = time.time() - start_time
                    samples_per_sec = (count + 1) / elapsed
                    print(f"  Processed {count + 1:,} samples ({samples_per_sec:.0f} samples/sec)")
    
    total_time = time.time() - start_time
    
    print(f"\nSplitting completed in {total_time:.1f}s")
    print(f"Train samples: {counts['train']:,}")
    print(f"Val samples:   {counts['val']:,}")
    print(f"Test samples:  {counts['test']:,}")
    print(f"Total:         {sum(counts.values()):,}")
    print(f"\nFiles saved to: {SPLIT_DATA_DIR}")
if __name__ == "__main__":
    split_data()
