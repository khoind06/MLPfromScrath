# feature_engineering.py
import os
import numpy as np
import pandas as pd
from scipy import sparse
import hashlib
import time
import warnings
warnings.filterwarnings('ignore')

from config import (
    PROCESSED_DATA_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    LABEL_COLUMN, HASHING_DIM, CROSS_HASH_DIM, CROSS_FEATURES,
    NUM_BINS, MIN_CATEGORY_FREQ
)

def stable_hash(feature_name: str, feature_value: str, hashing_dim: int) -> int:
    key = f"{feature_name}={feature_value}".encode('utf-8')
    return int(hashlib.md5(key).hexdigest(), 16) % hashing_dim

def process_dataframe_advanced(df: pd.DataFrame):
    # 1. Log Transform cho các cột số bị lệch (Skewed)
    skewed_cols = ['I2', 'I5', 'I7', 'I9'] 
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            # Fill NA bằng 0
            df[col] = df[col].fillna(0)
            if col in skewed_cols:
                df[col] = np.log1p(df[col])

    # 2. Smart Binning (Quantile Cut)
    bin_cols = []
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            try:
                # Dùng qcut để chia đều dữ liệu vào các giỏ
                # duplicates='drop' để xử lý trường hợp quá nhiều số 0
                new_col = f"{col}_bin"
                df[new_col] = pd.qcut(df[col], q=NUM_BINS, labels=False, duplicates='drop').astype(str)
                # Thêm tiền tố để tránh trùng hash
                df[new_col] = f"{col}_" + df[new_col]
                bin_cols.append(new_col)
            except ValueError:
                continue

    # 3. Frequency Filtering (Chỉ tính trên Train, áp dụng cho Val/Test)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('missing').astype(str)
            
    return df, bin_cols

def encode_categorical_fast(df: pd.DataFrame) -> sparse.csr_matrix:
    print("  > Hashing Features...")
    start_time = time.time()
    
    # Tiền xử lý (Log + Binning)
    df, bin_cols = process_dataframe_advanced(df)
    
    n_samples = len(df)
    rows, cols, data = [], [], []
    
    # Danh sách tất cả các cột cần Hash (Gốc + Binned)
    features_to_hash = CATEGORICAL_FEATURES + bin_cols
    
    # Lặp qua từng dòng để tạo Sparse Matrix
    for i, row in enumerate(df.itertuples(index=False)):
        row_dict = dict(zip(df.columns, row))
        
        # Hash Feature đơn
        for col in features_to_hash:
            val = str(row_dict.get(col, 'missing'))
            h_idx = stable_hash(col, val, HASHING_DIM)
            rows.append(i)
            cols.append(h_idx)
            data.append(1.0)
            
    X_main = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, HASHING_DIM))
    
    # Cross Features 
    rows_c, cols_c, data_c = [], [], []
    for i, row in enumerate(df.itertuples(index=False)):
        row_dict = dict(zip(df.columns, row))
        for (f1, f2) in CROSS_FEATURES:
            v1 = str(row_dict.get(f1, ''))
            v2 = str(row_dict.get(f2, ''))
            h_idx = stable_hash(f"{f1}_x_{f2}", f"{v1}x{v2}", CROSS_HASH_DIM)
            rows_c.append(i)
            cols_c.append(h_idx)
            data_c.append(1.0)
            
    X_cross = sparse.csr_matrix((data_c, (rows_c, cols_c)), shape=(n_samples, CROSS_HASH_DIM))
    
    # Add dense numeric features
    X_num = df[NUMERIC_FEATURES].fillna(0).values.astype(np.float32)
    
    X_final = sparse.hstack([X_main, X_cross, X_num]).tocsr()
    print(f"  > Encoded shape: {X_final.shape} in {time.time() - start_time:.2f}s")
    return X_final

def run_feature_engineering():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    from config import SPLIT_DATA_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE
    
    for name, fname in [('train', TRAIN_FILE), ('val', VAL_FILE), ('test', TEST_FILE)]:
        print(f"\nProcessing {name} set...")
        path = os.path.join(SPLIT_DATA_DIR, fname)
        if not os.path.exists(path): continue
        
        df = pd.read_csv(path, names=[LABEL_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        y = df[LABEL_COLUMN].values.astype(np.float32)
        X = encode_categorical_fast(df)
        
        sparse.save_npz(os.path.join(PROCESSED_DATA_DIR, f"{name}_X.npz"), X)
        np.save(os.path.join(PROCESSED_DATA_DIR, f"{name}_y.npy"), y)
        print(f"Saved {name} data.")

if __name__ == "__main__":
    run_feature_engineering()