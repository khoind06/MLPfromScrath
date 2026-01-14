import os
import numpy as np
import scipy.sparse as sp
from typing import Generator, Tuple
import time

class CTRDataset:
    """Dataset class cho CTR prediction với sparse matrix"""
    
    def __init__(self, X_path: str, y_path: str, name: str = "dataset"):
        self.name = name
        print(f"Loading {name}...")
        start_time = time.time()
        
        # Load dữ liệu
        self.X = sp.load_npz(X_path)
        self.y = np.load(y_path)
        
        # Chuyển về CSR format để slice nhanh hơn
        if not sp.isspmatrix_csr(self.X):
            self.X = self.X.tocsr()
            
        load_time = time.time() - start_time
        print(f"  Loaded {self.X.shape[0]:,} samples with {self.X.shape[1]:,} features in {load_time:.2f}s")
        print(f"  Positive ratio: {self.y.mean():.4f}")
        
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Lấy single sample"""
        return self.X[idx], self.y[idx]
    
    def batch_iterator(self, batch_size: int, shuffle: bool = True) -> Generator:
        """
        Generator cho batch training
        Args:
            batch_size: Kích thước batch
            shuffle: Có shuffle không
        Yields:
            Tuple của (X_batch, y_batch)
        """
        n_samples = len(self)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Lấy batch từ sparse matrix
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices].astype(np.float32)
            
            yield X_batch, y_batch
    
    def get_stats(self) -> dict:
        """Trả về statistics của dataset"""
        return {
            "n_samples": len(self),
            "n_features": self.X.shape[1],
            "positive_ratio": self.y.mean(),
            "sparsity": self.X.nnz / (self.X.shape[0] * self.X.shape[1]),
            "memory_MB": self.X.data.nbytes / (1024 * 1024)
        }