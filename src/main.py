import argparse
import os
import sys

from split_data import split_data
from feature_engineering import run_feature_engineering
from train import train

def run_pipeline():
    parser = argparse.ArgumentParser(description='CTR Prediction Pipeline')
    parser.add_argument('--step', type=str, default='all',
                       choices=['split', 'features', 'train', 'all'],
                       help='Chọn bước để chạy: split, features, train hoặc all')
    
    args = parser.parse_args()
    
    # Bước 1: Chia dữ liệu (Split Data)
    if args.step in ['split', 'all']:
        split_data()
        
    # Bước 2: Tạo đặc trưng (Feature Engineering)
    if args.step in ['features', 'all']:
        print("\n2. FEATURE ENGINEERING")
        run_feature_engineering()
        
    # Bước 3: Huấn luyện (Training)
    if args.step in ['train', 'all']:
        print("\n3. TRAINING")
        train()

if __name__ == "__main__":
    run_pipeline()