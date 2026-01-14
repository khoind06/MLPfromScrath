# CTR Prediction – MLP from Scratch

Dự án này triển khai một mô hình **Deep Learning (Multilayer Perceptron – MLP)** để dự đoán **tỷ lệ nhấp chuột (CTR – Click-Through Rate)** trên dữ liệu quảng cáo theo phong cách **Criteo**.

Điểm đặc biệt của project là **toàn bộ mô hình MLP được xây dựng hoàn toàn từ scratch bằng NumPy**, bao gồm:
- Forward Pass
- Backpropagation
- Optimizer
- Các layer nâng cao  
Các thư viện bên ngoài chỉ được dùng cho **tiền xử lý dữ liệu và đánh giá metric**.

---

## Điểm nổi bật

Dự án nhằm chứng minh sự hiểu biết sâu sắc về **toán học và cơ chế hoạt động bên trong Neural Networks**, thay vì chỉ sử dụng framework có sẵn.

### Core Engine 
- Tự cài đặt **Forward Pass**
- Tự tính **Backpropagation**
- Tính gradient thủ công bằng **Chain Rule**
- Hỗ trợ dữ liệu **high-dimensional sparse**

###  Optimizer
- **Adam Optimizer** tự cài đặt
- Bao gồm đầy đủ:
  - Momentum (\(m_t\))
  - RMSProp (\(v_t\))
  - Bias correction

###  Advanced Layers
- **Batch Normalization**
  - Tự tính mean / variance
  - Backpropagation qua normalization layer
- **Dropout**
  - Regularization bằng cách ngẫu nhiên tắt neuron
- **Weight Initialization**
  - He / Xavier initialization chuẩn lý thuyết

###  Feature Engineering
- Xử lý dữ liệu **sparse với hàng trăm nghìn đến hàng triệu chiều**
- **Hashing Trick**
  - Giảm chiều không gian categorical cực lớn (lên tới \(2^{17}\) features)
- **Feature Interaction**
  - Tạo đặc trưng chéo (Cross Product) thủ công

---

##  Performance

Mô hình đạt hiệu suất **tương đương các mô hình MLP cơ bản trong thư viện công nghiệp** trên tập test:

| Metric   | Giá trị (Test Set) | Ý nghĩa |
|--------|--------------------|--------|
| AUC    | ~0.74              | Khả năng phân biệt click / non-click tốt |
| LogLoss| ~0.118             | Xác suất dự đoán sát thực tế |
| Overfitting | Không          | Test Loss ≤ Validation Loss |

Cho thấy mô hình **generalize tốt**, không bị overfitting dù dữ liệu mất cân bằng mạnh.

---

##  Project Structure

```text
.
├── data/
│   ├── raw/                # Dữ liệu gốc (Criteo-style, gitignored)
│   ├── split/              # Train / Val / Test CSV (gitignored)
│   └── processed/          # Dữ liệu đã xử lý (.npz, sparse) (gitignored)
│
├── src/
│   ├── config.py           # Cấu hình hyperparameters
│   ├── dataset.py          # Dataset loader (sparse matrix)
│   ├── feature_engineering.py  # Hashing trick & cross features
│   ├── model.py            # Core MLP from scratch (Linear, BN, Adam, ...)
│   ├── train.py            # Training loop & evaluation
│   ├── split_data.py       # Chia dữ liệu raw thành train/val/test
│   └── main.py             # Entry point để chạy pipeline
│
├── requirements.txt        # Thư viện cần thiết
└── README.md               # Tài liệu dự án
```
## Installation

### Clone repository:
```
git clone <repository_url>
```
### Cài đặt thư viện cần thiết:
```
pip install numpy pandas scipy scikit-learn
```
Data Preparation (Important)

## Do dataset Criteo có kích thước rất lớn, dữ liệu không được upload kèm repo.

### Các bước chuẩn bị:

- Tải dataset Criteo (ví dụ: day_0.gz)

- Tạo thư mục:
  data/raw
- Đặt file vào:
```
data/raw/day_0.gz
```
## How to Run
### Chạy toàn bộ pipeline:
```
python main.py
```
### Hoặc chạy từng bước riêng lẻ:
#### 1. Chia dữ liệu raw
```
python main.py --step split
```
#### 2. Feature Engineering
```
python main.py --step features
```
#### 3. Huấn luyện mô hình
```
python main.py --step train
```
