
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_preprocessed_data(data_path="Data/preprocessed_data.csv"):
    """
    Load dữ liệu đã được tiền xử lý
    
    Returns:
        X, y: Features và labels, hoặc None, None nếu có lỗi
    """
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file {data_path}")
        return None, None
    
    df = pd.read_csv(data_path)
    print(f"Load thành công! Shape: {df.shape}")
    
    # Kiểm tra các cột cần thiết
    required_columns = ['combined_text', 'label_binary']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Thiếu các cột cần thiết: {missing_columns}")
        return None, None
    
    # Loại bỏ null
    df = df.dropna(subset=required_columns)
    print(f"Phân bố nhãn:\n{df['label_binary'].value_counts()}")
    
    X = df['combined_text']
    y = df['label_binary']
    
    return X, y

def print_data_info(X, y, dataset_name="Dataset"):
    """
    In thông tin tổng quan về dataset
    
    Args:
        X: Features
        y: Labels
        dataset_name: Tên dataset để hiển thị
    """
    print(f"\n=== {dataset_name.upper()} INFO ===")
    print(f"Số samples: {len(X):,}")
    print(f"Phân bố nhãn:")
    
    label_counts = pd.Series(y).value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "Real" if label == 1 else "Fake"
        percentage = count / len(y) * 100
        print(f"  {label_name} ({label}): {count:,} ({percentage:.1f}%)")
    
    # Kiểm tra class imbalance
    imbalance_ratio = label_counts.max() / label_counts.min()
    if imbalance_ratio > 1.5:
        print(f"⚠️  Cảnh báo: Dữ liệu mất cân bằng (tỷ lệ {imbalance_ratio:.2f}:1)")
    else:
        print("✅ Dữ liệu cân bằng tốt")

def save_model_info(model_name, model_params, save_path):
    """
    Lưu thông tin model vào file text
    
    Args:
        model_name: Tên model
        model_params: Parameters của model
        save_path: Đường dẫn file để lưu
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write("="*50 + "\n\n")
        f.write("Model Parameters:\n")
        for param, value in model_params.items():
            f.write(f"  {param}: {value}\n")
    
    print(f"Đã lưu thông tin model: {save_path}")

def create_model_directory():
    """Tạo thư mục Model nếu chưa tồn tại"""
    model_dir = "Model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Đã tạo thư mục: {model_dir}")
    return model_dir

if __name__ == "__main__":
    # Test các functions
    print("Testing utils.py...")
    
    # Test load data
    X, y = load_preprocessed_data()
    if X is not None:
        print_data_info(X, y, "Original Data")

    print("\nUtils test completed!")
