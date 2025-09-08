import numpy as np
import pandas as pd

df = pd.read_csv('Data/raw.csv')

# Thông tin cơ bản
print("THÔNG TIN CƠ BẢN:")
print(f"Số hàng: {df.shape[0]}")
print(f"Số cột: {df.shape[1]}")
print(f"Tên các cột: {list(df.columns)}")

# Xem dữ liệu mẫu
print("\n1 VÀI DỮ LIỆU MẪU:")
print(df.head()) # Hiển thị 5 dòng đầu tiên

# Thông tin chi tiết
print("\nTHÔNG TIN CHI TIẾT:")
print(df.info())

# Kiểm tra giá trị null
print("\nKIỂM TRA GIÁ TRỊ NULL:")
print(df.isnull().sum())

# Phân tích nhãn
print("\nPHÂN TÍCH NHÃN:")
label_counts = df['labels'].value_counts()
print(f"Số lượng: {label_counts}")
print("\n Tỷ lệ phần trăm:")
label_percentages = df['labels'].value_counts(normalize=True)*100
for label, percentage in label_percentages.items():
    print(f"{label}: {percentage:.1f}%")

# Phân tích theo thời gian
print("\nPHÂN TÍCH THEO THỜI GIAN:")
time_counts = df['year_month'].value_counts().sort_index()
print(f"\n{time_counts.head(10)}")  # Hiển thị 10 tháng đầu tiên
print(f"\nKhoảng thời gian: {df['year_month'].min()} đến {df['year_month'].max()}")

# Phân tích độ dài tiêu đề và nội dung
print("\nPHÂN TÍCH ĐỘ DÀI TIÊU ĐỀ VÀ NỘI DUNG:")
df['text_length'] = df['text'].str.len()
df['title_length'] = df['title'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"\nĐộ dài trung bình của text: {df['text_length'].mean():.0f} ký tự")
print(f"Độ dài trung bình của title: {df['title_length'].mean():.0f} ký tự")
print(f"Số từ trung bình: {df['word_count'].mean():.0f} từ")

# So sánh theo nhãn
print("\nSO SÁNH THEO NHÃN:")
print("Độ dài text trung bình theo nhãn:")
for label in df['labels'].unique():
    avg_length = df[df['labels'] == label]['text_length'].mean()
    print(f"  {label}: {avg_length:.0f} ký tự ")

print("\nSố từ trung bình theo nhãn:")
for label in df['labels'].unique():
    avg_words = df[df['labels'] == label]['word_count'].mean()
    print(f"  {label}: {avg_words:.0f} từ")

# Tóm tắt
print("\nTÓM TẮT:")
true_count = len(df[df['labels'] == 'true'])
fake_count = len(df[df['labels'] == 'fake'])
print(f"Tổng số bài: {len(df):,}")
print(f"Tin thật: {true_count:,} ({true_count/len(df)*100:.1f}%)")
print(f"Tin giả: {fake_count:,} ({fake_count/len(df)*100:.1f}%)")
if abs(true_count - fake_count) > len(df) * 0.2:
    print("Dữ liệu bị lệch nhãn.")
else:
    print("Dữ liệu cân bằng.")