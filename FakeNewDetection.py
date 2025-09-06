import pandas as pd
import numpy as np

print("="*50)
print("    PHÂN TÍCH DỮ LIỆU FAKE NEWS - CƠ BẢN")
print("="*50)

# Đọc dữ liệu
df = pd.read_csv('test.csv')

# 1. THÔNG TIN CƠ BẢN
print("\n1. THÔNG TIN CƠ BẢN:")
print(f"Số hàng: {df.shape[0]}")
print(f"Số cột: {df.shape[1]}")
print(f"Tên các cột: {list(df.columns)}")

# 2. XEM DỮ LIỆU MẪU
print("\n2. DỮ LIỆU MẪU (5 dòng đầu):")
print(df.head())

# 3. THÔNG TIN CHI TIẾT
print("\n3. THÔNG TIN VỀ CÁC CỘT:")
print(df.info())

# 4. KIỂM TRA GIÁ TRỊ NULL
print("\n4. KIỂM TRA GIÁ TRỊ THIẾU:")
null_counts = df.isnull().sum()
print(null_counts)
if null_counts.sum() == 0:
    print("✅ Không có giá trị thiếu!")
else:
    print("⚠️ Có giá trị thiếu cần xử lý!")

# 5. PHÂN TÍCH NHÃN
print("\n5. PHÂN TÍCH NHÃN:")
label_counts = df['labels'].value_counts()
print("Số lượng:")
print(label_counts)
print("\nTỷ lệ phần trăm:")
label_percent = df['labels'].value_counts(normalize=True) * 100
for label, percent in label_percent.items():
    print(f"{label}: {percent:.1f}%")

# 6. PHÂN TÍCH THEO THỜI GIAN
print("\n6. PHÂN TÍCH THEO THỜI GIAN:")
time_counts = df['year_month'].value_counts().sort_index()
print("Top 10 tháng có nhiều bài nhất:")
print(time_counts.head(10))
print(f"\nKhoảng thời gian: {df['year_month'].min()} đến {df['year_month'].max()}")

# 7. PHÂN TÍCH ĐỘ DÀI VĂN BẢN
print("\n7. PHÂN TÍCH ĐỘ DÀI VĂN BẢN:")
df['text_length'] = df['text'].str.len()
df['title_length'] = df['title'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"Độ dài trung bình của text: {df['text_length'].mean():.0f} ký tự")
print(f"Độ dài trung bình của title: {df['title_length'].mean():.0f} ký tự")
print(f"Số từ trung bình: {df['word_count'].mean():.0f} từ")

# 8. SO SÁNH THEO NHÃN
print("\n8. SO SÁNH THEO NHÃN:")
print("\nĐộ dài text trung bình:")
for label in df['labels'].unique():
    avg_length = df[df['labels'] == label]['text_length'].mean()
    print(f"  {label}: {avg_length:.0f} ký tự")

print("\nSố từ trung bình:")
for label in df['labels'].unique():
    avg_words = df[df['labels'] == label]['word_count'].mean()
    print(f"  {label}: {avg_words:.0f} từ")

# 9. TÓM TẮT
print("\n" + "="*50)
print("         TÓM TẮT")
print("="*50)
true_count = len(df[df['labels'] == 'true'])
false_count = len(df[df['labels'] == 'false'])
print(f"📊 Tổng số bài: {len(df):,}")
print(f"📰 Tin thật: {true_count:,} ({true_count/len(df)*100:.1f}%)")
print(f"� Tin giả: {false_count:,} ({false_count/len(df)*100:.1f}%)")

if abs(true_count - false_count) > len(df) * 0.2:
    print("⚠️  Dataset mất cân bằng!")
else:
    print("✅ Dataset cân bằng tốt")

print("\nEDA cơ bản hoàn thành! 🎉")