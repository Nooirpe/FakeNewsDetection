import pandas as pd
import numpy as np

print("TIỀN XỬ LÝ DỮ LIỆU FAKE NEWS DETECTION")
print("="*50)

# 1. Đọc dữ liệu
try:
    df = pd.read_csv('Data/raw.csv')
    print(f"Đọc thành công {len(df)} bản ghi")
except FileNotFoundError:
        print("Không tìm thấy file dữ liệu!")
        exit()

print(f"Kích thước ban đầu: {df.shape}")

# 2. Định nghĩa stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
    'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}

def clean_text(text):
    # Xử lý giá trị null
    if pd.isna(text):
        return ""
    
    # Chuyển về chữ thường
    text = str(text).lower()
    
    # Loại bỏ ký tự xuống dòng và tab
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    
    # Loại bỏ khoảng trắng thừa
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    text = text.strip()
    return text

def remove_stopwords(text):
    if not text:
        return ""
    
    words = text.split()
    filtered_words = []
    
    for word in words:
        # Chỉ giữ những từ không phải stopword và có độ dài > 2
        if word not in STOPWORDS and len(word) > 2:
            # Loại bỏ từ chỉ chứa số hoặc ký tự đặc biệt
            if word.isalpha():
                filtered_words.append(word)
    
    return ' '.join(filtered_words)

def simple_stemming(text):
    if not text:
        return ""
    
    words = text.split()
    stemmed_words = []
    
    for word in words:
        # Stemming đơn giản bằng cách loại bỏ hậu tố
        if len(word) > 4:
            if word.endswith('ing') and len(word) > 6:
                word = word[:-3]
            elif word.endswith('ed') and len(word) > 5:
                word = word[:-2]
            elif word.endswith('er') and len(word) > 5:
                word = word[:-2]
            elif word.endswith('ly') and len(word) > 5:
                word = word[:-2]
            elif word.endswith('tion') and len(word) > 7:
                word = word[:-4]
            elif word.endswith('ness') and len(word) > 7:
                word = word[:-4]
            elif word.endswith('ment') and len(word) > 7:
                word = word[:-4]
            elif word.endswith('s') and len(word) > 4:
                word = word[:-1]
        
        stemmed_words.append(word)
    
    return ' '.join(stemmed_words)

# 3. Làm sạch văn bản
print("\nLÀM SẠCH VĂN BẢN:")
df_processed = df.copy()

print("Xử lý cột title...")
df_processed['title_cleaned'] = df_processed['title'].apply(clean_text)
df_processed['title_cleaned'] = df_processed['title_cleaned'].apply(remove_stopwords)
df_processed['title_cleaned'] = df_processed['title_cleaned'].apply(simple_stemming)

print("Xử lý cột text...")
df_processed['text_cleaned'] = df_processed['text'].apply(clean_text)
df_processed['text_cleaned'] = df_processed['text_cleaned'].apply(remove_stopwords)
df_processed['text_cleaned'] = df_processed['text_cleaned'].apply(simple_stemming)

# 4. Xử lý cột thời gian
print("\nXỬ LÝ THỜI GIAN:")
# Sử dụng pandas str để tách year_month
df_processed['year'] = df_processed['year_month'].str[:4].astype(int)
df_processed['month'] = df_processed['year_month'].str[5:7].astype(int)
print(f"Đã tách year_month thành year ({df_processed['year'].min()}-{df_processed['year'].max()}) và month")

# 5. Xử lý nhãn
print("\nXỬ LÝ NHÃN:")
print(f"Nhãn ban đầu: {df_processed['labels'].value_counts().to_dict()}")

# Sử dụng pandas map để chuyển đổi nhãn
label_mapping = {'fake': 0, 'true': 1}
df_processed['label_binary'] = df_processed['labels'].map(label_mapping)

print(f"Nhãn sau chuyển đổi: {df_processed['label_binary'].value_counts().to_dict()}")

# 6. Ghép title và text
print("\nGHÉP TITLE VÀ TEXT:")
df_processed['combined_text'] = df_processed['title_cleaned'] + ' ' + df_processed['text_cleaned']

# Loại bỏ khoảng trắng thừa bằng pandas
df_processed['combined_text'] = df_processed['combined_text'].str.strip()

avg_length = df_processed['combined_text'].str.len().mean()
print(f"Độ dài trung bình sau khi ghép: {avg_length:.0f} ký tự")

# 7. Tạo DataFrame cuối cùng
print("\nTẠO DATAFRAME CUỐI CÙNG:")
final_columns = ['combined_text', 'year', 'month', 'label_binary']
df_final = df_processed[final_columns].copy()

# Loại bỏ văn bản quá ngắn sử dụng pandas
initial_count = len(df_final)
df_final = df_final[df_final['combined_text'].str.len() > 10]
final_count = len(df_final)
print(f"Số bản ghi: {initial_count} -> {final_count}")

# 8. Thống kê chất lượng
print("\nTHỐNG KÊ CHẤT LƯỢNG:")
label_dist = df_final['label_binary'].value_counts()
for label, count in label_dist.items():
    label_name = "Real" if label == 1 else "Fake"
    print(f"{label_name} ({label}): {count} ({count/len(df_final)*100:.1f}%)")

text_lengths = df_final['combined_text'].str.len()
print(f"\nĐộ dài văn bản:")
print(f"Trung bình: {text_lengths.mean():.0f} ký tự")
print(f"Min-Max: {text_lengths.min()}-{text_lengths.max()} ký tự")

# 9. Lưu kết quả
print("\nLƯU KẾT QUẢ:")
try:
    output_file = '../Data/preprocessed_data.csv'
    df_final.to_csv(output_file, index=False)
    print(f"Đã lưu vào: {output_file}")
except:
    output_file = 'preprocessed_data.csv'
    df_final.to_csv(output_file, index=False)
    print(f"Đã lưu vào: {output_file}")

# 10. Hiển thị mẫu
print("\nMẪU DỮ LIỆU SAU XỬ LÝ:")
print(df_final.head(3))
