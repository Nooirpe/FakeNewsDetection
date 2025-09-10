import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Thiết lập matplotlib
plt.rcParams['figure.figsize'] = (12, 6)

print("TIỀN XỬ LÝ DỮ LIỆU FAKE NEWS DETECTION")

# Đọc dữ liệu
try:
    df = pd.read_csv('Data/raw.csv')
    print(f"Đọc thành công {len(df)} bản ghi")
except FileNotFoundError:
    try:
        df = pd.read_csv('Data/raw.csv')
        print(f"Đọc thành công {len(df)} bản ghi")
    except FileNotFoundError:
        print("Không tìm thấy file dữ liệu!")
        exit()

print(f"Kích thước ban đầu: {df.shape}")

# Định nghĩa stopwords
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
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    # Thêm các từ phổ biến khác
    'said', 'say', 'says', 'one', 'two', 'would', 'could', 'get', 'also',
    'like', 'go', 'come', 'time', 'people', 'make', 'way', 'may', 'see',
    'know', 'new', 'first', 'last', 'good', 'great', 'little', 'old', 'right',
    'back', 'still', 'well', 'take', 'think', 'even', 'much', 'us', 'according'
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

def get_top_words(text_series, top_n=20):
    """
    Lấy top N từ phổ biến từ một series text
    """
    all_words = []
    for text in text_series:
        if pd.notna(text) and text:
            words = str(text).split()
            all_words.extend(words)
    
    # Đếm từ bằng pandas
    word_freq = pd.Series(all_words).value_counts()
    return word_freq.head(top_n)

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

# Làm sạch văn bản
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

# VISUALIZATION 1: So sánh độ dài văn bản trước và sau xử lý
print("\nVISUALIZATION: SO SÁNH ĐỘ DÀI VĂN BẢN TRƯỚC VÀ SAU XỬ LÝ")

# Tính độ dài trước và sau xử lý
original_title_length = df['title'].str.len()
original_text_length = df['text'].str.len()
original_combined_length = original_title_length + original_text_length

processed_title_length = df_processed['title_cleaned'].str.len()
processed_text_length = df_processed['text_cleaned'].str.len()
processed_combined_length = processed_title_length + processed_text_length

plt.figure(figsize=(15, 10))

# Bar chart so sánh độ dài trung bình
plt.subplot(1, 2, 1)
categories = ['Title', 'Text', 'Combined']
original_means = [original_title_length.mean(), original_text_length.mean(), original_combined_length.mean()]
processed_means = [processed_title_length.mean(), processed_text_length.mean(), processed_combined_length.mean()]

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, original_means, width, label='Trước xử lý', color='lightcoral', alpha=0.8)
plt.bar(x + width/2, processed_means, width, label='Sau xử lý', color='lightblue', alpha=0.8)

plt.xlabel('Loại văn bản')
plt.ylabel('Độ dài trung bình (ký tự)')
plt.title('So sánh độ dài trung bình trước và sau xử lý')
plt.xticks(x, categories)
plt.legend()

# So sánh chung
plt.subplot(1, 2, 2)
plt.hist(original_combined_length, bins=30, alpha=0.8, color='red', label='Trước xử lý')
plt.hist(processed_combined_length, bins=30, alpha=0.8, color='lightblue', label='Sau xử lý')
plt.xlabel('Độ dài (ký tự)')
plt.ylabel('Số lượng')
plt.title('So sánh phân bố độ dài')
plt.legend()

plt.tight_layout()
plt.show()

# VISUALIZATION 2: Top 20 từ phổ biến trước và sau stopword removal
print("\nVISUALIZATION: TOP 20 TỪ PHỔ BIẾN TRƯỚC VÀ SAU STOPWORD REMOVAL")

# Lấy top từ trước khi remove stopwords (chỉ clean text)
df_before_stopword = df_processed.copy()
df_before_stopword['title_before_stopword'] = df_processed['title'].apply(clean_text)
df_before_stopword['text_before_stopword'] = df_processed['text'].apply(clean_text)
df_before_stopword['combined_before_stopword'] = df_before_stopword['title_before_stopword'] + ' ' + df_before_stopword['text_before_stopword']

top_words_before = get_top_words(df_before_stopword['combined_before_stopword'])
top_words_after = get_top_words(df_processed['title_cleaned'] + ' ' + df_processed['text_cleaned'])

plt.figure(figsize=(15, 8))

# Top từ trước khi remove stopwords
plt.subplot(1, 2, 1)
plt.barh(range(len(top_words_before)), top_words_before.values, color='lightcoral', alpha=0.8)
plt.yticks(range(len(top_words_before)), top_words_before.index)
plt.xlabel('Tần suất')
plt.title('Top 20 từ phổ biến - Trước remove stopwords')
plt.gca().invert_yaxis()

# Top từ sau khi remove stopwords
plt.subplot(1, 2, 2)
plt.barh(range(len(top_words_after)), top_words_after.values, color='lightblue', alpha=0.8)
plt.yticks(range(len(top_words_after)), top_words_after.index)
plt.xlabel('Tần suất')
plt.title('Top 20 từ phổ biến - Sau remove stopwords')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

print(f"Top 5 từ trước remove stopwords: {list(top_words_before.head().index)}")
print(f"Top 5 từ sau remove stopwords: {list(top_words_after.head().index)}")

# Xử lý cột thời gian
print("\nXỬ LÝ THỜI GIAN:")
# Sử dụng pandas str để tách year_month
df_processed['year'] = df_processed['year_month'].str[:4].astype(int)
df_processed['month'] = df_processed['year_month'].str[5:7].astype(int)
print(f"Đã tách year_month thành year ({df_processed['year'].min()}-{df_processed['year'].max()}) và month")

# Xử lý nhãn
print("\nXỬ LÝ NHÃN:")
print(f"Nhãn ban đầu: {df_processed['labels'].value_counts().to_dict()}")

# Sử dụng pandas map để chuyển đổi nhãn
label_mapping = {'fake': 0, 'true': 1}
df_processed['label_binary'] = df_processed['labels'].map(label_mapping)

print(f"Nhãn sau chuyển đổi: {df_processed['label_binary'].value_counts().to_dict()}")
# Ghép title và text
print("\nGHÉP TITLE VÀ TEXT:")
df_processed['combined_text'] = df_processed['title_cleaned'] + ' ' + df_processed['text_cleaned']

# Loại bỏ khoảng trắng thừa bằng pandas
df_processed['combined_text'] = df_processed['combined_text'].str.strip()

avg_length = df_processed['combined_text'].str.len().mean()
print(f"Độ dài trung bình sau khi ghép: {avg_length:.0f} ký tự")

# Tạo DataFrame cuối cùng
print("\nTẠO DATAFRAME CUỐI CÙNG:")
final_columns = ['combined_text', 'year', 'month', 'label_binary']
df_final = df_processed[final_columns].copy()

# Loại bỏ văn bản quá ngắn sử dụng pandas
initial_count = len(df_final)
df_final = df_final[df_final['combined_text'].str.len() > 10]
final_count = len(df_final)
print(f"Số bản ghi: {initial_count} -> {final_count}")

# Thống kê chất lượng
print("\nTHỐNG KÊ CHẤT LƯỢNG:")
label_dist = df_final['label_binary'].value_counts()
for label, count in label_dist.items():
    label_name = "Real" if label == 1 else "Fake"
    print(f"{label_name} ({label}): {count} ({count/len(df_final)*100:.1f}%)")

text_lengths = df_final['combined_text'].str.len()
print(f"\nĐộ dài văn bản:")
print(f"Trung bình: {text_lengths.mean():.0f} ký tự")
print(f"Min-Max: {text_lengths.min()}-{text_lengths.max()} ký tự")

# Lưu kết quả
print("\nLƯU KẾT QUẢ:")
try:
    output_file = '../Data/preprocessed_data.csv'
    df_final.to_csv(output_file, index=False)
    print(f"Đã lưu vào: {output_file}")
except:
    output_file = 'preprocessed_data.csv'
    df_final.to_csv(output_file, index=False)
    print(f"Đã lưu vào: {output_file}")

# Hiển thị mẫu
print("\nMẪU DỮ LIỆU SAU XỬ LÝ:")
print(df_final.head(3))

print("\n" + "="*50)
print("HOÀN THÀNH TIỀN XỬ LÝ!")
print(f"Kết quả: {len(df_final)} bản ghi sẵn sàng cho Feature Extraction")

# VISUALIZATION TỔNG KẾT
print("\nVISUALIZATION TỔNG KẾT:")

plt.figure(figsize=(15, 8))

# Tóm tắt độ giảm độ dài
plt.subplot(2, 3, 1)
reduction_percent = ((original_combined_length.mean() - processed_combined_length.mean()) / original_combined_length.mean()) * 100
plt.bar(['Trước xử lý', 'Sau xử lý'], 
        [original_combined_length.mean(), processed_combined_length.mean()],
        color=['lightcoral', 'lightblue'], alpha=0.8)
plt.title(f'Giảm độ dài: {reduction_percent:.1f}%')
plt.ylabel('Độ dài trung bình')

# Phân bố nhãn cuối cùng
plt.subplot(2, 3, 2)
final_labels = df_final['label_binary'].value_counts().sort_index()
plt.pie(final_labels.values, labels=['Fake (0)', 'Real (1)'], 
        autopct='%1.1f%%', colors=['red', 'green'], startangle=90)
plt.title('Phân bố nhãn cuối cùng')

# Số lượng từ trung bình sau xử lý
plt.subplot(2, 3, 3)
word_counts_final = df_final['combined_text'].str.split().str.len()
plt.hist(word_counts_final, bins=30, color='lightgreen', alpha=0.7)
plt.xlabel('Số từ')
plt.ylabel('Số lượng')
plt.title(f'Phân bố số từ\n(TB: {word_counts_final.mean():.0f} từ)')

# So sánh top 10 từ sau xử lý cuối cùng
plt.subplot(2, 3, 4)
top_words_final = get_top_words(df_final['combined_text'], top_n=10)
plt.barh(range(len(top_words_final)), top_words_final.values, color='purple', alpha=0.7)
plt.yticks(range(len(top_words_final)), top_words_final.index)
plt.xlabel('Tần suất')
plt.title('Top 10 từ sau xử lý hoàn chỉnh')
plt.gca().invert_yaxis()

# Phân bố độ dài theo nhãn
plt.subplot(2, 3, 5)
fake_lengths = df_final[df_final['label_binary'] == 0]['combined_text'].str.len()
real_lengths = df_final[df_final['label_binary'] == 1]['combined_text'].str.len()
plt.hist([fake_lengths, real_lengths], bins=30, alpha=0.7, 
         label=['Fake', 'Real'], color=['red', 'green'])
plt.xlabel('Độ dài văn bản')
plt.ylabel('Số lượng')
plt.title('Độ dài theo nhãn')
plt.legend()

# Thống kê tổng quan
plt.subplot(2, 3, 6)
stats_data = [
    len(df),
    len(df_final),
    original_combined_length.mean(),
    processed_combined_length.mean(),
    word_counts_final.mean()
]
stats_labels = ['Bản ghi\ngốc', 'Bản ghi\ncuối', 'Độ dài\ngốc', 'Độ dài\nsau', 'Số từ\nTB']
colors = ['lightblue', 'blue', 'lightcoral', 'red', 'lightgreen']

# Chuẩn hóa để vẽ cùng một biểu đồ
stats_normalized = [(x/max(stats_data))*100 for x in stats_data]
bars = plt.bar(stats_labels, stats_normalized, color=colors, alpha=0.7)

# Thêm giá trị thực lên cột
for i, (bar, value) in enumerate(zip(bars, stats_data)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{value:.0f}', ha='center', va='bottom', fontsize=8)

plt.title('Thống kê tổng quan')
plt.ylabel('Giá trị chuẩn hóa (%)')

plt.tight_layout()
plt.show()
