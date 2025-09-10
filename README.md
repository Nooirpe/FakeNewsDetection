# Fake News Detection - Machine Learning Project

## 📋 Tổng quan dự án

Dự án **Fake News Detection** sử dụng Machine Learning để phân loại tin tức thật và tin giả. Hệ thống được xây dựng với kiến trúc modular, sử dụng các thuật toán ML tiên tiến và đánh giá bằng Cross-Validation để đảm bảo độ tin cậy cao.

## 🏗️ Kiến trúc dự án

```
📦 Machine learning/
├── 📂 Data/                           # Dữ liệu
│   ├── raw.csv                        # Dữ liệu gốc
│   └── preprocessed_data.csv          # Dữ liệu đã xử lý
├── 📂 src/                            # Source code
│   ├── EDA.py                         # Exploratory Data Analysis
│   ├── Data_Preprocessing.py          # Tiền xử lý dữ liệu
│   ├── Model_Comparison.py            # So sánh các models
│   ├── Hyperparameter_Tuning.py      # Tối ưu hyperparameters
│   └── Utils.py                       # Utility functions
├── 📂 Model/                          # Models đã train
│   ├── Best_*.pkl                     # Models tốt nhất
│   ├── TF_IDF_Vectorizer.pkl         # TF-IDF vectorizer
│   ├── confusion_matrices.png        # Confusion matrices
│   └── Best_Parameters_*.txt         # Parameters tối ưu
├── 📂 Report/                         # Báo cáo
│   └── BÁO CÁO FAKE NEW DETECTIONS.pdf
├── README.md                          # Tài liệu hướng dẫn
```

## 🚀 Quy trình Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)

- **File**: `src/EDA.py`
- **Chức năng**:
  - Phân tích phân bố nhãn (Real vs Fake)
  - Thống kê độ dài văn bản
  - Phân tích theo thời gian
  - Visualization tổng quan dataset
- **Kết quả**: Hiểu rõ đặc tính dữ liệu, phát hiện class imbalance, xu hướng theo thời gian

### 2. Data Preprocessing

- **File**: `src/Data_Preprocessing.py`
- **Các bước xử lý**:
  - **Text Cleaning**: Loại bỏ ký tự đặc biệt, chuẩn hóa lowercase
  - **Stopwords Removal**: Loại bỏ 60+ stopwords tiếng Anh
  - **Simple Stemming**: Cắt hậu tố (ing, ed, er, ly, tion, ness, ment)
  - **Text Combination**: Ghép title + text thành combined_text
  - **Label Encoding**: fake→0, true→1
- **Visualization**: So sánh độ dài trước/sau xử lý, top words frequency

### 3. Model Comparison

- **File**: `src/Model_Comparison.py`
- **Models được so sánh**:

  1. **Logistic Regression** - Baseline linear model
  2. **Random Forest** - Ensemble method
  3. **Naive Bayes** - Probabilistic classifier
  4. **Linear SVC** - Support Vector Machine

- **Feature Engineering**:

  - **TF-IDF Vectorization**: max_features=10000, ngram_range=(1,2)
  - **Stop words filtering**: English stopwords
  - **Document frequency filtering**: min_df=2, max_df=0.95

- **Evaluation Strategy**:
  - **5-Fold Stratified Cross-Validation**
  - **Metrics**: Accuracy, Precision, Recall, F1-score
  - **Confusion Matrix**: Sử dụng cross_val_predict
  - **Overall Ranking**: Weighted scoring system

### 4. Hyperparameter Tuning

- **File**: `src/Hyperparameter_Tuning.py`
- **Tối ưu hóa**:

  - **Pipeline Optimization**: TF-IDF + Classifier
  - **GridSearchCV / RandomizedSearchCV**
  - **Cross-Validation**: 5-fold stratified
  - **Scoring**: F1-score (tốt cho imbalanced data)

- **Overfitting Detection**:
  - **Learning Curves**: Training vs Validation scores
  - **Validation Curves**: Parameter sensitivity analysis
  - **Train-Val Gap Analysis**: Phát hiện overfitting sớm

### 5. Model Evaluation

- **Cross-Validation Results**: Đánh giá robust trên multiple folds
- **Confusion Matrix Visualization**: Phân tích chi tiết classification errors
- **Performance Metrics**: Comprehensive scoring across all metrics
- **Overfitting Analysis**: Đảm bảo model generalize tốt

## 🛠️ Công nghệ sử dụng

### Core Libraries

- **pandas**: Data manipulation và analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib + seaborn**: Data visualization

### Machine Learning Stack

- **TfidfVectorizer**: Text feature extraction
- **StratifiedKFold**: Cross-validation strategy
- **RandomizedSearchCV**: Hyperparameter optimization
- **Pipeline**: ML workflow management

### Models & Algorithms

- **Logistic Regression**: Linear classification với regularization
- **Random Forest**: Ensemble bagging method
- **Multinomial Naive Bayes**: Probabilistic text classifier
- **Linear SVC**: Support Vector Classification

## 📊 Kết quả và Performance

### Dataset Statistics

- **Tổng samples**: ~8400+ tin tức
- **Features**: Combined title + text sau preprocessing
- **Labels**: Binary classification (0=Fake, 1=Real)
- **Time range**: Multi-year dataset với temporal analysis

### Model Performance (Cross-Validation)

- **Best Model**: Được chọn dựa trên weighted overall score
- **Evaluation Metrics**:
  - Accuracy: 95-98%
  - F1-Score: 97-99%
  - Precision & Recall: Balanced performance
- **Training Time**: < 10 seconds cho most models
- **Prediction Time**: < 1 second

### Key Features

- **Cross-Validation**: 5-fold stratified để đánh giá robust
- **Overfitting Detection**: Learning curves analysis
- **Confusion Matrix**: Visual analysis của classification errors
- **Hyperparameter Tuning**: Automated optimization
- **Model Persistence**: Lưu models và parameters

## 🚀 Hướng dẫn sử dụng

### Yêu cầu hệ thống

```bash
Python 3.7+
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
numpy>=1.21.0
```

### Chạy pipeline hoàn chỉnh

1. **Exploratory Data Analysis**:

```bash
cd src
python EDA.py
```

2. **Data Preprocessing**:

```bash
python Data_Preprocessing.py
```

3. **Model Comparison**:

```bash
python Model_Comparison.py
```

4. **Hyperparameter Tuning**:

```bash
python Hyperparameter_Tuning.py
```

### Sử dụng trained model

```python
import joblib
import pandas as pd

# Load model và vectorizer
model = joblib.load('Model/Best_Model_Name.pkl')
vectorizer = joblib.load('Model/TF_IDF_Vectorizer.pkl')

# Predict new text
new_text = ["This is a news article to classify..."]
X_new = vectorizer.transform(new_text)
prediction = model.predict(X_new)
probability = model.predict_proba(X_new)

print(f"Prediction: {'Real' if prediction[0] == 1 else 'Fake'}")
print(f"Confidence: {max(probability[0]):.2f}")
```

### Development Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run individual modules to test functionality
