import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix
import time
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import load_preprocessed_data

def vectorize_text(X):
    """TF-IDF Vectorization cho toàn bộ dataset"""
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        max_df=0.95,
        min_df=2
    )
    
    print("Đang vectorize text...")
    X_tfidf = tfidf.fit_transform(X)
    
    print(f"Features shape: {X_tfidf.shape}")
    
    return X_tfidf, tfidf

def create_models():
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'Naive Bayes': MultinomialNB(
            alpha=1.0
        ),
        'Linear SVC': LinearSVC(
            random_state=42,
            max_iter=5000,
            dual=False,  # Tốt hơn cho sparse features
            C=1.0
        )
    }
    
    return models

def evaluate_model_cv(name, model, X_tfidf, y):
    """Đánh giá model bằng Cross-Validation (5-fold)"""
    print(f"\n{name}")
    
    # Tạo StratifiedKFold để đảm bảo phân bố nhãn đều
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Định nghĩa các scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall', 
        'f1': 'f1'
    }
    
    # Đo thời gian training
    start_time = time.time()
    
    # Cross-validation với nhiều metrics
    cv_results = cross_validate(
        model, X_tfidf, y, 
        cv=cv, 
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    train_time = time.time() - start_time
    
    # Tính toán các metrics trung bình
    accuracy_mean = cv_results['test_accuracy'].mean()
    accuracy_std = cv_results['test_accuracy'].std()
    
    precision_mean = cv_results['test_precision'].mean()
    precision_std = cv_results['test_precision'].std()
    
    recall_mean = cv_results['test_recall'].mean()
    recall_std = cv_results['test_recall'].std()
    
    f1_mean = cv_results['test_f1'].mean()
    f1_std = cv_results['test_f1'].std()
    
    # Đo thời gian prediction (fit 1 lần và predict)
    start_time = time.time()
    model.fit(X_tfidf, y)
    y_pred = model.predict(X_tfidf)
    pred_time = time.time() - start_time
    
    results = {
        'Model': name,
        'Train_Time': train_time,
        'Pred_Time': pred_time,
        'Accuracy_Mean': accuracy_mean,
        'Accuracy_Std': accuracy_std,
        'Precision_Mean': precision_mean,
        'Precision_Std': precision_std,
        'Recall_Mean': recall_mean,
        'Recall_Std': recall_std,
        'F1_Mean': f1_mean,
        'F1_Std': f1_std
    }
    
    print(f"Training time: {train_time:.3f}s")
    print(f"Prediction time: {pred_time:.3f}s")
    print(f"Accuracy: {accuracy_mean:.4f} (+/- {accuracy_std*2:.4f})")
    print(f"Precision: {precision_mean:.4f} (+/- {precision_std*2:.4f})")
    print(f"Recall: {recall_mean:.4f} (+/- {recall_std*2:.4f})")
    print(f"F1-score: {f1_mean:.4f} (+/- {f1_std*2:.4f})")
    
    return results, model

def compare_models(results_list):
    """So sánh kết quả các models với cross-validation"""
    print("\nSO SÁNH CÁC MODELS (Cross-Validation)")
    
    # Tạo DataFrame để dễ so sánh
    df_results = pd.DataFrame(results_list)
    
    print("\nBảng so sánh chi tiết:")
    display_columns = ['Model', 'Accuracy_Mean', 'Accuracy_Std', 'F1_Mean', 'F1_Std', 
                      'Precision_Mean', 'Recall_Mean', 'Train_Time']
    print(df_results[display_columns].round(4).to_string(index=False))
    
    # Tìm model tốt nhất cho từng metric
    print(f"\n--- XẾP HẠNG ---")
    print(f"Fastest Training: {df_results.loc[df_results['Train_Time'].idxmin(), 'Model']}")
    print(f"Fastest Prediction: {df_results.loc[df_results['Pred_Time'].idxmin(), 'Model']}")
    print(f"Best Accuracy: {df_results.loc[df_results['Accuracy_Mean'].idxmax(), 'Model']}")
    print(f"Best F1-Score: {df_results.loc[df_results['F1_Mean'].idxmax(), 'Model']}")
    print(f"Best Precision: {df_results.loc[df_results['Precision_Mean'].idxmax(), 'Model']}")
    print(f"Best Recall: {df_results.loc[df_results['Recall_Mean'].idxmax(), 'Model']}")
    print(f"Most Stable (Accuracy): {df_results.loc[df_results['Accuracy_Std'].idxmin(), 'Model']}")
    
    # Tính overall score (điều chỉnh cho cross-validation)
    df_results['Overall_Score'] = (
        df_results['Accuracy_Mean'] * 0.3 +     # 30% accuracy
        df_results['F1_Mean'] * 0.3 +           # 30% f1-score  
        df_results['Precision_Mean'] * 0.15 +   # 15% precision
        df_results['Recall_Mean'] * 0.15 +      # 15% recall
        (1 - df_results['Train_Time']/df_results['Train_Time'].max()) * 0.05 +  # 5% speed
        (1 - df_results['Accuracy_Std']) * 0.05  # 5% stability
    )
    
    best_model_idx = df_results['Overall_Score'].idxmax()
    best_model_name = df_results.loc[best_model_idx, 'Model']
    
    print(f"\n*** MODEL ĐƯỢC KHUYẾN NGHỊ: {best_model_name} ***")
    print(f"Overall Score: {df_results.loc[best_model_idx, 'Overall_Score']:.4f}")
    print(f"Accuracy: {df_results.loc[best_model_idx, 'Accuracy_Mean']:.4f} (+/- {df_results.loc[best_model_idx, 'Accuracy_Std']*2:.4f})")
    print(f"F1-Score: {df_results.loc[best_model_idx, 'F1_Mean']:.4f} (+/- {df_results.loc[best_model_idx, 'F1_Std']*2:.4f})")
    
    return best_model_name, df_results

def plot_confusion_matrices(models, X_tfidf, y):
    """Vẽ confusion matrix cho tất cả các models với cross-validation"""
    from sklearn.model_selection import cross_val_predict
    
    plt.figure(figsize=(15, 10))
    
    # Sử dụng cùng CV strategy như trong evaluate_model_cv
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (name, model) in enumerate(models.items(), 1):
        plt.subplot(2, 2, i)
        
        # Sử dụng cross_val_predict để get predictions từ cross-validation
        y_pred = cross_val_predict(model, X_tfidf, y, cv=cv)
        
        # Tính confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Vẽ confusion matrix với seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        
        plt.title(f'{name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Thêm accuracy vào title (CV accuracy)
        accuracy = accuracy_score(y, y_pred)
        plt.title(f'{name}\nCV Accuracy: {accuracy:.4f}')
    
    plt.tight_layout()
    
    # Tạo folder Model nếu chưa có
    if not os.path.exists("Model"):
        os.makedirs("Model")
    
    plt.savefig('Model/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Đã lưu confusion matrices tại: Model/confusion_matrices.png")
    print("Confusion matrix được tính từ cross-validation predictions")

def save_best_model(best_model_name, models, tfidf):
    """Lưu model tốt nhất"""
    if not os.path.exists("Model"):
        os.makedirs("Model")
    
    # Lưu model
    model_path = f"Model/Best_{best_model_name.replace(' ', '_')}.pkl"
    joblib.dump(models[best_model_name], model_path)
    
    # Lưu vectorizer
    vectorizer_path = "Model/TF_IDF_Vectorizer.pkl"
    joblib.dump(tfidf, vectorizer_path)
    
    print(f"\nĐã lưu model tốt nhất: {model_path}")
    print(f"Đã lưu vectorizer: {vectorizer_path}")

def main():
    print("=== MODEL COMPARISON - FAKE NEWS DETECTION (Cross-Validation) ===")
    
    # Load dữ liệu
    print("\n1. Load dữ liệu...")
    X, y = load_preprocessed_data()
    if X is None:
        return
    
    # Vectorize text
    print("\n2. Vectorize text...")
    X_tfidf, tfidf = vectorize_text(X)
    
    # Tạo models  
    print("\n3. Tạo models...")
    models = create_models()
    print(f"Sẽ so sánh {len(models)} models: {list(models.keys())}")
    
    # Đánh giá từng model bằng Cross-Validation
    print("\n4. Đánh giá models bằng 5-Fold Cross-Validation...")
    results_list = []
    trained_models = {}
    
    for name, model in models.items():
        results, trained_model = evaluate_model_cv(
            name, model, X_tfidf, y
        )
        results_list.append(results)
        trained_models[name] = trained_model
    
    # So sánh kết quả
    print("\n5. So sánh kết quả...")
    best_model_name, df_results = compare_models(results_list)
    
    # Vẽ confusion matrices
    print("\n6. Vẽ Confusion Matrices...")
    plot_confusion_matrices(models, X_tfidf, y)
    
    # Lưu model tốt nhất
    print("\n7. Lưu model tốt nhất...")
    save_best_model(best_model_name, trained_models, tfidf)
    
    print(f"Model được khuyến nghị cho tuning: {best_model_name}")
if __name__ == "__main__":
    main()
