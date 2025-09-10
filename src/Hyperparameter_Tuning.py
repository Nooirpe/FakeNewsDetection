import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, validation_curve, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
import joblib
import os
import time
import matplotlib.pyplot as plt
from Utils import load_preprocessed_data, print_data_info, create_model_directory



def get_tuning_params(model_name):
    """Lấy parameters để tune cho từng model với Cross-Validation"""
    if model_name == "Logistic Regression":
        return {
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.9, 0.95],
            'tfidf__min_df': [1, 2, 5],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'lbfgs'],
            'classifier__max_iter': [1000, 2000]
        }
    elif model_name == "Random Forest":
        return {
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.9, 0.95],
            'tfidf__min_df': [1, 2, 5],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5, 10]
        }
    elif model_name == "Naive Bayes":
        return {
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.9, 0.95],
            'tfidf__min_df': [1, 2, 5],
            'classifier__alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
        }
    elif model_name == "Linear SVC":
        return {
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.9, 0.95],
            'tfidf__min_df': [1, 2, 5],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__max_iter': [1000, 2000, 5000],
            'classifier__dual': [False]  # Better for sparse features
        }
    else:
        return None

def create_model(model_name):
    """Tạo model dựa trên tên"""
    if model_name == "Logistic Regression":
        return LogisticRegression(random_state=42)
    elif model_name == "Random Forest":
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_name == "Naive Bayes":
        return MultinomialNB()
    elif model_name == "Linear SVC":
        return LinearSVC(random_state=42)
    else:
        return None

def create_pipeline(model_name):
    """Tạo pipeline với TF-IDF + Model"""
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=10000,  # Default values
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2
    )
    
    model = create_model(model_name)
    if model is None:
        return None
    
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', model)
    ])
    
    return pipeline

def tune_hyperparameters_cv(X, y, model_name, use_grid_search=False):
    """Tune hyperparameters bằng Cross-Validation"""
    
    print(f"\n TUNING {model_name.upper()}")
    
    # Tạo pipeline
    pipeline = create_pipeline(model_name)
    if pipeline is None:
        print(f"Model {model_name} không được hỗ trợ!")
        return None, None, None
    
    # Lấy parameter grid
    param_grid = get_tuning_params(model_name)
    if param_grid is None:
        print(f"Không có parameter grid cho {model_name}!")
        return None, None, None
    
    # Tạo StratifiedKFold cho cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Chọn search strategy
    if use_grid_search:
        print("Sử dụng GridSearchCV (có thể mất nhiều thời gian)...")
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
    else:
        print("Sử dụng RandomizedSearchCV (nhanh hơn)...")
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=30,  # Test 30 combinations
            cv=cv,
            scoring='f1',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    
    # Bắt đầu tuning
    print(f"Bắt đầu tuning với {len(X)} samples...")
    start_time = time.time()
    
    search.fit(X, y)
    
    tune_time = time.time() - start_time
    
    # In kết quả
    print(f"\n=== KẾT QUẢ TUNING ===")
    print(f"Thời gian tuning: {tune_time:.2f}s")
    print(f"Best CV F1-score: {search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    return search.best_estimator_, search.best_params_, search.best_score_

def check_overfitting_with_learning_curves(model, X, y, model_name):
    """
    Kiểm tra overfitting bằng Learning Curves
    Vẽ biểu đồ train vs validation score theo số lượng samples
    """
    print(f"\n=== KIỂM TRA OVERFITTING - LEARNING CURVES ===")
    
    # Tạo range số samples để test
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    print("Đang tính toán learning curves...")
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )
    
    # Tính mean và std
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title(f'Learning Curves - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Phân tích kết quả
    final_train_score = train_mean[-1]
    final_val_score = val_mean[-1]
    gap = final_train_score - final_val_score
    
    print(f"Train Score (cuối): {final_train_score:.4f}")
    print(f"Validation Score (cuối): {final_val_score:.4f}")
    print(f"Gap (Train - Val): {gap:.4f}")
    
    if gap > 0.1:
        print("CẢNH BÁO: Có dấu hiệu OVERFITTING mạnh!")
        print("   → Train score cao hơn validation score đáng kể")
    elif gap > 0.05:
        print("Có dấu hiệu overfitting nhẹ")
        print("   → Nên cân nhắc regularization")
    else:
        print("Model KHÔNG bị overfitting")
        print("   → Train và validation scores gần nhau")
    
    return {
        'train_score': final_train_score,
        'val_score': final_val_score,
        'gap': gap,
        'overfitting': gap > 0.05
    }



def check_overfitting_comprehensive(model, X, y, model_name):
    """
    Kiểm tra overfitting toàn diện
    """
    print(f"\n{'='*60}")
    print(f"🔍 PHÂN TÍCH OVERFITTING TOÀN DIỆN - {model_name}")
    print(f"{'='*60}")
    
    # 1. Learning Curves
    learning_results = check_overfitting_with_learning_curves(model, X, y, model_name)
        
    # 3. Cross-validation với train scores
    print(f"\n TRAIN VS VALIDATION SCORES ")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Tính train scores
    train_scores = []
    val_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit và predict
        model.fit(X_train_fold, y_train_fold)
        
        train_pred = model.predict(X_train_fold)
        val_pred = model.predict(X_val_fold)
        
        train_score = f1_score(y_train_fold, train_pred)
        val_score = f1_score(y_val_fold, val_pred)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    train_mean = np.mean(train_scores)
    val_mean = np.mean(val_scores)
    gap_mean = train_mean - val_mean
    
    print(f"Train F1 (CV): {train_mean:.4f} (+/- {np.std(train_scores)*2:.4f})")
    print(f"Val F1 (CV): {val_mean:.4f} (+/- {np.std(val_scores)*2:.4f})")
    print(f"Gap: {gap_mean:.4f}")
    
    # 4. Tổng kết
    print(f"\nTỔNG KẾT OVERFITTING ANALYSIS")
   
    
    overall_overfitting = False
    
    if learning_results and learning_results['overfitting']:
        print(" Learning Curves: Có overfitting")
        overall_overfitting = True
    else:
        print(" Learning Curves: Không overfitting")
    
    
    if gap_mean > 0.05:
        print(" Train vs Val: Có overfitting")
        overall_overfitting = True
    else:
        print(" Train vs Val: Không overfitting")
    
    print(f"\n KẾT LUẬN CUỐI CÙNG:")
    if overall_overfitting:
        print(" Model BỊ OVERFITTING!")
        print(" Khuyến nghị:")
        print("   - Tăng regularization (C nhỏ hơn cho Logistic/SVC)")
        print("   - Giảm complexity (max_depth nhỏ hơn cho Random Forest)")
        print("   - Thêm dữ liệu training")
        print("   - Feature selection")
        print("   - Early stopping")
    else:
        print(" Model KHÔNG bị overfitting - Tốt!")
    
    return {
        'overall_overfitting': overall_overfitting,
        'learning_curves': learning_results,
        'train_val_gap': gap_mean
    }

def evaluate_final_model(best_model, X, y):
    """Đánh giá final model bằng cross-validation"""
    print("\n=== ĐÁNH GIÁ FINAL MODEL ===")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate với nhiều metrics
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    print("Đang đánh giá với 5-fold cross-validation...")
    results = {}
    
    for metric in scoring:
        scores = cross_val_score(best_model, X, y, cv=cv, scoring=metric)
        results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"{metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    return results
def evaluate_final_model(best_model, X, y):
    """Đánh giá final model bằng cross-validation"""
    print("\nĐÁNH GIÁ FINAL MODEL")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate với nhiều metrics
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    print("Đang đánh giá với 5-fold cross-validation...")
    results = {}
    
    for metric in scoring:
        scores = cross_val_score(best_model, X, y, cv=cv, scoring=metric)
        results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"{metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    return results

def save_tuned_model(best_model, model_name, best_params, cv_score):
    """Lưu model và parameters đã tune"""
    model_dir = create_model_directory()
    
    # Lưu model
    model_path = os.path.join(model_dir, f"Tuned_{model_name.replace(' ', '_')}.pkl")
    joblib.dump(best_model, model_path)
    
    # Lưu parameters
    params_path = os.path.join(model_dir, f"Best_Parameters_{model_name.replace(' ', '_')}.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write(f"Best Model: {model_name}\n")
        f.write(f"Best CV F1-Score: {cv_score:.4f}\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write("="*50 + "\n\n")
        f.write("Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
    
    print(f"\nĐã lưu tuned model: {model_path}")
    print(f"Đã lưu parameters: {params_path}")
    
    return model_path, params_path

def main():
    print("HYPERPARAMETER TUNING - FAKE NEWS DETECTION (Cross-Validation)")
    
    # Danh sách models có sẵn
    available_models = ["Logistic Regression", "Random Forest", "Naive Bayes", "Linear SVC"]
    
    print("Các models có sẵn:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    
    # Nhập tên model cần tune
    model_choice = input("\nNhập số thứ tự hoặc tên model (enter = Naive Bayes): ").strip()
    
    if model_choice.isdigit():
        model_idx = int(model_choice) - 1
        if 0 <= model_idx < len(available_models):
            model_name = available_models[model_idx]
        else:
            print("Số không hợp lệ, sử dụng Naive Bayes")
            model_name = "Naive Bayes"
    elif model_choice in available_models:
        model_name = model_choice
    elif not model_choice:
        model_name = "Naive Bayes"
    else:
        print("Tên model không hợp lệ, sử dụng Naive Bayes")
        model_name = "Naive Bayes"
    
    print(f"Sẽ tune model: {model_name}")
    
    # Chọn search strategy
    search_choice = input("\nSử dụng GridSearch? (y/N): ").strip().lower()
    use_grid_search = search_choice in ['y', 'yes']
    
    if use_grid_search:
        print(" GridSearch sẽ mất nhiều thời gian hơn nhưng tìm được params tốt hơn")
    else:
        print(" RandomizedSearch sẽ nhanh hơn và vẫn cho kết quả tốt")
    
    # Chọn có kiểm tra overfitting không
    overfitting_choice = input("\nKiểm tra Overfitting? (Y/n): ").strip().lower()
    check_overfitting = overfitting_choice not in ['n', 'no']
    
    if check_overfitting:
        print(" Sẽ phân tích overfitting với learning curves và validation curves")
    else:
        print(" Bỏ qua kiểm tra overfitting")
    
    # Load dữ liệu
    print("\n1. Load dữ liệu...")
    X, y = load_preprocessed_data()
    if X is None:
        return
    
    print_data_info(X, y, "Full Dataset")
    
    # Tune hyperparameters
    print("\n2. Tune hyperparameters...")
    best_model, best_params, best_score = tune_hyperparameters_cv(
        X, y, model_name, use_grid_search
    )
    
    if best_model is None:
        print("Tuning thất bại!")
        return
    
    # Đánh giá final model
    print("\n3. Đánh giá final model...")
    final_results = evaluate_final_model(best_model, X, y)
    
    # Kiểm tra overfitting nếu được yêu cầu
    overfitting_results = None
    if check_overfitting:
        print("\n4. Kiểm tra Overfitting...")
        overfitting_results = check_overfitting_comprehensive(best_model, X, y, model_name)
    
    # Lưu model đã tune
    step_num = "5" if check_overfitting else "4"
    print(f"\n{step_num}. Lưu model đã tune...")
    model_path, params_path = save_tuned_model(
        best_model, model_name, best_params, best_score
    )
    
    # Tóm tắt kết quả
    print(f"\nHOÀN THÀNH TUNING")
    print(f"Model: {model_name}")
    print(f"Best CV F1-Score: {best_score:.4f}")
    print(f"Final Accuracy: {final_results['accuracy']['mean']:.4f} (+/- {final_results['accuracy']['std']*2:.4f})")
    print(f"Final F1-Score: {final_results['f1']['mean']:.4f} (+/- {final_results['f1']['std']*2:.4f})")
    
    if overfitting_results:
        if overfitting_results['overall_overfitting']:
            print(" Overfitting: CÓ - Cần xem xét lại parameters!")
        else:
            print(" Overfitting: KHÔNG - Model tốt!")
    
    print(f"Model saved: {model_path}")

    return best_model, best_params, final_results, overfitting_results

if __name__ == "__main__":
    main()
