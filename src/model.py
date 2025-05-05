import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.preprocessing import load_dataset, preprocess_data
from src.evaluate import evaluate_model  # if evaluate.py is in src/

# 🎯 Hyperparameter Tuning Function
def tune_model_with_gridsearch(X_train, y_train):
    """Hyperparameter tuning using GridSearchCV"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=1, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    
    print("🔍 Best Params:", grid_search.best_params_)
    return grid_search.best_estimator_

# 🧠 Train model
def train_model(X_train, y_train):
    """Train Random Forest model"""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# 💾 Save model
def save_model(clf, path='models/model.pkl'):
    """Save trained model"""
    joblib.dump(clf, path)
    print(f"✅ Model saved to: {path}")

# 🚀 Main block (ONLY when this file is run directly)
if __name__ == "__main__":
    # 1. Load dataset
    path = r'C:\Users\Lenovo\OneDrive\Desktop\Employee-Attrition-Prediction model for clg\data\IBM-HR2.csv'
    df = load_dataset(path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 2. Train model (simple or tuned)
    clf = train_model(X_train, y_train)
    # OR use tuning:
    # clf = tune_model_with_gridsearch(X_train, y_train)

    # 3. Evaluate model
    evaluate_model(clf, X_test, y_test)

    # 4. Save model
    save_model(clf)
