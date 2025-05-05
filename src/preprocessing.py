import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_dataset(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Drop irrelevant columns if any
    if 'EmployeeNumber' in df.columns:
        df.drop('EmployeeNumber', axis=1, inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to balance training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test
