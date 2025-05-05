# test_preprocessing.py
from src.preprocessing import load_dataset, preprocess_data

# Dataset path
path = r'data/IBM-HR2.csv'

# Load
df = load_dataset(path)

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(df)

# Shape print to verify
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
