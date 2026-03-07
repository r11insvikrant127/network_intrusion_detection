import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


# -----------------------------
# Load Dataset
# -----------------------------
def load_data(train_path, test_path):

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


# -----------------------------
# Preprocess Features
# -----------------------------
def preprocess_features(df):

    X = df.drop("label", axis=1)
    y = df["label"]

    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


# -----------------------------
# Plot Class Distribution
# -----------------------------
def plot_class_distribution(y, filename):

    counts = pd.Series(y).value_counts()

    plt.figure()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.title("Class Distribution")
    plt.savefig(filename)
    plt.close()


# -----------------------------
# Apply SMOTE
# -----------------------------
def apply_smote(X, y):

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled


# -----------------------------
# Feature Scaling
# -----------------------------
def scale_features(X_train, X_test):

    scaler = StandardScaler(with_mean=False)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler