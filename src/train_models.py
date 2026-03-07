# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import joblib

# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA

# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.svm import LinearSVC

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from imblearn.over_sampling import SMOTE

# # -----------------------
# # Create folders
# # -----------------------
# os.makedirs("outputs", exist_ok=True)
# os.makedirs("models", exist_ok=True)

# # -----------------------
# # Load dataset
# # -----------------------
# train = pd.read_csv("data/UNSW_NB15_training-set.csv")
# test = pd.read_csv("data/UNSW_NB15_testing-set.csv")
# df = pd.concat([train, test])

# X = df.drop("label", axis=1)
# y = df["label"]

# # -----------------------
# # Pie Chart BEFORE SMOTE
# # -----------------------
# plt.figure()
# y.value_counts().plot.pie(autopct="%1.1f%%")
# plt.title("Before SMOTE")
# plt.savefig("outputs/before_smote.png")
# plt.close()

# # -----------------------
# # Preprocessing
# # -----------------------
# categorical_cols = X.select_dtypes(include=['object']).columns
# numeric_cols = X.select_dtypes(exclude=['object']).columns

# preprocessor = ColumnTransformer([
#     ("num", StandardScaler(), numeric_cols),
#     ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
# ])

# X_processed = preprocessor.fit_transform(X)

# # -----------------------
# # Train-Test Split
# # -----------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X_processed, y, test_size=0.2, random_state=42
# )

# # -----------------------
# # SMOTE (Train only)
# # -----------------------
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# # -----------------------
# # Pie Chart AFTER SMOTE
# # -----------------------
# plt.figure()
# pd.Series(y_train).value_counts().plot.pie(autopct="%1.1f%%")
# plt.title("After SMOTE")
# plt.savefig("outputs/after_smote.png")
# plt.close()

# # -----------------------
# # Correlation Heatmap
# # -----------------------
# numeric_df = df.select_dtypes(include=['number'])
# plt.figure(figsize=(12,8))
# sns.heatmap(numeric_df.corr(), cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.savefig("outputs/correlation_heatmap.png")
# plt.close()

# # -----------------------
# # Feature Importance
# # -----------------------
# rf = RandomForestClassifier(n_estimators=100)
# rf.fit(X_train, y_train)

# importance = rf.feature_importances_ * 100

# plt.figure(figsize=(10,5))
# plt.bar(range(len(importance)), importance)
# plt.title("Feature Importance (%) - Random Forest")
# plt.savefig("outputs/feature_importance.png")
# plt.close()

# # -----------------------
# # PCA
# # -----------------------
# pca = PCA(n_components=20)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# # -----------------------
# # 6 Classifiers
# # -----------------------
# models = {
#     "Logistic": LogisticRegression(max_iter=2000),
#     "KNN": KNeighborsClassifier(),
#     "DecisionTree": DecisionTreeClassifier(),
#     "RandomForest": RandomForestClassifier(),
#     "SVM": LinearSVC(max_iter=5000),
#     "GradientBoost": GradientBoostingClassifier()
# }

# results = []

# for name, model in models.items():

#     cv_scores = cross_val_score(model, X_train_pca, y_train, cv=10)

#     model.fit(X_train_pca, y_train)
#     y_pred = model.predict(X_test_pca)

#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred, average="weighted")
#     rec = recall_score(y_test, y_pred, average="weighted")
#     f1 = f1_score(y_test, y_pred, average="weighted")

#     print(f"\n{name}")
#     print("10-Fold CV Mean Accuracy:", cv_scores.mean())
#     print("Test Accuracy:", acc)
#     print("Precision:", prec)
#     print("Recall:", rec)
#     print("F1:", f1)

#     results.append([name, cv_scores.mean(), acc, prec, rec, f1])

# # Save comparison
# results_df = pd.DataFrame(results,
#                           columns=["Model", "CV Accuracy", "Test Accuracy", "Precision", "Recall", "F1"])
# results_df.to_csv("outputs/model_comparison.csv", index=False)

# # -----------------------
# # GridSearchCV
# # -----------------------
# param_grid = {
#     "n_estimators": [100, 200],
#     "max_depth": [5, 10]
# }

# grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
# grid.fit(X_train_pca, y_train)

# print("\nBest Parameters from GridSearch:", grid.best_params_)

# best_model = grid.best_estimator_

# joblib.dump(best_model, "models/final_model.pkl")

# print("Final tuned model saved.")

import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from data_preprocessing import *
from feature_engineering import *
from evaluate import *

# -----------------------
# Create folders
# -----------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------
# Load Data
# -----------------------
train, test = load_data(
    "data/UNSW_NB15_training-set.csv",
    "data/UNSW_NB15_testing-set.csv"
)

df = pd.concat([train, test])

X, y, preprocessor = preprocess_features(df)

# -----------------------
# Pie chart BEFORE SMOTE
# -----------------------
plot_class_distribution(y, "outputs/before_smote.png")

# -----------------------
# Correlation Heatmap
# -----------------------
correlation_heatmap(df)

# -----------------------
# Train Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# SMOTE
# -----------------------
X_train, y_train = apply_smote(X_train, y_train)

plot_class_distribution(y_train, "outputs/after_smote.png")

# -----------------------
# Scaling
# -----------------------
X_train, X_test, scaler = scale_features(X_train, X_test)

# -----------------------
# Feature Importance
# -----------------------
importance = feature_importance(X_train, y_train)

# Plot feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.bar(range(len(importance)), importance)
plt.title("Feature Importance (%)")
plt.savefig("outputs/feature_importance.png")
plt.close()

# -----------------------
# PCA
# -----------------------
X_train_pca, pca = apply_pca(X_train, 20)
X_test_pca = pca.transform(X_test)

# -----------------------
# Models + Parameter Grids
# -----------------------
models = {

    "Logistic": (
        LogisticRegression(max_iter=2000),
        {"C":[0.1,1,10]}
    ),

    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors":[3,5,7]}
    ),

    "DecisionTree": (
        DecisionTreeClassifier(),
        {"max_depth":[5,10,20]}
    ),

    "RandomForest": (
        RandomForestClassifier(),
        {"n_estimators":[100,200],
         "max_depth":[5,10]}
    ),

    "SVM": (
        LinearSVC(max_iter=5000),
        {"C":[0.1,1,10]}
    ),

    "NaiveBayes": (
    GaussianNB(),
    {
        "var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
    }
)
}

results = []

best_f1 = 0
best_recall = 0
best_accuracy = 0
best_model = None
best_model_name = ""

# -----------------------
# Train Models
# -----------------------
for name,(model,param_grid) in models.items():

    print(f"\n{name} Model")

    # 10 Fold Cross Validation
    cv_scores = cross_val_score(model, X_train_pca, y_train, cv=10)

    # GridSearchCV Hyperparameter tuning
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train_pca, y_train)

    tuned_model = grid.best_estimator_

    # Prediction
    y_pred = tuned_model.predict(X_test_pca)

    # Evaluation metrics
    acc, prec, rec, f1 = evaluate_model(y_test, y_pred)

    print("10 Fold CV Accuracy:", cv_scores.mean())
    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)

    results.append([name,cv_scores.mean(),acc,prec,rec,f1])

    # -----------------------
    # Best Model Selection
    # Priority: F1 → Recall → Accuracy
    # -----------------------
    if (
        f1 > best_f1 or
        (f1 == best_f1 and rec > best_recall) or
        (f1 == best_f1 and rec == best_recall and acc > best_accuracy)
    ):
        best_f1 = f1
        best_recall = rec
        best_accuracy = acc
        best_model = tuned_model
        best_model_name = name

# -----------------------
# Save Best Model
# -----------------------
joblib.dump(best_model,"models/final_model.pkl")

print(f"\nBest model selected: {best_model_name}")
print("Model saved successfully.")