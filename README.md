# Network Intrusion Detection using UNSW-NB15

This project implements a machine learning pipeline for network intrusion detection.

Features included:

1. Data preprocessing
2. Data balancing using SMOTE
3. Data visualization (Pie chart)
4. Feature scaling
5. 6 classification algorithms
6. 10-fold cross validation
7. Evaluation metrics (Accuracy, Precision, Recall, F1-score)
8. Hyperparameter tuning using GridSearchCV
9. Feature engineering
   - Feature importance
   - Correlation heatmap
   - PCA
10. GUI for prediction

Classifiers used:
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- SVM
- Gradient Boosting

Run training:
python src/train_models.py

Run GUI:
python src/gui.py