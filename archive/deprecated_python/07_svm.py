# scripts/07_svm.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from scipy.stats import loguniform, uniform
import os
import joblib

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
RESULTS_DIR = 'results/study1_svm' # New results directory
MODEL_FILE = os.path.join(RESULTS_DIR, 'svm_model.joblib')
N_SPLITS_CV = 5
RANDOM_SEARCH_ITER = 50 # Number of iterations for RandomizedSearchCV
RANDOM_STATE = 42

# --- Create Results Directory ---
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load Processed Data ---
print(f"Loading processed data from {PROCESSED_DATA_DIR}...")
try:
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).squeeze()
    print(f"Loaded Training Data: X={X_train.shape}, y={y_train.shape}")
    print(f"Loaded Testing Data: X={X_test.shape}, y={y_test.shape}")
except FileNotFoundError as e:
    print(f"Error: Data file not found - {e}. Make sure 'scripts/02_preprocess_for_ml.py' ran successfully.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Feature Scaling --- 
print("\nApplying StandardScaler to features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Hyperparameter Tuning (Randomized Search CV) ---
print(f"\nStarting Hyperparameter Tuning for SVM ({RANDOM_SEARCH_ITER} iterations, {N_SPLITS_CV}-fold CV)...")

# Define parameter distributions
# Focus on RBF kernel first, but could add 'linear'
param_dist = {
    'C': loguniform(1e-3, 1e3),          # Regularization parameter (log scale)
    'kernel': ['rbf'],                    # Kernel type
    'gamma': loguniform(1e-4, 1e1),        # Kernel coefficient for RBF (log scale)
    'class_weight': ['balanced', None]
}

# Stratified K-Fold for cross-validation
cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

# Initialize SVM Classifier
svm_clf = SVC(probability=True, random_state=RANDOM_STATE)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=svm_clf,
    param_distributions=param_dist,
    n_iter=RANDOM_SEARCH_ITER,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit RandomizedSearchCV to the SCALED training data
try:
    random_search.fit(X_train_scaled, y_train)
except Exception as e:
    print(f"Error during RandomizedSearchCV: {e}")
    exit()

# Print best parameters and score
print(f"\nBest parameters found by RandomizedSearchCV: {random_search.best_params_}")
print(f"Best ROC AUC score during CV: {random_search.best_score_:.4f}")

# --- Train Final Model with Best Parameters ---
print("\nTraining final SVM model using best parameters found...")
best_svm_clf = random_search.best_estimator_

# Save the trained model
print(f"Saving trained model to {MODEL_FILE}...")
joblib.dump(best_svm_clf, MODEL_FILE)

# --- Predict on SCALED Test Set ---
print("\nMaking predictions on the scaled test set...")
y_pred = best_svm_clf.predict(X_test_scaled)
y_pred_proba = best_svm_clf.predict_proba(X_test_scaled)[:, 1] # Probabilities for ROC curve

# --- Evaluate Performance ---
print("\nEvaluating model performance on the test set...")

# Classification Report
print("\nClassification Report:")
report = classification_report(y_test, y_pred)
print(report)
with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")
with open(os.path.join(RESULTS_DIR, 'roc_auc_score.txt'), 'w') as f:
    f.write(f"{roc_auc:.4f}\n")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_svm_clf.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Test Set - SVM)')
plt.tight_layout()
cm_fig_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_fig_path)
print(f"Confusion Matrix plot saved to {cm_fig_path}")
plt.close()

# Plot ROC Curve
plt.figure(figsize=(7, 6))
RocCurveDisplay.from_estimator(best_svm_clf, X_test_scaled, y_test)
plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
plt.title('ROC Curve (Test Set - SVM)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
roc_fig_path = os.path.join(RESULTS_DIR, 'roc_curve.png')
plt.savefig(roc_fig_path)
print(f"ROC Curve plot saved to {roc_fig_path}")
plt.close()

# Note: Feature importance is not straightforward for non-linear SVM kernels like RBF.

print("\nSVM analysis complete.") 