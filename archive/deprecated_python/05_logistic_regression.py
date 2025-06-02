# scripts/05_logistic_regression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for plotting
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
import os
import joblib

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
RESULTS_DIR = 'results/study1_logistic' # New results directory
MODEL_FILE = os.path.join(RESULTS_DIR, 'logistic_model.joblib')
COEFFICIENTS_FILE = os.path.join(RESULTS_DIR, 'logistic_coefficients.csv')
N_SPLITS_CV = 5
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
    feature_names = X_train.columns # Store feature names before scaling
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

# --- Hyperparameter Tuning (Grid Search CV) ---
print(f"\nStarting Hyperparameter Tuning for Logistic Regression ({N_SPLITS_CV}-fold CV)...")

# Define parameter grid
# Note: 'l1' penalty requires 'liblinear' or 'saga' solver. 'l2' works with default 'lbfgs'.
# Using liblinear as it works well for smaller datasets and supports both penalties.
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], # Inverse of regularization strength
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None],
    'solver': ['liblinear'] # Good solver that handles l1/l2
}

# Stratified K-Fold for cross-validation
cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

# Initialize Logistic Regression Classifier
logreg_clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000) # Increased max_iter

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=logreg_clf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# Fit GridSearchCV to the SCALED training data
try:
    grid_search.fit(X_train_scaled, y_train)
except Exception as e:
    print(f"Error during GridSearchCV: {e}")
    exit()

# Print best parameters and score
print(f"\nBest parameters found by GridSearchCV: {grid_search.best_params_}")
print(f"Best ROC AUC score during CV: {grid_search.best_score_:.4f}")

# --- Train Final Model with Best Parameters ---
print("\nTraining final Logistic Regression model using best parameters found...")
best_logreg_clf = grid_search.best_estimator_

# Save the trained model
print(f"Saving trained model to {MODEL_FILE}...")
joblib.dump(best_logreg_clf, MODEL_FILE)

# --- Predict on SCALED Test Set ---
print("\nMaking predictions on the scaled test set...")
y_pred = best_logreg_clf.predict(X_test_scaled)
y_pred_proba = best_logreg_clf.predict_proba(X_test_scaled)[:, 1] # Probabilities for ROC curve

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
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_logreg_clf.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Test Set - Logistic Regression)')
plt.tight_layout()
cm_fig_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_fig_path)
print(f"Confusion Matrix plot saved to {cm_fig_path}")
plt.close()

# Plot ROC Curve
plt.figure(figsize=(7, 6))
RocCurveDisplay.from_estimator(best_logreg_clf, X_test_scaled, y_test)
plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
plt.title('ROC Curve (Test Set - Logistic Regression)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
roc_fig_path = os.path.join(RESULTS_DIR, 'roc_curve.png')
plt.savefig(roc_fig_path)
print(f"ROC Curve plot saved to {roc_fig_path}")
plt.close()

# --- Feature Coefficients --- 
print("\nExtracting and saving model coefficients...")
# Coefficients are in a nested list for binary classification, grab the first element
coefficients = best_logreg_clf.coef_[0]

coeff_df = pd.DataFrame({
    'feature': feature_names, # Use names stored before scaling
    'coefficient': coefficients
})

# Calculate odds ratios (exponentiated coefficients)
coeff_df['odds_ratio'] = np.exp(coeff_df['coefficient'])

coeff_df = coeff_df.sort_values('coefficient', key=abs, ascending=False)

print(f"\nTop 10 Features by Absolute Coefficient:")
print(coeff_df.head(10))

# Save coefficients
coeff_df.to_csv(COEFFICIENTS_FILE, index=False)
print(f"Coefficients saved to {COEFFICIENTS_FILE}")

print("\nLogistic Regression analysis complete.") 