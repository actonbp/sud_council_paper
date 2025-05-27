# scripts/08_logistic_regression_fs.py

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
PREV_LOGISTIC_RESULTS_DIR = 'results/study1_logistic' # To load coefficients
RESULTS_DIR = 'results/study1_logistic_fs' # New results directory for Feature Selection
MODEL_FILE = os.path.join(RESULTS_DIR, 'logistic_fs_model.joblib')
COEFFICIENTS_FILE = os.path.join(RESULTS_DIR, 'logistic_fs_coefficients.csv')
SELECTED_FEATURES_FILE = os.path.join(RESULTS_DIR, 'selected_features.txt')
N_SPLITS_CV = 5
RANDOM_STATE = 42
COEFF_THRESHOLD = 1e-6 # Threshold to consider a coefficient non-zero (handles floating point issues)

# --- Create Results Directory ---
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load Processed Data ---
print(f"Loading processed data from {PROCESSED_DATA_DIR}...")
try:
    X_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
    X_test_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).squeeze()
    print(f"Loaded Full Training Data: X={X_train_full.shape}, y={y_train.shape}")
    print(f"Loaded Full Testing Data: X={X_test_full.shape}, y={y_test.shape}")
except FileNotFoundError as e:
    print(f"Error: Data file not found - {e}. Make sure previous scripts ran successfully.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Feature Selection based on previous L1 Logistic Regression ---
print(f"\nLoading coefficients from {PREV_LOGISTIC_RESULTS_DIR}...")
prev_coeffs_file = os.path.join(PREV_LOGISTIC_RESULTS_DIR, 'logistic_coefficients.csv')
try:
    prev_coeffs_df = pd.read_csv(prev_coeffs_file)
except FileNotFoundError:
    print(f"Error: Coefficient file not found: {prev_coeffs_file}. Run 'scripts/05_logistic_regression.py' first.")
    exit()

# Select features where the absolute coefficient is greater than the threshold
selected_features = prev_coeffs_df[abs(prev_coeffs_df['coefficient']) > COEFF_THRESHOLD]['feature'].tolist()

if not selected_features:
    print("Error: No features selected based on the coefficients. Check the threshold or the coefficients file.")
    exit()

print(f"Selected {len(selected_features)} features out of {X_train_full.shape[1]}")

# Filter the datasets
X_train = X_train_full[selected_features]
X_test = X_test_full[selected_features]

# Save selected features list
with open(SELECTED_FEATURES_FILE, 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")
print(f"Selected feature list saved to {SELECTED_FEATURES_FILE}")


# --- Feature Scaling (on Selected Features) --- 
print("\nApplying StandardScaler to selected features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Hyperparameter Tuning (Grid Search CV on Selected Features) ---
print(f"\nStarting Hyperparameter Tuning for Logistic Regression on {len(selected_features)} features ({N_SPLITS_CV}-fold CV)...")

# Define parameter grid (can potentially try L2 now as well, L1 may not be needed)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None],
    'solver': ['liblinear'] # Still use liblinear as it handles both penalties
}

# Stratified K-Fold for cross-validation
cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

# Initialize Logistic Regression Classifier
logreg_clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=logreg_clf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# Fit GridSearchCV to the SCALED and REDUCED training data
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

# --- Predict on SCALED Test Set (Selected Features) ---
print("\nMaking predictions on the scaled, selected-feature test set...")
y_pred = best_logreg_clf.predict(X_test_scaled)
y_pred_proba = best_logreg_clf.predict_proba(X_test_scaled)[:, 1]

# --- Evaluate Performance ---
print("\nEvaluating model performance on the test set (selected features)...")

# Classification Report
print("\nClassification Report:")
report = classification_report(y_test, y_pred)
print(report)
with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
    f.write(f"Selected Features: {len(selected_features)}\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"CV ROC AUC: {grid_search.best_score_:.4f}\n\n")
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
plt.title(f'Confusion Matrix (Logistic Regression w/ {len(selected_features)} Features)')
plt.tight_layout()
cm_fig_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_fig_path)
print(f"Confusion Matrix plot saved to {cm_fig_path}")
plt.close()

# Plot ROC Curve
plt.figure(figsize=(7, 6))
RocCurveDisplay.from_estimator(best_logreg_clf, X_test_scaled, y_test)
plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
plt.title(f'ROC Curve (Logistic Regression w/ {len(selected_features)} Features)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
roc_fig_path = os.path.join(RESULTS_DIR, 'roc_curve.png')
plt.savefig(roc_fig_path)
print(f"ROC Curve plot saved to {roc_fig_path}")
plt.close()

# --- Feature Coefficients (Selected Features) --- 
print("\nExtracting and saving model coefficients for selected features...")
coefficients = best_logreg_clf.coef_[0]

coeff_df = pd.DataFrame({
    'feature': selected_features, # Use the list of selected features
    'coefficient': coefficients
})

# Calculate odds ratios
coeff_df['odds_ratio'] = np.exp(coeff_df['coefficient'])

coeff_df = coeff_df.sort_values('coefficient', key=abs, ascending=False)

print(f"\nTop 10 Features by Absolute Coefficient (from selected set):")
print(coeff_df.head(10))

# Save coefficients
coeff_df.to_csv(COEFFICIENTS_FILE, index=False)
print(f"Coefficients saved to {COEFFICIENTS_FILE}")

print("\nLogistic Regression analysis with feature selection complete.") 