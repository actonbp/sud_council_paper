# scripts/04_rf_smote_tuned.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import os
import joblib

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
RESULTS_DIR = 'results/study1_rf_smote_tuned' # New results directory
MODEL_FILE = os.path.join(RESULTS_DIR, 'rf_smote_tuned_model.joblib')
N_SPLITS_CV = 5
RANDOM_SEARCH_ITER = 50 # Number of parameter settings to sample (adjust as needed)
RANDOM_STATE = 42
TOP_N_FEATURES = 20

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
    print(f"Error: Data file not found - {e}.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Apply SMOTE to Training Data --- 
print(f"\nApplying SMOTE to the training data...")
print(f"Original training distribution:\n{y_train.value_counts(normalize=True)}")
smote = SMOTE(random_state=RANDOM_STATE)
try:
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled training distribution:\n{y_train_resampled.value_counts(normalize=True)}")
    print(f"Resampled training shape: X={X_train_resampled.shape}, y={y_train_resampled.shape}")
except Exception as e:
    print(f"Error during SMOTE application: {e}")
    exit()

# --- Hyperparameter Tuning (Randomized Search CV) ---
print(f"\nStarting Hyperparameter Tuning (RandomizedSearchCV with {RANDOM_SEARCH_ITER} iterations, {N_SPLITS_CV}-fold CV)...")

# Define parameter distributions - wider ranges and add max_features
param_dist = {
    'n_estimators': randint(100, 500),      # Wider range for number of trees
    'max_depth': [None, 10, 20, 30],       # Include deeper options, None
    'min_samples_split': randint(2, 11),     # Range for min samples to split
    'min_samples_leaf': randint(1, 6),      # Range for min samples per leaf
    'max_features': ['sqrt', 'log2', None], # Options for features considered at each split
    'class_weight': [None] # Using SMOTE, so typically don't need balanced weights here
}

# Stratified K-Fold for cross-validation
cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

# Initialize RF Classifier
rf_clf = RandomForestClassifier(random_state=RANDOM_STATE)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_clf,
    param_distributions=param_dist,
    n_iter=RANDOM_SEARCH_ITER,
    scoring='roc_auc', # Still use ROC AUC
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit RandomizedSearchCV to the RESAMPLED training data
print("\nFitting RandomizedSearchCV on SMOTE-resampled data...")
try:
    random_search.fit(X_train_resampled, y_train_resampled)
except Exception as e:
    print(f"Error during RandomizedSearchCV: {e}")
    exit()

# Print best parameters and score
print(f"\nBest parameters found by RandomizedSearchCV: {random_search.best_params_}")
print(f"Best ROC AUC score during CV (on resampled data): {random_search.best_score_:.4f}")

# --- Train Final Model with Best Parameters ---
print("\nTraining final Random Forest model using best parameters found...")
best_rf_clf = random_search.best_estimator_

# Save the trained model
print(f"Saving trained model to {MODEL_FILE}...")
joblib.dump(best_rf_clf, MODEL_FILE)

# --- Predict on ORIGINAL Test Set ---
print("\nMaking predictions on the ORIGINAL (un-resampled) test set...")
y_pred = best_rf_clf.predict(X_test)
y_pred_proba = best_rf_clf.predict_proba(X_test)[:, 1]

# --- Evaluate Performance on ORIGINAL Test Set ---
print("\nEvaluating model performance on the ORIGINAL test set...")

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
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf_clf.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Test Set - SMOTE + Tuned)')
plt.tight_layout()
cm_fig_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_fig_path)
print(f"Confusion Matrix plot saved to {cm_fig_path}")
plt.close()

# Plot ROC Curve
plt.figure(figsize=(7, 6))
RocCurveDisplay.from_estimator(best_rf_clf, X_test, y_test)
plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
plt.title('ROC Curve (Test Set - SMOTE + Tuned)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
roc_fig_path = os.path.join(RESULTS_DIR, 'roc_curve.png')
plt.savefig(roc_fig_path)
print(f"ROC Curve plot saved to {roc_fig_path}")
plt.close()

# --- Feature Importance --- 
print("\nCalculating and saving feature importances...")
importances = best_rf_clf.feature_importances_
feature_names = X_train.columns # Use original X_train columns

feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Save all feature importances
importance_all_path = os.path.join(RESULTS_DIR, 'feature_importances_all.csv')
feature_importance_df.to_csv(importance_all_path, index=False)
print(f"All feature importances saved to {importance_all_path}")

# Get Top N features
top_features_df = feature_importance_df.head(TOP_N_FEATURES)
print(f"\nTop {TOP_N_FEATURES} Features:")
print(top_features_df)

# Save Top N feature importances
importance_top_path = os.path.join(RESULTS_DIR, f'feature_importances_top_{TOP_N_FEATURES}.csv')
top_features_df.to_csv(importance_top_path, index=False)
print(f"Top {TOP_N_FEATURES} feature importances saved to {importance_top_path}")

# Plot Top N Feature Importances
plt.figure(figsize=(10, TOP_N_FEATURES * 0.3))
sns.barplot(x='importance', y='feature', data=top_features_df, palette='viridis')
plt.title(f'Top {TOP_N_FEATURES} Feature Importances (SMOTE + Tuned)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
importance_fig_path = os.path.join(RESULTS_DIR, f'feature_importances_top_{TOP_N_FEATURES}.png')
plt.savefig(importance_fig_path)
print(f"Top {TOP_N_FEATURES} feature importances plot saved to {importance_fig_path}")
plt.close()

print("\nRandom Forest analysis with SMOTE and RandomizedSearch complete.") 