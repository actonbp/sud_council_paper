# scripts/06_xgboost.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for plotting
import xgboost as xgb # XGBoost library
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from scipy.stats import randint, uniform
import os
import joblib

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
RESULTS_DIR = 'results/study1_xgboost' # New results directory
MODEL_FILE = os.path.join(RESULTS_DIR, 'xgboost_model.joblib')
IMPORTANCE_FILE = os.path.join(RESULTS_DIR, 'xgboost_feature_importances.csv')
N_SPLITS_CV = 5
RANDOM_SEARCH_ITER = 50
EARLY_STOPPING_ROUNDS = 10
RANDOM_STATE = 42
TOP_N_FEATURES = 20
VALIDATION_SPLIT_SIZE = 0.2 # For early stopping

# --- Create Results Directory ---
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load Processed Data ---
print(f"Loading processed data from {PROCESSED_DATA_DIR}...")
try:
    X_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
    y_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).squeeze()
    feature_names = X_train_full.columns
    print(f"Loaded Full Training Data: X={X_train_full.shape}, y={y_train_full.shape}")
    print(f"Loaded Testing Data: X={X_test.shape}, y={y_test.shape}")
except FileNotFoundError as e:
    print(f"Error: Data file not found - {e}.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Calculate scale_pos_weight for imbalance handling ---
neg_count = np.sum(y_train_full == 0)
pos_count = np.sum(y_train_full == 1)
scale_pos_weight = neg_count / pos_count
print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")

# --- Hyperparameter Tuning (Randomized Search CV) ---
print(f"\nStarting Hyperparameter Tuning for XGBoost ({RANDOM_SEARCH_ITER} iterations, {N_SPLITS_CV}-fold CV)...")

# Define parameter distributions
param_dist = {
    'n_estimators': randint(100, 1000),       # Number of boosting rounds
    'learning_rate': uniform(0.01, 0.3),      # Step size shrinkage
    'max_depth': randint(3, 10),              # Max depth of trees
    'subsample': uniform(0.6, 0.4),         # Fraction of samples (0.6 to 1.0)
    'colsample_bytree': uniform(0.6, 0.4),    # Fraction of features per tree
    'gamma': uniform(0, 0.5),               # Min loss reduction (regularization)
    'scale_pos_weight': [scale_pos_weight]     # Handle imbalance
}

# Stratified K-Fold
cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

# Initialize XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc', # Use AUC for evaluation metric during training/CV
    use_label_encoder=False, # Recommended starting with XGBoost > 1.0
    random_state=RANDOM_STATE
)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=RANDOM_SEARCH_ITER,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit RandomizedSearchCV to the full training data
print("\nFitting RandomizedSearchCV...")
try:
    random_search.fit(X_train_full, y_train_full)
except Exception as e:
    print(f"Error during RandomizedSearchCV: {e}")
    exit()

# Print best parameters and score
print(f"\nBest parameters found by RandomizedSearchCV: {random_search.best_params_}")
print(f"Best ROC AUC score during CV: {random_search.best_score_:.4f}")

best_params = random_search.best_params_

# --- Train Final Model with Best Parameters (No Early Stopping) ---
print("\nTraining final XGBoost model using best parameters for the full number of estimators...")

# Initialize final model with best parameters found by RandomizedSearch
# Remove early_stopping_rounds here
final_xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    # early_stopping_rounds=EARLY_STOPPING_ROUNDS, # REMOVED
    **best_params # Unpack the best parameters found
)

# Train on the FULL training set (X_train_full, y_train_full)
# Remove eval_set
try:
    final_xgb_clf.fit(
        X_train_full, # Use full training set
        y_train_full,
        # eval_set=[(X_val_es, y_val_es)], # REMOVED
        verbose=100 # Print progress every 100 rounds
    )
    print(f"Training complete after {best_params.get('n_estimators', 'N/A')} rounds.") # Use n_estimators from best_params
except Exception as e:
    print(f"Error during final model training: {e}")
    exit()

# Save the trained model
print(f"Saving trained model to {MODEL_FILE}...")
joblib.dump(final_xgb_clf, MODEL_FILE)

# --- Predict on Test Set ---
print("\nMaking predictions on the test set...")
y_pred = final_xgb_clf.predict(X_test)
y_pred_proba = final_xgb_clf.predict_proba(X_test)[:, 1]

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
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_xgb_clf.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Test Set - XGBoost)')
plt.tight_layout()
cm_fig_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_fig_path)
print(f"Confusion Matrix plot saved to {cm_fig_path}")
plt.close()

# Plot ROC Curve
plt.figure(figsize=(7, 6))
RocCurveDisplay.from_estimator(final_xgb_clf, X_test, y_test)
plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
plt.title('ROC Curve (Test Set - XGBoost)')
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
importances = final_xgb_clf.feature_importances_

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
plt.title(f'Top {TOP_N_FEATURES} Feature Importances (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
importance_fig_path = os.path.join(RESULTS_DIR, f'feature_importances_top_{TOP_N_FEATURES}.png')
plt.savefig(importance_fig_path)
print(f"Top {TOP_N_FEATURES} feature importances plot saved to {importance_fig_path}")
plt.close()

print("\nXGBoost analysis complete.") 