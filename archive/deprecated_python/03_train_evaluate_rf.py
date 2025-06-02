# scripts/03_train_evaluate_rf.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
import os
import joblib # To save the model

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
RESULTS_DIR = 'results/study1_rf'
MODEL_FILE = os.path.join(RESULTS_DIR, 'rf_model.joblib')
N_SPLITS_CV = 5 # Number of folds for cross-validation
RANDOM_STATE = 42
TOP_N_FEATURES = 20 # How many top features to report/plot

# --- Create Results Directory ---
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load Processed Data ---
print(f"Loading processed data from {PROCESSED_DATA_DIR}...")
try:
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).squeeze() # Ensure y is a Series
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).squeeze()
    print(f"Loaded Training Data: X={X_train.shape}, y={y_train.shape}")
    print(f"Loaded Testing Data: X={X_test.shape}, y={y_test.shape}")
except FileNotFoundError as e:
    print(f"Error: Data file not found - {e}. Make sure 'scripts/02_preprocess_for_ml.py' ran successfully.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Hyperparameter Tuning (Grid Search CV) ---
print(f"\nStarting Hyperparameter Tuning (GridSearchCV with {N_SPLITS_CV}-fold CV)...")

# Define parameter grid - adjust ranges as needed
param_grid = {
    'n_estimators': [100, 200, 300],      # Number of trees
    'max_depth': [None, 10, 20],          # Max depth of trees (None means unlimited until leaves are pure or min_samples_split)
    'min_samples_split': [2, 5],      # Min samples required to split an internal node
    'min_samples_leaf': [1, 3],       # Min samples required at a leaf node
    'class_weight': ['balanced', None] # Handle class imbalance or not
}

# Stratified K-Fold for cross-validation
cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

# Initialize RF Classifier
rf_clf = RandomForestClassifier(random_state=RANDOM_STATE)

# Initialize GridSearchCV - using roc_auc as it's good for imbalanced data
grid_search = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid,
    scoring='roc_auc', # Or 'f1_weighted', 'accuracy' etc.
    cv=cv,
    n_jobs=-1, # Use all available CPU cores
    verbose=1  # Show progress
)

# Fit GridSearchCV to the training data
try:
    grid_search.fit(X_train, y_train)
except Exception as e:
    print(f"Error during GridSearchCV: {e}")
    exit()

# Print best parameters and score
print(f"\nBest parameters found by GridSearchCV: {grid_search.best_params_}")
print(f"Best ROC AUC score during CV: {grid_search.best_score_:.4f}")

# --- Train Final Model with Best Parameters ---
print("\nTraining final Random Forest model using best parameters found...")
best_rf_clf = grid_search.best_estimator_ # Get the best model directly
# No need to fit again, GridSearchCV refits the best model on the whole training set by default if refit=True (which is the default)

# Save the trained model
print(f"Saving trained model to {MODEL_FILE}...")
joblib.dump(best_rf_clf, MODEL_FILE)

# --- Predict on Test Set ---
print("\nMaking predictions on the test set...")
y_pred = best_rf_clf.predict(X_test)
y_pred_proba = best_rf_clf.predict_proba(X_test)[:, 1] # Probabilities for ROC curve

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
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf_clf.classes_)
cm_display.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
cm_fig_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_fig_path)
print(f"Confusion Matrix plot saved to {cm_fig_path}")
plt.close() # Close the plot

# Plot ROC Curve
plt.figure(figsize=(7, 6))
RocCurveDisplay.from_estimator(best_rf_clf, X_test, y_test)
plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)') # Add chance line
plt.title('Receiver Operating Characteristic (ROC) Curve (Test Set)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
roc_fig_path = os.path.join(RESULTS_DIR, 'roc_curve.png')
plt.savefig(roc_fig_path)
print(f"ROC Curve plot saved to {roc_fig_path}")
plt.close() # Close the plot

# --- Feature Importance --- 
print("\nCalculating and saving feature importances...")
importances = best_rf_clf.feature_importances_
feature_names = X_train.columns

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
plt.figure(figsize=(10, TOP_N_FEATURES * 0.3)) # Adjust figure size based on N
sns.barplot(x='importance', y='feature', data=top_features_df, palette='viridis')
plt.title(f'Top {TOP_N_FEATURES} Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
importance_fig_path = os.path.join(RESULTS_DIR, f'feature_importances_top_{TOP_N_FEATURES}.png')
plt.savefig(importance_fig_path)
print(f"Top {TOP_N_FEATURES} feature importances plot saved to {importance_fig_path}")
plt.close()

print("\nRandom Forest analysis complete.") 