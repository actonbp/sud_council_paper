# scripts/09_ordinal_logistic_regression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, accuracy_score
import mord # For Ordinal Logistic Regression
import os
import joblib # To save the model

# --- Configuration ---\n# Assuming the ordinal target is in the output of script 01
INPUT_FILE = 'data/processed/survey_processed.csv'
OUTPUT_DIR = 'results/study1_ordinal_logistic'
ORDINAL_TARGET_COLUMN = 'interest_dv' # Assuming this holds the 1, 2, 3, 4 values
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns to remove before defining features (predictors)
# Match columns removed in script 02_preprocess_for_ml.py
POTENTIAL_COLS_TO_REMOVE = [
    ORDINAL_TARGET_COLUMN,
    'careless_responder', 
    'completed',
    'progress' 
]

# --- Create Output Directory ---\nprint(f"Ensuring output directory exists: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Data ---\nprint(f"Loading data from {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_FILE}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Verify Ordinal Target ---\nif ORDINAL_TARGET_COLUMN not in df.columns:
    print(f"ERROR: Ordinal target column '{ORDINAL_TARGET_COLUMN}' not found in {INPUT_FILE}.")
    exit()

print(f"Original value counts for ordinal target '{ORDINAL_TARGET_COLUMN}':")
print(df[ORDINAL_TARGET_COLUMN].value_counts(dropna=False))

# --- Drop Rows with Missing Target ---
original_rows = len(df)
df.dropna(subset=[ORDINAL_TARGET_COLUMN], inplace=True)
rows_after_target_drop = len(df)
print(f"Dropped {original_rows - rows_after_target_drop} rows with missing ordinal target variable ('{ORDINAL_TARGET_COLUMN}' was NaN).")
print(f"Dataset size for Ordinal ML: {rows_after_target_drop} rows.")

# Convert ordinal target to integer (important for some modeling steps/metrics)
# Ensure it's treated as ordered categories later if needed by specific libraries
df[ORDINAL_TARGET_COLUMN] = df[ORDINAL_TARGET_COLUMN].astype(int)

# --- Separate Features (X) and Target (y) ---\nprint("Separating features (X) and ordinal target (y)...")
y_ordinal = df[ORDINAL_TARGET_COLUMN]

# Filter list of columns to remove to only include those actually present
cols_to_remove_for_X = [col for col in POTENTIAL_COLS_TO_REMOVE if col in df.columns]
print(f"Columns being removed to create feature set X: {cols_to_remove_for_X}")
X = df.drop(columns=cols_to_remove_for_X)
print(f"Initial features (X) shape: {X.shape}")

# --- Replicate Preprocessing from script 02 ---\n\n# Group Sparse Categories (mirroring script 02)
print("\\nGrouping sparse categories for 'race' and 'gender_identity'...")
race_sparse_cats = ['Black', 'Other (please specify):', 'Middle Eastern', 'I prefer not to answer']
gender_sparse_cats = ['I prefer not to answer', 'Nonbinary', 'Gender queer', 'Transgender', 'Agender']

if 'race' in X.columns:
    print(f"Original 'race' categories: {X['race'].unique()}")
    X['race'] = X['race'].replace(race_sparse_cats, 'Other/Multiple/Unknown Race')
    print(f"New 'race' categories: {X['race'].unique()}")
else:
    print("'race' column not found for grouping.")

if 'gender_identity' in X.columns:
    print(f"Original 'gender_identity' categories: {X['gender_identity'].unique()}")
    X['gender_identity'] = X['gender_identity'].replace(gender_sparse_cats, 'Gender Diverse/Unknown')
    print(f"New 'gender_identity' categories: {X['gender_identity'].unique()}")
else:
    print("'gender_identity' column not found for grouping.")

# Identify Feature Types
nominal_categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_ordinal_features = X.select_dtypes(include=['number']).columns.tolist()
print(f"\\nIdentified {len(nominal_categorical_features)} nominal categorical features.")
print(f"Identified {len(numeric_ordinal_features)} numeric/ordinal features.")

# Impute Missing Values (mirroring script 02)
print("\\nImputing missing values...")
if numeric_ordinal_features:
    num_imputer = SimpleImputer(strategy='median')
    X[numeric_ordinal_features] = num_imputer.fit_transform(X[numeric_ordinal_features])
    print(f"Imputed numeric/ordinal features using median.")
else:
     print("No numeric/ordinal features to impute.")

if nominal_categorical_features:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[nominal_categorical_features] = cat_imputer.fit_transform(X[nominal_categorical_features])
    print(f"Imputed nominal categorical features using mode.")
else:
    print("No nominal categorical features to impute.")

if X.isnull().sum().sum() > 0:
     print("WARNING: NaNs remain after imputation!")
else:
     print("Imputation complete. No NaNs remaining in features.")

# One-Hot Encode Nominal Categorical Features (mirroring script 02)
print(f"\\nOne-hot encoding {len(nominal_categorical_features)} nominal categorical features...")
original_feature_count = X.shape[1]
if nominal_categorical_features:
    X = pd.get_dummies(X, columns=nominal_categorical_features, drop_first=False, dummy_na=False)
    print(f"Features shape after one-hot encoding: {X.shape}")
else:
    print("No nominal categorical features to encode.")

encoded_feature_names = X.columns.tolist() # Get feature names after encoding

# --- Split Data ---\nprint(f"\\nSplitting data into train/test sets (Test size: {TEST_SIZE}, Random State: {RANDOM_STATE})...")
# Stratify based on the *ordinal* target variable
X_train, X_test, y_train_ordinal, y_test_ordinal = train_test_split(
    X, y_ordinal, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_ordinal
)

print(f"\\nTraining set shape: X={X_train.shape}, y={y_train_ordinal.shape}")
print(f"Testing set shape: X={X_test.shape}, y={y_test_ordinal.shape}")
print(f"\\nOrdinal Class distribution in y_train_ordinal:\\n{y_train_ordinal.value_counts(normalize=True).sort_index()}")
print(f"\\nOrdinal Class distribution in y_test_ordinal:\\n{y_test_ordinal.value_counts(normalize=True).sort_index()}")

# --- Train Ordinal Logistic Regression Model ---\nprint("\\nTraining Ordinal Logistic Regression model (LogisticAT)...")
# LogisticAT: All-Threshold variant, often a good default
# alpha=0 means no regularization (equivalent to standard Logistic Regression)
# Consider adding regularization (e.g., alpha > 0) if needed, similar to Ridge
model = mord.LogisticAT(alpha=0) 

try:
    model.fit(X_train, y_train_ordinal)
    print("Model training complete.")
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# --- Evaluate Model ---\nprint("\\nEvaluating model on the test set...")
y_pred_ordinal = model.predict(X_test)

# --- Metrics ---\nprint("\\n--- Performance Metrics ---")

# Overall Accuracy (Exact Match)
accuracy = accuracy_score(y_test_ordinal, y_pred_ordinal)
print(f"Overall Accuracy (Exact Match): {accuracy:.4f}")

# Mean Absolute Error (Lower is better, measures average distance between predicted and true class)
mae = mean_absolute_error(y_test_ordinal, y_pred_ordinal)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Classification Report (Treating as multi-class for precision/recall/f1 per class)
print("\\nClassification Report (treating ordinal as multi-class):")
# Ensure labels are sorted correctly for the report
target_names = [f'Class {i}' for i in sorted(y_ordinal.unique())]
try:
    class_report = classification_report(y_test_ordinal, y_pred_ordinal, target_names=target_names)
    print(class_report)
    # Save classification report
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(class_report)
except Exception as e:
    print(f"Could not generate classification report: {e}")

# Confusion Matrix
print("\\nConfusion Matrix:")
cm = confusion_matrix(y_test_ordinal, y_pred_ordinal)
print(cm)
# Save confusion matrix
np.savetxt(os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'), cm, delimiter=',', fmt='%d')

# --- Feature Coefficients ---\nprint("\\n--- Model Coefficients ---")
try:
    # mord models store coefficients similar to sklearn
    coefficients = model.coef_[0] 
    # Create a DataFrame for easier viewing/saving
    # Ensure feature names match the coefficients
    if len(encoded_feature_names) == len(coefficients):
        coeffs_df = pd.DataFrame({
            'feature': encoded_feature_names,
            'coefficient': coefficients
        })
        # Sort by absolute magnitude
        coeffs_df['abs_coefficient'] = coeffs_df['coefficient'].abs()
        coeffs_df_sorted = coeffs_df.sort_values('abs_coefficient', ascending=False).drop(columns=['abs_coefficient'])
        
        print(coeffs_df_sorted.head(20)) # Print top 20
        
        # Save coefficients
        coeffs_df_sorted.to_csv(os.path.join(OUTPUT_DIR, 'ordinal_coefficients.csv'), index=False)
        print(f"Coefficients saved to {os.path.join(OUTPUT_DIR, 'ordinal_coefficients.csv')}")
    else:
        print(f"WARNING: Mismatch between number of features ({len(encoded_feature_names)}) and coefficients ({len(coefficients)}). Cannot reliably map coefficients.")
        print("Coefficients array:")
        print(coefficients)

except Exception as e:
    print(f"Could not extract or save coefficients: {e}")

# --- Save the Model ---
model_filename = os.path.join(OUTPUT_DIR, 'ordinal_logistic_model.joblib')
print(f"\\nSaving the trained model to {model_filename}...")
try:
    joblib.dump(model, model_filename)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

# --- Save Predictions ---\nprint(f"\\nSaving predictions to {OUTPUT_DIR}...")
try:
    # Create a DataFrame with true and predicted values
    predictions_df = pd.DataFrame({
        'y_true_ordinal': y_test_ordinal,
        'y_pred_ordinal': y_pred_ordinal
    })
    predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
    print("Predictions saved successfully.")
except Exception as e:
    print(f"Error saving predictions: {e}")


print("\\nOrdinal Logistic Regression script finished.") 