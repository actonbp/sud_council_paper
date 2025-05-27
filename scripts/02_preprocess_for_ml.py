# scripts/02_preprocess_for_ml.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os

# --- Configuration ---
INPUT_FILE = 'data/processed/survey_processed.csv'
OUTPUT_DIR = 'data/processed'
TARGET_COLUMN = 'interest_dv' # Target is already named interest_dv from script 01
NEW_TARGET_COLUMN = 'interest_dv' # Keep the same name
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Load Data ---
print(f"Loading data from {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_FILE}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Create Binary Target Variable ---
print(f"Creating binary target variable '{NEW_TARGET_COLUMN}' from '{TARGET_COLUMN}'...")

# Map original ordinal values (1=No, 2=Slight, 3=Mod, 4=Def) to binary (0=No, 1=Any)
# NaN values (originally 'Prefer not to answer') will remain NaN for now
interest_map_binary = {
    1.0: 0, # Not interested
    2.0: 1, # Slightly interested
    3.0: 1, # Moderately interested
    4.0: 1  # Definitely interested
}
df[NEW_TARGET_COLUMN] = df[TARGET_COLUMN].map(interest_map_binary)

# --- Drop Rows with Missing Target ---
original_rows = len(df)
df.dropna(subset=[NEW_TARGET_COLUMN], inplace=True)
rows_after_target_drop = len(df)
print(f"Dropped {original_rows - rows_after_target_drop} rows with missing target variable ('{TARGET_COLUMN}' was NaN).")
print(f"Dataset size for ML: {rows_after_target_drop} rows.")

# Convert target to integer
df[NEW_TARGET_COLUMN] = df[NEW_TARGET_COLUMN].astype(int)

# --- Separate Features (X) and Target (y) ---
print("Separating features (X) and target (y)...")
y = df[NEW_TARGET_COLUMN]

# Define potential columns to remove for X
potential_cols_to_remove = [
    TARGET_COLUMN, 
    # NEW_TARGET_COLUMN, # Redundant if same as TARGET_COLUMN
    'careless_responder', 
    'completed',
    'progress' 
]

# Filter list to only include columns actually present in the dataframe
cols_to_remove_for_X = [col for col in potential_cols_to_remove if col in df.columns]
# Ensure target column is always removed if present, even if names differ slightly
if TARGET_COLUMN in df.columns and TARGET_COLUMN not in cols_to_remove_for_X:
    cols_to_remove_for_X.append(TARGET_COLUMN)

print(f"Columns being removed to create feature set X: {cols_to_remove_for_X}")

X = df.drop(columns=cols_to_remove_for_X)
print(f"Features (X) shape: {X.shape}")

# --- Group Sparse Categories --- 
print("\nGrouping sparse categories for 'race' and 'gender_identity'...")

# Define the categories to group
race_sparse_cats = ['Black', 'Other (please specify):', 'Middle Eastern', 'I prefer not to answer']
gender_sparse_cats = ['I prefer not to answer', 'Nonbinary', 'Gender queer', 'Transgender', 'Agender']

# Apply grouping for race
if 'race' in X.columns:
    print(f"Original 'race' categories: {X['race'].unique()}")
    X['race'] = X['race'].replace(race_sparse_cats, 'Other/Multiple/Unknown Race')
    print(f"New 'race' categories: {X['race'].unique()}")
else:
    print("'race' column not found for grouping.")

# Apply grouping for gender_identity
if 'gender_identity' in X.columns:
    print(f"Original 'gender_identity' categories: {X['gender_identity'].unique()}")
    X['gender_identity'] = X['gender_identity'].replace(gender_sparse_cats, 'Gender Diverse/Unknown')
    print(f"New 'gender_identity' categories: {X['gender_identity'].unique()}")
else:
    print("'gender_identity' column not found for grouping.")

# --- Identify Feature Types ---
# Identify columns still containing strings (nominal categorical)
# Exclude potential numerical columns misidentified as object due to NaNs if any (check dtypes)
nominal_categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Identify numeric/ordinal columns (already numeric from previous step)
numeric_ordinal_features = X.select_dtypes(include=['number']).columns.tolist()

print(f"\nIdentified {len(nominal_categorical_features)} nominal categorical features:")
print(nominal_categorical_features)
print(f"\nIdentified {len(numeric_ordinal_features)} numeric/ordinal features:")
print(numeric_ordinal_features)

# --- Handle Missing Predictor Data (Imputation) ---
print("\nImputing missing values...")

# Impute numeric/ordinal with median
if numeric_ordinal_features:
    num_cols_to_impute = X[numeric_ordinal_features].isnull().any(axis=0).sum()
    print(f"Imputing {num_cols_to_impute} numeric/ordinal columns with median...")
    num_imputer = SimpleImputer(strategy='median')
    X[numeric_ordinal_features] = num_imputer.fit_transform(X[numeric_ordinal_features])
else:
     print("No numeric/ordinal features to impute.")

# Impute nominal categorical with mode (most frequent)
# This handles NaNs AND any remaining text like "I prefer not to answer" in nominal vars
if nominal_categorical_features:
    cat_cols_to_impute = X[nominal_categorical_features].isnull().any(axis=0).sum()
    print(f"Imputing {cat_cols_to_impute} nominal categorical columns with mode...")
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[nominal_categorical_features] = cat_imputer.fit_transform(X[nominal_categorical_features])
else:
    print("No nominal categorical features to impute.")

# Check if any NaNs remain after imputation
if X.isnull().sum().sum() > 0:
     print("WARNING: NaNs remain after imputation!")
     print(X.isnull().sum()[X.isnull().sum() > 0])
else:
     print("Imputation complete. No NaNs remaining in features.")


# --- Encode Nominal Categorical Features ---
print(f"\nOne-hot encoding {len(nominal_categorical_features)} nominal categorical features...")
if nominal_categorical_features:
    X = pd.get_dummies(X, columns=nominal_categorical_features, drop_first=False, dummy_na=False)
    print(f"Features shape after one-hot encoding: {X.shape}")
else:
    print("No nominal categorical features to encode.")

# --- Split Data ---
print(f"\nSplitting data into train/test sets (Test size: {TEST_SIZE}, Random State: {RANDOM_STATE})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTraining set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing set shape: X={X_test.shape}, y={y_test.shape}")
print(f"\nClass distribution in y_train:\n{y_train.value_counts(normalize=True)}")
print(f"\nClass distribution in y_test:\n{y_test.value_counts(normalize=True)}")

# --- Save Processed Data ---
print(f"\nSaving processed data splits to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, 'y_train.csv'), index=False, header=True)
    y_test.to_csv(os.path.join(OUTPUT_DIR, 'y_test.csv'), index=False, header=True)
    print("Successfully saved data splits.")
except Exception as e:
    print(f"Error saving data splits: {e}")

print("\nPreprocessing for ML complete.") 