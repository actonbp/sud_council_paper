import pandas as pd
import numpy as np
import os
import yaml
from sentence_transformers import SentenceTransformer

# --- Configuration ---
RAW_DATA_PATH = 'data/survey/survey_raw.csv'
PROCESSED_DIR = 'data/processed'
PROCESSED_FILE_PATH = os.path.join(PROCESSED_DIR, 'survey_processed.csv')
CONFIG_PATH = 'config/study1_config.yaml'
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

# --- Load Config ---
print(f"Loading configuration from {CONFIG_PATH}...")
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    numeric_cols = config['numeric_cols']
    categorical_cols = config['categorical_cols']
    likert_cols = config['likert_cols']
    drop_cols = config['drop_cols']
    dv_col = config['dv_col']
except FileNotFoundError:
    print(f"Error: Configuration file not found at {CONFIG_PATH}")
    exit()
except KeyError as e:
    print(f"Error: Missing key in configuration file: {e}")
    exit()
except Exception as e:
    print(f"Error loading configuration: {e}")
    exit()

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Load Data ---
print(f"Loading raw data from {RAW_DATA_PATH}...")
try:
    df = pd.read_csv(RAW_DATA_PATH, skiprows=[1, 2])
    print(f"Raw data loaded: {df.shape}")
except FileNotFoundError:
    print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error loading raw data: {e}")
    exit()

# --- Preprocessing Steps ---
print("\nStarting preprocessing...")

# 1. Drop specified columns
print(f"Dropping columns: {len(drop_cols)}")
df_processed = df.drop(columns=drop_cols, errors='ignore')

# 1.5 Rename demo_gender to gender_identity for consistency with script 02 grouping
if 'demo_gender' in df_processed.columns:
    print("Renaming 'demo_gender' to 'gender_identity'")
    df_processed = df_processed.rename(columns={'demo_gender': 'gender_identity'})
    # Update categorical_cols list if rename happened
    if 'demo_gender' in categorical_cols:
        categorical_cols.remove('demo_gender')
        if 'gender_identity' not in categorical_cols:
             categorical_cols.append('gender_identity')

# ***** ADDED: Include mh_not_sud *****
# Treat 'mh_not_sud' as a categorical predictor if it exists
if 'mh_not_sud' in df_processed.columns and 'mh_not_sud' not in categorical_cols:
    print("Adding 'mh_not_sud' to categorical columns list.")
    categorical_cols.append('mh_not_sud')
# ***** END ADDED SECTION *****

# 2. Rename DV column for clarity
print(f"Renaming DV column '{dv_col}' to 'interest_dv'")
df_processed = df_processed.rename(columns={dv_col: 'interest_dv'})

# ******* ADDED: Manually map text Likert scales to numeric *******
print("\nApplying manual mappings for text-based Likert scales...")

# Define mappings (based on dictionary/raw data inspection)
# Note: Raw data has full sentences for career_2, need exact text if possible
# Assuming standard phrasing for now, may need refinement
interest_map = {
    "I am not interested in becoming a substance use disorder counselor.": 1,
    "I am slightly interested in becoming a substance use disorder counselor.": 2,
    "I am moderately interested in becoming a substance use disorder counselor.": 3,
    "I am definitely interested in becoming a substance use disorder counselor.": 4,
    "I prefer not to answer": np.nan # Map non-response to NaN
}

wellbeing_map = {
    "Not at all Stressful": 1,
    "A Little Stressful": 2,
    "Moderately Stressful": 3,
    "Very Stressful": 4,
    "Extremely Stressful": 5,
    "I prefer not to answer": np.nan # Map non-response to NaN
}

# Apply mapping to the DV column (now named 'interest_dv')
if 'interest_dv' in df_processed.columns:
    original_dv_type = df_processed['interest_dv'].dtype
    df_processed['interest_dv'] = df_processed['interest_dv'].map(interest_map).fillna(df_processed['interest_dv'])
    print(f"Applied mapping to 'interest_dv'. Original type: {original_dv_type}, New type: {df_processed['interest_dv'].dtype}")
    # Convert to numeric after mapping, coercing any non-matches not caught by map to NaN
    df_processed['interest_dv'] = pd.to_numeric(df_processed['interest_dv'], errors='coerce')
    print(f"Coerced 'interest_dv' to numeric. NaN count: {df_processed['interest_dv'].isnull().sum()}")

    # ***** ADDED: Binarize the DV (0 = No interest, 1 = Any interest) *****
    print("Binarizing 'interest_dv': 0 for original value 1, 1 for original values > 1")
    # Keep the original numeric column, create a new binary one
    df_processed['interest_dv_binary'] = df_processed['interest_dv'].apply(lambda x: 0 if x == 1 else (1 if x > 1 else np.nan))
    # Drop the original multi-level DV and rename the binary one
    # df_processed = df_processed.drop(columns=['interest_dv'])
    # df_processed = df_processed.rename(columns={'interest_dv_binary': 'interest_dv'})
    # print(f"Created binary 'interest_dv'. Value counts:\\n{df_processed['interest_dv'].value_counts(dropna=False)}")
    print(f"Created new binary column 'interest_dv_binary'. Value counts:\\n{df_processed['interest_dv_binary'].value_counts(dropna=False)}")
    # ***** END ADDED SECTION *****

else:
    print("Warning: DV column 'interest_dv' not found for mapping.")

# Apply mapping to wellbeing columns
wellbeing_cols = [f'wellbeing_{i}' for i in range(1, 11)]
for col in wellbeing_cols:
    if col in df_processed.columns:
        original_wb_type = df_processed[col].dtype
        df_processed[col] = df_processed[col].map(wellbeing_map).fillna(df_processed[col])
        print(f"Applied mapping to '{col}'. Original type: {original_wb_type}, New type: {df_processed[col].dtype}")
        # Convert to numeric after mapping, coercing any non-matches not caught by map to NaN
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        print(f"Coerced '{col}' to numeric. NaN count: {df_processed[col].isnull().sum()}")
    else:
        print(f"Warning: Wellbeing column '{col}' not found for mapping.")

# --- ADD MAPPINGS FOR career_1 and demo_people ---

career_1_map = {
    "Reading the above information is the extent of my familiarity with the profession.": 1, # Treat as 'No familiarity'
    "I am slightly familiar with the profession.": 2,
    "I am moderately familiar with the profession.": 3,
    "I am very familiar with the profession.": 4,
    "I prefer not to answer": np.nan
}

demo_people_map = {
    "Less than once a week": 1,
    "1 or 2 times a week": 2,
    "3 to 5 times a week": 3,
    "6 or more times a week": 4,
    "I prefer not to answer": np.nan
}

# Apply mapping to career_1
if 'career_1' in df_processed.columns:
    original_c1_type = df_processed['career_1'].dtype
    df_processed['career_1'] = df_processed['career_1'].map(career_1_map).fillna(df_processed['career_1'])
    print(f"Applied mapping to 'career_1'. Original type: {original_c1_type}, New type: {df_processed['career_1'].dtype}")
    df_processed['career_1'] = pd.to_numeric(df_processed['career_1'], errors='coerce')
    print(f"Coerced 'career_1' to numeric. NaN count: {df_processed['career_1'].isnull().sum()}")
else:
    print("Warning: Column 'career_1' not found for mapping.")

# Apply mapping to demo_people
if 'demo_people' in df_processed.columns:
    original_dp_type = df_processed['demo_people'].dtype
    df_processed['demo_people'] = df_processed['demo_people'].map(demo_people_map).fillna(df_processed['demo_people'])
    print(f"Applied mapping to 'demo_people'. Original type: {original_dp_type}, New type: {df_processed['demo_people'].dtype}")
    df_processed['demo_people'] = pd.to_numeric(df_processed['demo_people'], errors='coerce')
    print(f"Coerced 'demo_people' to numeric. NaN count: {df_processed['demo_people'].isnull().sum()}")
else:
    print("Warning: Column 'demo_people' not found for mapping.")

# ******* END ADDED SECTION *******

# 3. Convert Likert scale columns (assuming they are already numerically coded or can be coerced)
print(f"\nProcessing Likert columns (post-mapping): {len(likert_cols)}")
for col in likert_cols:
    if col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    else:
        print(f"Warning: Likert column '{col}' not found in DataFrame.")

# 4. Convert explicitly numeric columns
print(f"Processing numeric columns: {len(numeric_cols)}")
for col in numeric_cols:
    if col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    else:
        print(f"Warning: Numeric column '{col}' not found in DataFrame.")

# 5. Convert categorical columns to string type for consistent handling
print(f"Processing categorical columns: {len(categorical_cols)}")
for col in categorical_cols:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].astype(str)
    else:
        print(f"Warning: Categorical column '{col}' not found in DataFrame.")

# --- Handle Missing Values (Example: Simple median/mode imputation) ---
print("\nHandling missing values...")
numeric_and_likert_cols_present = [col for col in numeric_cols + likert_cols if col in df_processed.columns]
categorical_cols_present = [col for col in categorical_cols if col in df_processed.columns]

# Impute numeric/Likert with median
for col in numeric_and_likert_cols_present:
    if df_processed[col].isnull().any():
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        print(f"Imputed missing values in '{col}' with median ({median_val})")

# Impute categorical with mode (or a constant like 'Missing')
for col in categorical_cols_present:
    # Check for actual NaN or string representations
    if df_processed[col].isnull().any() or df_processed[col].astype(str).isin(['nan', 'None', '', ' ']).any():
        # Fill actual NaNs first if any
        if df_processed[col].isnull().any():
             mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Missing'
             df_processed[col].fillna(mode_val, inplace=True)
             print(f"Imputed actual NaN in '{col}' with mode ('{mode_val}')")

        # Replace string representations of missingness
        missing_strings = ['nan', 'None', '', ' ']
        mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Missing' # Recalculate mode in case NaNs changed it
        df_processed[col] = df_processed[col].astype(str).replace(missing_strings, mode_val)
        print(f"Replaced string representations of missingness in '{col}' with mode ('{mode_val}')")

# --- Final Column Selection --- 
print("\nSelecting final columns based on config...")
# Ensure interest_dv_binary is also included if created
final_cols = numeric_cols + categorical_cols + likert_cols + ['interest_dv'] 
if 'interest_dv_binary' in df_processed.columns:
    final_cols.append('interest_dv_binary')

# Ensure all columns exist before selecting
final_cols = [col for col in final_cols if col in df_processed.columns]
missing_final_cols = [col for col in (numeric_cols + categorical_cols + likert_cols) if col not in final_cols]
if missing_final_cols:
    print(f"Warning: The following configured columns were not found in the processed data and will be excluded: {missing_final_cols}")
df_processed = df_processed[final_cols]
print(f"Selected {len(df_processed.columns)} final columns.")

# --- Final Check ---
print("\nFinal check for missing values after processing:")
print(df_processed.isnull().sum().sort_values(ascending=False).head())

# --- Save Processed Data ---
print(f"\nSaving processed data to {PROCESSED_FILE_PATH}...")
try:
    df_processed.to_csv(PROCESSED_FILE_PATH, index=False)
    print(f"Processed data saved successfully: {df_processed.shape}")
except Exception as e:
    print(f"Error saving processed data: {e}")

print("\nPreprocessing complete.") 