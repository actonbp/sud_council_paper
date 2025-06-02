# 02_preprocess_for_ml.R
# This script mirrors the functionality of the Python 02_preprocess_for_ml.py script
# It prepares the processed survey data for machine learning analysis

# Load required packages
library(tidyverse)
library(recipes)
library(rsample)
library(here)

# Configuration
input_file <- here("data", "processed", "survey_processed.csv")
output_dir <- here("data", "processed", "r")
target_column <- "sud_counselor_interest"
new_target_column <- "interest_dv"
test_size <- 0.2
random_state <- 42

# Create output directory if it doesn't exist
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("Starting ML preprocessing...\n")

# Load processed data
cat(paste0("Loading data from ", input_file, "...\n"))
tryCatch({
  df <- read_csv(input_file, show_col_types = FALSE)
  cat(paste0("Loaded ", nrow(df), " rows and ", ncol(df), " columns.\n"))
}, error = function(e) {
  stop(paste0("Error loading input file: ", e$message))
})

# Create binary target variable
cat(paste0("Creating binary target variable '", new_target_column, "' from '", target_column, "'...\n"))

# Map original ordinal values (1=No, 2=Slight, 3=Mod, 4=Def) to binary (0=No, 1=Any)
# NaN values (originally 'Prefer not to answer') will remain NA
interest_map_binary <- c(
  "1" = 0,  # Not interested
  "2" = 1,  # Slightly interested
  "3" = 1,  # Moderately interested
  "4" = 1   # Definitely interested
)

df <- df %>%
  mutate(!!new_target_column := if_else(!is.na(!!sym(target_column)), 
                                       interest_map_binary[as.character(!!sym(target_column))], 
                                       NA_real_))

# Drop rows with missing target
original_rows <- nrow(df)
df <- df %>% filter(!is.na(!!sym(new_target_column)))
rows_after_target_drop <- nrow(df)

cat(paste0("Dropped ", original_rows - rows_after_target_drop, " rows with missing target variable ('", target_column, "' was NA).\n"))
cat(paste0("Dataset size for ML: ", rows_after_target_drop, " rows.\n"))

# Convert target to integer
df <- df %>% mutate(!!new_target_column := as.integer(!!sym(new_target_column)))

# Separate features (X) and target (y)
cat("Separating features (X) and target (y)...\n")

# Define potential columns to remove for X
potential_cols_to_remove <- c(
  target_column,
  "careless_responder",
  "completed",
  "progress"
)

# Filter list to only include columns actually present in the dataframe
cols_to_remove_for_X <- intersect(potential_cols_to_remove, names(df))

# Ensure target column is always removed if present
if(target_column %in% names(df) && !(target_column %in% cols_to_remove_for_X)) {
  cols_to_remove_for_X <- c(cols_to_remove_for_X, target_column)
}

cat(paste0("Columns being removed to create feature set X: ", paste(cols_to_remove_for_X, collapse = ", "), "\n"))

X <- df %>% select(-all_of(cols_to_remove_for_X), -!!sym(new_target_column))
y <- df %>% pull(!!sym(new_target_column))

cat(paste0("Features (X) shape: ", nrow(X), " rows, ", ncol(X), " columns\n"))

# Group sparse categories
cat("\nGrouping sparse categories for 'race' and 'gender_identity'...\n")

# Define categories to group
race_sparse_cats <- c("Black", "Other (please specify):", "Middle Eastern", "I prefer not to answer")
gender_sparse_cats <- c("I prefer not to answer", "Nonbinary", "Gender queer", "Transgender", "Agender")

# Apply grouping for race
if("race" %in% names(X)) {
  cat(paste0("Original 'race' categories: ", paste(unique(X$race), collapse = ", "), "\n"))
  X <- X %>% mutate(race = ifelse(race %in% race_sparse_cats, "Other/Multiple/Unknown Race", race))
  cat(paste0("New 'race' categories: ", paste(unique(X$race), collapse = ", "), "\n"))
} else {
  cat("'race' column not found for grouping.\n")
}

# Apply grouping for gender_identity
if("gender_identity" %in% names(X)) {
  cat(paste0("Original 'gender_identity' categories: ", paste(unique(X$gender_identity), collapse = ", "), "\n"))
  X <- X %>% mutate(gender_identity = ifelse(gender_identity %in% gender_sparse_cats, "Gender Diverse/Unknown", gender_identity))
  cat(paste0("New 'gender_identity' categories: ", paste(unique(X$gender_identity), collapse = ", "), "\n"))
} else {
  cat("'gender_identity' column not found for grouping.\n")
}

# Identify feature types
# Identify columns with character type (nominal categorical)
nominal_categorical_features <- X %>%
  select(where(is.character)) %>%
  names()

# Identify numeric/ordinal columns
numeric_ordinal_features <- X %>%
  select(where(is.numeric)) %>%
  names()

cat(paste0("\nIdentified ", length(nominal_categorical_features), " nominal categorical features:\n"))
cat(paste(nominal_categorical_features, collapse = ", "), "\n")

cat(paste0("\nIdentified ", length(numeric_ordinal_features), " numeric/ordinal features:\n"))
cat(paste(numeric_ordinal_features, collapse = ", "), "\n")

# Create a recipe for preprocessing
cat("\nCreating preprocessing recipe...\n")

# Initialize recipe
rec <- recipe(~ ., data = X)

# Impute numeric/ordinal with median
if(length(numeric_ordinal_features) > 0) {
  num_cols_to_impute <- X %>% 
    select(all_of(numeric_ordinal_features)) %>% 
    summarise(across(everything(), ~ sum(is.na(.)))) %>% 
    pivot_longer(everything(), names_to = "column", values_to = "missing_count") %>%
    filter(missing_count > 0) %>%
    nrow()
  
  cat(paste0("Imputing ", num_cols_to_impute, " numeric/ordinal columns with median...\n"))
  rec <- rec %>% step_impute_median(all_of(numeric_ordinal_features))
} else {
  cat("No numeric/ordinal features to impute.\n")
}

# Impute nominal categorical with mode (most frequent)
if(length(nominal_categorical_features) > 0) {
  cat_cols_to_impute <- X %>% 
    select(all_of(nominal_categorical_features)) %>% 
    summarise(across(everything(), ~ sum(is.na(.)))) %>% 
    pivot_longer(everything(), names_to = "column", values_to = "missing_count") %>%
    filter(missing_count > 0) %>%
    nrow()
  
  cat(paste0("Imputing ", cat_cols_to_impute, " nominal categorical columns with mode...\n"))
  rec <- rec %>% step_impute_mode(all_of(nominal_categorical_features))
} else {
  cat("No nominal categorical features to impute.\n")
}

# One-hot encode nominal categorical features
if(length(nominal_categorical_features) > 0) {
  cat(paste0("\nOne-hot encoding ", length(nominal_categorical_features), " nominal categorical features...\n"))
  rec <- rec %>% step_dummy(all_of(nominal_categorical_features), one_hot = TRUE)
} else {
  cat("No nominal categorical features to encode.\n")
}

# Prepare the recipe
prepped_rec <- prep(rec)

# Apply the recipe to X
X_processed <- bake(prepped_rec, new_data = NULL)
cat(paste0("Features shape after one-hot encoding: ", nrow(X_processed), " rows, ", ncol(X_processed), " columns\n"))

# Create a dataframe with the target column for splitting
df_for_split <- X_processed %>%
  mutate(!!new_target_column := y)

# Split data
cat(paste0("\nSplitting data into train/test sets (Test size: ", test_size, ", Random State: ", random_state, ")...\n"))

# Set seed for reproducibility
set.seed(random_state)

# Create initial split
split <- initial_split(df_for_split, prop = 1 - test_size, strata = !!sym(new_target_column))
train_data <- training(split)
test_data <- testing(split)

# Separate X and y from the split data
X_train <- train_data %>% select(-!!sym(new_target_column))
y_train <- train_data %>% pull(!!sym(new_target_column))

X_test <- test_data %>% select(-!!sym(new_target_column))
y_test <- test_data %>% pull(!!sym(new_target_column))

cat(paste0("\nTraining set shape: X=", nrow(X_train), " rows, ", ncol(X_train), " columns, y=", length(y_train), " elements\n"))
cat(paste0("Testing set shape: X=", nrow(X_test), " rows, ", ncol(X_test), " columns, y=", length(y_test), " elements\n"))

# Class distribution
cat("\nClass distribution in y_train:\n")
print(table(y_train) / length(y_train))

cat("\nClass distribution in y_test:\n")
print(table(y_test) / length(y_test))

# Save processed data
cat(paste0("\nSaving processed data splits to ", output_dir, "...\n"))

# Save X and y separately
write_csv(X_train, file.path(output_dir, "X_train.csv"))
write_csv(X_test, file.path(output_dir, "X_test.csv"))
write_csv(tibble(y = y_train), file.path(output_dir, "y_train.csv"))
write_csv(tibble(y = y_test), file.path(output_dir, "y_test.csv"))

# Also save combined train and test sets
train_data %>% write_csv(file.path(output_dir, "train_data.csv"))
test_data %>% write_csv(file.path(output_dir, "test_data.csv"))

# Save the recipe as an RDS file
saveRDS(prepped_rec, file.path(output_dir, "preprocessing_recipe.rds"))

cat("Successfully saved data splits and recipe.\n")
cat("\nPreprocessing for ML complete.\n")