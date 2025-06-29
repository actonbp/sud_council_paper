# Required R packages for the SUD Counselor Paper refactoring project
# Install these packages using install.packages() as needed

# Core tidyverse packages
required_packages <- c(
  # Data manipulation
  "tidyverse",     # Main tidyverse meta-package
  "dplyr",         # Data manipulation
  "tidyr",         # Data tidying
  "readr",         # Data import
  "readxl",        # Excel import
  "purrr",         # Functional programming
  "stringr",       # String manipulation
  
  # Machine learning with tidymodels
  "tidymodels",    # Main tidymodels meta-package
  "recipes",       # Feature engineering 
  "parsnip",       # Model specification
  "workflows",     # Modeling workflows
  "tune",          # Hyperparameter tuning
  "rsample",       # Resampling methods
  "dials",         # Hyperparameter definitions
  "yardstick",     # Model metrics
  "broom",         # Model tidying
  "themis",        # Class imbalance handling
  
  # Specific modeling engines
  "glmnet",        # Regularized regression
  "ranger",        # Random forest
  "xgboost",       # XGBoost

  # Visualization
  "ggplot2",       # Grammar of graphics
  "patchwork",     # Plot composition
  "scales",        # Scale functions
  "viridis",       # Color palettes
  "ggrepel",       # Text repelling for labels
  "corrplot",      # Correlation plots
  
  # Parallel processing
  "doParallel",    # Parallel backend
  "foreach",       # Parallel loops
  
  # Results reporting
  "knitr",         # Dynamic report generation
  "kableExtra",    # Table formatting
  "gt",            # Grammar of tables
  "here",          # Project-relative paths
  
  # Text analysis with tidymodels (Study 2 - NEW APPROACH)
  "textrecipes",   # Text preprocessing for tidymodels (MAIN PACKAGE)
  "tidytext",      # Tidy text analysis foundation
  "SnowballC",     # Porter stemming
  "stopwords",     # Additional stopword sources
  
  # Topic modeling engines  
  "topicmodels",   # LDA engine for textrecipes
  "text2vec",      # Alternative text vectorization
  
  # Additional text analysis (for comparison/validation)
  "widyr",         # Pairwise operations for co-occurrence analysis
  "quanteda",      # Text analysis (backup methods)
  "textfeatures",  # Text feature extraction
  "ldatuning",     # LDA tuning for optimal k (if needed)
  "stm",           # Structural topic models (alternative)
  "BTM",           # Biterm Topic Models for short texts (comparison)
  "udpipe",        # NLP tasks
  "umap",          # Dimension reduction
  "factoextra",    # Clustering visualization
  "textstem",      # Text stemming utilities
  
  # Enhanced visualization for text
  "RColorBrewer",  # Color palettes for topic plots
  "wordcloud"      # Word cloud generation (optional)
)

# Check which packages are already installed
installed <- installed.packages()[, "Package"]
missing_packages <- setdiff(required_packages, installed)

if (length(missing_packages) > 0) {
  cat("Missing packages that need to be installed:\n")
  cat(paste(missing_packages, collapse = ", "), "\n")
  
  # Uncomment to install missing packages
  # install.packages(missing_packages)
} else {
  cat("All required packages are already installed.\n")
}

# Load the core packages
library(tidyverse)
library(tidymodels)