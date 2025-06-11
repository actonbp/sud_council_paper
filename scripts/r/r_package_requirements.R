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
  
  # For qualitative analysis with tidytext (Study 2)
  "tidytext",      # Tidy text analysis
  "SnowballC",     # Porter stemming
  "widyr",         # Pairwise operations for co-occurrence analysis
  "quanteda",      # Text analysis
  "text2vec",      # Word embeddings
  "textfeatures",  # Text feature extraction
  "topicmodels",   # Topic modeling
  "ldatuning",     # LDA tuning for optimal k
  "stm",           # Structural topic models
  "udpipe",        # NLP tasks
  "umap",          # Dimension reduction
  "factoextra",    # Clustering visualization
  "BTM",           # Biterm Topic Models for short texts
  "textstem"       # Text stemming utilities
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