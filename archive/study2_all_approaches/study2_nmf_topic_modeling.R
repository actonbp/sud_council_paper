# =============================================================================
# Study 2 (Qualitative): NMF Topic Modeling with tidytext pipeline
# SUD Counseling Career Research Project – Focus-group utterances
# -----------------------------------------------------------------------------
# This script replaces the heavy SBERT + k-means pipeline with a lightweight,
# pure-R approach using Non-negative Matrix Factorisation (NMF) on a TF–IDF
# matrix.  It is fully reproducible with CRAN packages only and fits naturally
# into the tidytext / tidymodels workflow.
# -----------------------------------------------------------------------------
# Steps
# 1. Load substantive utterances (output of study2_data_preparation.R)
# 2. Tokenise + build TF-IDF with text2vec (pipe-friendly)
# 3. Fit NMF for k = 2–8, record reconstruction error
# 4. Choose k via elbow (largest relative drop, then flatten)
# 5. Extract topic-term (H) and document-topic (W) matrices
# 6. Save tidy outputs in results/r/study2_nmf_modeling/
# -----------------------------------------------------------------------------
# Author: Auto-generated 2025-06-11
# =============================================================================

# ---- Packages ----------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,      # dplyr, readr, etc.
  tidytext,       # tidy tokenisation helpers
  text2vec,       # fast TF-IDF + vocabulary
  NNLM,           # Non-negative Matrix Factorisation (pure R, CRAN)
  here            # path management
)
set.seed(123)

# ---- Directories -------------------------------------------------------------
results_dir <- here("results", "r", "study2_nmf_modeling")
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

data_file <- here("data", "focus_group_substantive.csv")

# ---- 1. Load data ------------------------------------------------------------
substantive_data <- read_csv(data_file, show_col_types = FALSE)
cat("Loaded", nrow(substantive_data), "substantive utterances\n")

# ---- 2. Tokenise + TF-IDF ----------------------------------------------------
cat("Building TF–IDF matrix...\n")

# text2vec expects iterator over tokens
it <- itoken(substantive_data$cleaned_text,
             tokenizer = word_tokenizer,
             progressbar = FALSE)

vocab <- create_vocabulary(it, stopwords = stop_words$word)
# prune very rare / very common terms
vocab <- prune_vocabulary(vocab, term_count_min = 3, doc_proportion_max = 0.6)

vectoriser <- vocab_vectorizer(vocab)
term_doc <- create_dtm(it, vectoriser)

# TF-IDF weighting
tfidf <- TfIdf$new()
X <- fit_transform(term_doc, tfidf)
cat("DTM dimensions:", dim(X)[1], "docs ×", dim(X)[2], "terms\n")

# ---- 3. Fit NMF for k = 2–8 --------------------------------------------------
ks <- 2:8
nmf_models <- list()
reconstruction_err <- numeric(length(ks))
for (i in seq_along(ks)) {
  k <- ks[i]
  nmf_models[[i]] <- NNLM::nnmf(X, k = k, method = "scd", loss = "mse", max.iter = 300, check.k = FALSE)
  reconstruction_err[i] <- tail(nmf_models[[i]]$mse, 1)
  cat("k =", k, "→ MSE =", round(reconstruction_err[i], 4), "\n")
}

model_comp <- tibble(k = ks, residual = reconstruction_err)
write_csv(model_comp, file.path(results_dir, "nmf_model_comparison.csv"))

# ---- 4. Choose k via elbow ---------------------------------------------------
# biggest relative drop then flatten criterion
rel_drop <- c(NA, diff(reconstruction_err)) / reconstruction_err[-length(reconstruction_err)]
chosen_k <- ks[which.min(reconstruction_err)]
# heuristic: if residual flattens after first big drop, take smaller k
if (chosen_k > 2) {
  sdrop <- abs(rel_drop)
  elbow <- which.max(sdrop[-1]) + 1 # index shift
  chosen_k <- ks[elbow]
}
cat("Chosen k via elbow =", chosen_k, "topics\n")
final_model <- nmf_models[[which(ks == chosen_k)]]

# ---- 5. Extract tidy results -------------------------------------------------
# final_model$W : n_docs x k, final_model$H : k x n_terms
H_mat <- final_model$H
W_mat <- final_model$W
colnames(H_mat) <- colnames(X)

# top terms per topic
term_df <- map_dfr(seq_len(nrow(H_mat)), function(t) {
  beta <- H_mat[t, ]
  top_idx <- order(beta, decreasing = TRUE)[1:10]
  tibble(topic = t,
         term = colnames(H_mat)[top_idx],
         beta = beta[top_idx])
})

write_csv(term_df, file.path(results_dir, "top_terms_per_topic.csv"))

# document-topic weights
doc_topic <- as_tibble(W_mat) %>%
  mutate(document = row_number()) %>%
  pivot_longer(-document, names_to = "topic", values_to = "weight")

write_csv(doc_topic, file.path(results_dir, "document_topic_weights.csv"))

# summary
summary_df <- tibble(
  analysis_date = Sys.Date(),
  chosen_k = chosen_k,
  residual_error = reconstruction_err[which(ks == chosen_k)]
)
write_csv(summary_df, file.path(results_dir, "nmf_model_summary.csv"))

cat("\n=== NMF Topic Modeling Complete ===\n")
cat("Documents:", dim(X)[1], "  Terms:", dim(X)[2], "\n")
cat("Chosen k:", chosen_k, "topics  Residual:", round(summary_df$residual_error, 3), "\n")
cat("Results written to", results_dir, "\n") 