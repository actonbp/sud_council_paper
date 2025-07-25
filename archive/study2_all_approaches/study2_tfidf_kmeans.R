# =============================================================================
# Study 2 (Qualitative): TF–IDF + SVD + K-Means Clustering
# -----------------------------------------------------------------------------
# Pure-R pipeline, CRAN only – no Bioconductor / Python / Java.
# 1. Load substantive utterances
# 2. Build TF-IDF matrix (text2vec)
# 3. Reduce to 50 dims via sparse SVD (RSpectra)
# 4. Evaluate k-means for k = 2-8 using silhouette; pick best k
# 5. Label clusters with top tf-idf words
# 6. Save results under results/r/study2_tfidf_kmeans/
# -----------------------------------------------------------------------------
# Author: autogenerated 2025-06-11
# =============================================================================

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, tidytext, text2vec, RSpectra, cluster, here, SnowballC)

set.seed(123)

results_dir <- here("results", "r", "study2_tfidf_kmeans")
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

data <- read_csv(here("data", "focus_group_substantive.csv"), show_col_types = FALSE)
cat("Loaded", nrow(data), "utterances\n")

# Build TF-IDF
extra_stop <- c(
  # career-specific words to exclude
  "mental", "health", "job", "career", "field", "counsel", "counselor", 
  "counselors", "psychology", "psychologist", "psych", "patient", "therapy",
  "therapist", "clinic", "nurs", "medical", "medicine", "work", "profession",
  # filler/discourse words we previously saw dominating
  "people", "yeah", "don", "lot", "guess", "ve", "feel", "gotcha",
  "uh", "um", "okay", "alright", "sorta", "kinda", "stuff", "thing", "things",
  "really", "actually", "basically", "going", "gonna", "just", "like", "think", "know", "see"
)

custom_tokeniser <- function(x){
  tokens <- word_tokenizer(x)
  lapply(tokens, function(t){
    t <- SnowballC::wordStem(t, language = "en")
    t[!(t %in% extra_stop) & !(t %in% stop_words$word) & !grepl("[0-9]", t)]
  })
}

it <- itoken(data$cleaned_text, tokenizer = custom_tokeniser, progressbar = FALSE)
vocab <- create_vocabulary(it, stopwords = stop_words$word) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.6)
vec  <- vocab_vectorizer(vocab)
DTM  <- create_dtm(it, vec)
TFIDF <- TfIdf$new()
X <- fit_transform(DTM, TFIDF)
cat("DTM dims:", dim(X)[1], "x", dim(X)[2], "\n")

# ---- SVD to 50 dims ----------------------------------------------------------
cat("Computing SVD (rank 50)...\n")
rank <- min(50, dim(X)[2]-1)
svd_res <- RSpectra::svds(X, k = rank)
X_red  <- svd_res$u %*% diag(svd_res$d)   # n_docs x rank

# ---- K-means grid search -----------------------------------------------------
ks <- 2:8
sil <- numeric(length(ks))
models <- vector("list", length(ks))
for(i in seq_along(ks)){
  k <- ks[i]
  km <- kmeans(X_red, centers = k, nstart = 25)
  sil[i] <- mean(cluster::silhouette(km$cluster, dist(X_red))[,3])
  models[[i]] <- km
  cat("k=", k, "silhouette=", round(sil[i],3), "\n")
}
model_comp <- tibble(k = ks, silhouette = sil)
write_csv(model_comp, file.path(results_dir, "kmeans_silhouette.csv"))

best_k <- model_comp %>% slice_max(silhouette, with_ties = FALSE) %>% pull(k)
km_best <- models[[which(ks==best_k)]]
cat("Best k by silhouette:", best_k, "\n")

# ---- Label clusters ----------------------------------------------------------
assignments <- data %>% mutate(cluster = km_best$cluster)

# top terms per cluster via tf-idf on filtered tokens
cluster_terms <- assignments %>%
  mutate(doc_id = row_number()) %>%
  unnest_tokens(word, cleaned_text) %>%
  mutate(word = SnowballC::wordStem(word, language = "en")) %>%
  filter(!(word %in% extra_stop)) %>%
  anti_join(stop_words, by="word") %>%
  count(cluster, word) %>%
  bind_tf_idf(word, cluster, n) %>%
  group_by(cluster) %>%
  slice_max(tf_idf, n = 10) %>%
  ungroup()

write_csv(assignments %>% select(response_id, cluster, cleaned_text),
          file.path(results_dir, "utterance_cluster_assignments.csv"))
write_csv(cluster_terms, file.path(results_dir, "top_terms_per_cluster.csv"))

cat("\n=== TF-IDF K-Means Complete ===\n")
cat("Best k:", best_k, "  Avg silhouette:", round(max(sil),3), "\n")
cat("Results in:", results_dir, "\n") 