# Comprehensive Topic Modeling Method Comparison for Study 2
# Purpose: Test multiple methods and compare interpretability/validity
# Input: data/focus_group_substantive.csv
# Output: Comparison of all methods with evaluation metrics

# Load required libraries
library(tidyverse)
library(tidymodels)
library(textrecipes)
library(tidytext)
library(topicmodels)
library(BTM)
library(SnowballC)
library(NMF)
library(cluster)
library(umap)
library(here)
library(glue)

# Set up paths
data_path <- here("data", "focus_group_substantive.csv")
results_dir <- here("results", "r", "study2_method_comparison")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# Read and preprocess data (consistent across all methods)
cat("\n=== Data Preprocessing (Consistent Across Methods) ===\n")
focus_data <- read_csv(data_path, show_col_types = FALSE)

# SUD terms to remove
sud_terms <- c(
  "substance", "substances", "counselor", "counselors", "counseling",
  "alcohol", "alcoholic", "alcoholism", "drug", "drugs", "addiction", "addictions",
  "sud", "abuse", "disorder", "disorders", "treatment", "treatments", "recovery",
  "addict", "addicts", "addicted", "mental", "health", "therapist", "therapists", 
  "therapy", "therapies", "psychology", "psychologist", "psychologists", "psych",
  "psychiatric", "psychiatrist", "social", "worker", "workers", "nurse", "nurses",
  "nursing", "clinical", "clinician", "practitioner", "practitioners"
)

sud_pattern <- paste0("\\b(", paste(sud_terms, collapse = "|"), ")\\b")

# Clean data
cleaned_data <- focus_data %>%
  mutate(
    cleaned_for_analysis = str_replace_all(cleaned_text, regex(sud_pattern, ignore_case = TRUE), ""),
    cleaned_for_analysis = str_squish(cleaned_for_analysis),
    word_count = str_count(cleaned_for_analysis, "\\w+")
  ) %>%
  filter(word_count >= 5) %>%
  mutate(doc_id = row_number())

cat(glue("Clean dataset: {nrow(cleaned_data)} utterances\n"))

# Function to evaluate cluster quality
evaluate_clusters <- function(cluster_assignments, method_name, top_terms = NULL) {
  if(length(unique(cluster_assignments)) == 1) {
    return(data.frame(
      method = method_name,
      n_clusters = 1,
      silhouette_score = 0,
      interpretability = "Single cluster - not interpretable"
    ))
  }
  
  # For silhouette, we need the feature space (use simple TF-IDF)
  tfidf_matrix <- cleaned_data %>%
    select(doc_id, cleaned_for_analysis) %>%
    unnest_tokens(word, cleaned_for_analysis) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) > 2) %>%
    count(doc_id, word) %>%
    bind_tf_idf(word, doc_id, n) %>%
    select(doc_id, word, tf_idf) %>%
    pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%
    column_to_rownames("doc_id") %>%
    as.matrix()
  
  # Calculate silhouette if we have the right dimensions
  if(nrow(tfidf_matrix) == length(cluster_assignments)) {
    sil_score <- mean(silhouette(cluster_assignments, dist(tfidf_matrix))[, 3])
  } else {
    sil_score <- NA
  }
  
  # Basic cluster statistics
  cluster_sizes <- table(cluster_assignments)
  
  result <- data.frame(
    method = method_name,
    n_clusters = length(unique(cluster_assignments)),
    silhouette_score = round(sil_score, 3),
    min_cluster_size = min(cluster_sizes),
    max_cluster_size = max(cluster_sizes),
    cluster_balance = round(min(cluster_sizes) / max(cluster_sizes), 2)
  )
  
  if(!is.null(top_terms)) {
    result$top_terms_sample <- top_terms
  }
  
  return(result)
}

# Store all results
all_results <- list()
detailed_results <- list()

# =============================================================================
# METHOD 1: BTM (Biterm Topic Model)
# =============================================================================
cat("\n=== METHOD 1: BTM (Biterm Topic Model) ===\n")

try({
  # Tokenize for BTM
  btm_tokens <- cleaned_data %>%
    unnest_tokens(word, cleaned_for_analysis) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) > 2) %>%
    mutate(word = SnowballC::wordStem(word)) %>%
    filter(nchar(word) >= 3) %>%
    count(doc_id, word) %>%
    filter(n >= 2) %>%  # Minimum frequency
    select(doc_id, word) %>%
    as.data.frame()
  
  # Test K=2,3,4 for BTM
  btm_models <- map(2:4, function(k) {
    cat(glue("BTM K={k}..."))
    model <- BTM(btm_tokens, k = k, iter = 1000, trace = FALSE)
    loglik <- logLik(model)$ll
    cat("done\n")
    list(k = k, model = model, loglik = loglik)
  })
  
  # Select best BTM model
  best_btm <- btm_models[[which.max(map_dbl(btm_models, "loglik"))]]
  
  # Get cluster assignments
  btm_topics <- predict(best_btm$model, newdata = btm_tokens)
  btm_clusters <- apply(btm_topics, 1, which.max)
  
  # Get top terms
  btm_terms <- terms(best_btm$model, top_n = 10)
  btm_top_terms <- paste(btm_terms[[1]][1:5], collapse = ", ")
  
  # Evaluate
  btm_eval <- data.frame(
    method = "BTM",
    n_clusters = best_btm$k,
    silhouette_score = NA,  # BTM uses different evaluation
    loglik = round(best_btm$loglik, 1),
    top_terms_sample = btm_top_terms
  )
  
  all_results[["BTM"]] <- btm_eval
  detailed_results[["BTM"]] <- list(
    cluster_assignments = btm_clusters,
    top_terms = btm_terms,
    model = best_btm$model
  )
  
  cat(glue("✓ BTM complete: K={best_btm$k}, LogLik={round(best_btm$loglik, 1)}\n"))
}, silent = TRUE)

# =============================================================================
# METHOD 2: Traditional LDA
# =============================================================================
cat("\n=== METHOD 2: Traditional LDA ===\n")

try({
  # Create document-term matrix for LDA
  lda_dtm <- cleaned_data %>%
    unnest_tokens(word, cleaned_for_analysis) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) > 2) %>%
    count(doc_id, word) %>%
    cast_dtm(doc_id, word, n)
  
  # Test different K values
  lda_models <- map(2:4, function(k) {
    cat(glue("LDA K={k}..."))
    model <- LDA(lda_dtm, k = k, control = list(seed = 123))
    perp <- perplexity(model, lda_dtm)
    cat("done\n")
    list(k = k, model = model, perplexity = perp)
  })
  
  # Select best LDA model (lowest perplexity)
  best_lda <- lda_models[[which.min(map_dbl(lda_models, "perplexity"))]]
  
  # Get cluster assignments
  lda_gamma <- tidy(best_lda$model, matrix = "gamma")
  lda_clusters <- lda_gamma %>%
    group_by(document) %>%
    slice_max(gamma) %>%
    pull(topic)
  
  # Get top terms
  lda_beta <- tidy(best_lda$model, matrix = "beta")
  lda_top_terms <- lda_beta %>%
    group_by(topic) %>%
    slice_max(beta, n = 5) %>%
    filter(topic == 1) %>%
    pull(term) %>%
    paste(collapse = ", ")
  
  # Evaluate
  lda_eval <- data.frame(
    method = "LDA",
    n_clusters = best_lda$k,
    silhouette_score = NA,
    perplexity = round(best_lda$perplexity, 1),
    top_terms_sample = lda_top_terms
  )
  
  all_results[["LDA"]] <- lda_eval
  detailed_results[["LDA"]] <- list(
    cluster_assignments = lda_clusters,
    model = best_lda$model,
    beta = lda_beta,
    gamma = lda_gamma
  )
  
  cat(glue("✓ LDA complete: K={best_lda$k}, Perplexity={round(best_lda$perplexity, 1)}\n"))
}, silent = TRUE)

# =============================================================================
# METHOD 3: Textrecipes TF-IDF + Clustering (our recent approach)
# =============================================================================
cat("\n=== METHOD 3: Textrecipes TF-IDF + Clustering ===\n")

try({
  # Create recipe
  recipe_obj <- recipe(word_count ~ cleaned_for_analysis, data = cleaned_data) %>%
    step_tokenize(cleaned_for_analysis) %>%
    step_stopwords(cleaned_for_analysis) %>%
    step_tokenfilter(cleaned_for_analysis, min_times = 3) %>%
    step_tfidf(cleaned_for_analysis)
  
  # Prepare and bake
  recipe_prep <- prep(recipe_obj)
  tfidf_features <- bake(recipe_prep, new_data = cleaned_data) %>%
    select(-word_count) %>%
    as.matrix()
  
  # PCA reduction
  pca_result <- prcomp(tfidf_features, center = TRUE, scale. = TRUE)
  n_components <- min(which(cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2) >= 0.90), 15)
  reduced_features <- pca_result$x[, 1:n_components]
  
  # UMAP
  umap_result <- umap(reduced_features, config = umap.defaults)
  umap_coords <- umap_result$layout
  
  # Test clustering
  textrecipes_silhouettes <- map_dbl(2:5, function(k) {
    kmeans_result <- kmeans(umap_coords, centers = k, nstart = 20)
    mean(silhouette(kmeans_result$cluster, dist(umap_coords))[, 3])
  })
  
  optimal_k <- which.max(textrecipes_silhouettes) + 1
  final_kmeans <- kmeans(umap_coords, centers = optimal_k, nstart = 20)
  
  # Get top terms via TF-IDF analysis
  cluster_data_tr <- cleaned_data %>%
    mutate(cluster = final_kmeans$cluster)
  
  tr_terms <- cluster_data_tr %>%
    unnest_tokens(word, cleaned_for_analysis) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) > 2) %>%
    count(cluster, word) %>%
    bind_tf_idf(word, cluster, n) %>%
    group_by(cluster) %>%
    slice_max(tf_idf, n = 5) %>%
    filter(cluster == 1) %>%
    pull(word) %>%
    paste(collapse = ", ")
  
  # Evaluate
  textrecipes_eval <- data.frame(
    method = "Textrecipes_TFIDF",
    n_clusters = optimal_k,
    silhouette_score = round(max(textrecipes_silhouettes), 3),
    pca_components = n_components,
    top_terms_sample = tr_terms
  )
  
  all_results[["Textrecipes"]] <- textrecipes_eval
  detailed_results[["Textrecipes"]] <- list(
    cluster_assignments = final_kmeans$cluster,
    umap_coords = umap_coords,
    pca_result = pca_result
  )
  
  cat(glue("✓ Textrecipes complete: K={optimal_k}, Silhouette={round(max(textrecipes_silhouettes), 3)}\n"))
}, silent = TRUE)

# =============================================================================
# METHOD 4: NMF (Non-negative Matrix Factorization)
# =============================================================================
cat("\n=== METHOD 4: NMF (Non-negative Matrix Factorization) ===\n")

try({
  # Create TF-IDF matrix for NMF
  nmf_tfidf <- cleaned_data %>%
    unnest_tokens(word, cleaned_for_analysis) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) > 2) %>%
    count(doc_id, word) %>%
    bind_tf_idf(word, doc_id, n) %>%
    select(doc_id, word, tf_idf) %>%
    pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%
    column_to_rownames("doc_id") %>%
    as.matrix()
  
  # Test different ranks
  nmf_results <- map(2:4, function(k) {
    cat(glue("NMF K={k}..."))
    nmf_model <- nmf(nmf_tfidf, rank = k, seed = 123, nrun = 1)
    # Calculate reconstruction error
    recon_error <- nmf_model@fit@measure[1]
    cat("done\n")
    list(k = k, model = nmf_model, error = recon_error)
  })
  
  # Select best NMF (lowest reconstruction error)
  best_nmf <- nmf_results[[which.min(map_dbl(nmf_results, "error"))]]
  
  # Get cluster assignments (max loading per document)
  nmf_H <- coef(best_nmf$model)  # Document-topic loadings
  nmf_clusters <- apply(t(nmf_H), 1, which.max)
  
  # Get top terms
  nmf_W <- basis(best_nmf$model)  # Term-topic loadings
  top_terms_idx <- apply(nmf_W, 2, function(x) order(x, decreasing = TRUE)[1:5])
  nmf_top_terms <- colnames(nmf_tfidf)[top_terms_idx[, 1]] %>%
    paste(collapse = ", ")
  
  # Evaluate
  nmf_eval <- data.frame(
    method = "NMF",
    n_clusters = best_nmf$k,
    silhouette_score = NA,
    reconstruction_error = round(best_nmf$error, 3),
    top_terms_sample = nmf_top_terms
  )
  
  all_results[["NMF"]] <- nmf_eval
  detailed_results[["NMF"]] <- list(
    cluster_assignments = nmf_clusters,
    H_matrix = nmf_H,
    W_matrix = nmf_W,
    model = best_nmf$model
  )
  
  cat(glue("✓ NMF complete: K={best_nmf$k}, Error={round(best_nmf$error, 3)}\n"))
}, silent = TRUE)

# =============================================================================
# METHOD 5: Simple TF-IDF + K-means
# =============================================================================
cat("\n=== METHOD 5: Simple TF-IDF + K-means ===\n")

try({
  # Create simple TF-IDF matrix
  simple_tfidf <- cleaned_data %>%
    unnest_tokens(word, cleaned_for_analysis) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) > 2) %>%
    count(doc_id, word) %>%
    bind_tf_idf(word, doc_id, n) %>%
    select(doc_id, word, tf_idf) %>%
    pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%
    column_to_rownames("doc_id") %>%
    as.matrix()
  
  # Test clustering
  simple_silhouettes <- map_dbl(2:5, function(k) {
    kmeans_result <- kmeans(simple_tfidf, centers = k, nstart = 20)
    mean(silhouette(kmeans_result$cluster, dist(simple_tfidf))[, 3])
  })
  
  optimal_k_simple <- which.max(simple_silhouettes) + 1
  final_kmeans_simple <- kmeans(simple_tfidf, centers = optimal_k_simple, nstart = 20)
  
  # Get top terms (highest TF-IDF in cluster centers)
  cluster_centers <- final_kmeans_simple$centers
  top_term_indices <- apply(cluster_centers, 1, function(x) order(x, decreasing = TRUE)[1:5])
  simple_top_terms <- colnames(simple_tfidf)[top_term_indices[, 1]] %>%
    paste(collapse = ", ")
  
  # Evaluate
  simple_eval <- data.frame(
    method = "Simple_TFIDF_Kmeans",
    n_clusters = optimal_k_simple,
    silhouette_score = round(max(simple_silhouettes), 3),
    feature_dimensions = ncol(simple_tfidf),
    top_terms_sample = simple_top_terms
  )
  
  all_results[["Simple"]] <- simple_eval
  detailed_results[["Simple"]] <- list(
    cluster_assignments = final_kmeans_simple$cluster,
    cluster_centers = cluster_centers,
    tfidf_matrix = simple_tfidf
  )
  
  cat(glue("✓ Simple TF-IDF complete: K={optimal_k_simple}, Silhouette={round(max(simple_silhouettes), 3)}\n"))
}, silent = TRUE)

# =============================================================================
# COMPARISON AND EVALUATION
# =============================================================================
cat("\n=== COMPREHENSIVE METHOD COMPARISON ===\n")

# Combine all results
comparison_df <- bind_rows(all_results)
print(comparison_df)

# Save detailed comparison
write_csv(comparison_df, file.path(results_dir, "method_comparison_summary.csv"))

# Detailed analysis for each method
for(method_name in names(detailed_results)) {
  method_data <- detailed_results[[method_name]]
  
  cat(glue("\n--- {method_name} Detailed Results ---\n"))
  
  if("cluster_assignments" %in% names(method_data)) {
    cluster_sizes <- table(method_data$cluster_assignments)
    cat(glue("Cluster sizes: {paste(cluster_sizes, collapse = ', ')}\n"))
    
    # Create cluster assignments file
    cluster_df <- cleaned_data %>%
      select(doc_id, session_id, participant_id, cleaned_text, cleaned_for_analysis) %>%
      mutate(cluster = method_data$cluster_assignments)
    
    write_csv(cluster_df, file.path(results_dir, glue("{method_name}_cluster_assignments.csv")))
  }
}

# Recommendations based on results
cat("\n=== RECOMMENDATIONS ===\n")

if(nrow(comparison_df) > 0) {
  # Find method with highest silhouette score
  best_silhouette <- comparison_df %>%
    filter(!is.na(silhouette_score)) %>%
    slice_max(silhouette_score, n = 1)
  
  if(nrow(best_silhouette) > 0) {
    cat(glue("🏆 Best silhouette score: {best_silhouette$method} (score: {best_silhouette$silhouette_score})\n"))
  }
  
  # Find most balanced clustering
  balanced_clusters <- comparison_df %>%
    filter(n_clusters >= 2, n_clusters <= 4) %>%
    arrange(desc(silhouette_score))
  
  if(nrow(balanced_clusters) > 0) {
    cat(glue("📊 Most interpretable: {balanced_clusters$method[1]} ({balanced_clusters$n_clusters[1]} clusters)\n"))
  }
  
  # Overall recommendation
  cat("\n🎯 For scientific validity and interpretability:\n")
  cat("1. Check silhouette scores for cluster quality\n")
  cat("2. Review top terms for thematic coherence\n") 
  cat("3. Examine cluster balance (avoid very small/large clusters)\n")
  cat("4. Consider domain expertise for final method selection\n")
} else {
  cat("⚠️ No methods completed successfully. Check error logs.\n")
}

cat("\n=== Analysis Complete ===\n")
cat(glue("All results saved to: {results_dir}\n"))
cat("\nReview the method_comparison_summary.csv and individual cluster assignments\n")
cat("to determine which approach provides the most scientifically valid and interpretable topics.\n")