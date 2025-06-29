# COMPREHENSIVE CLUSTERING VALIDATION SCRIPT
# Tests rigor, robustness, and validity of hierarchical clustering approach

library(tidyverse)
library(tidytext)
library(here)
library(cluster)
library(SnowballC)
library(widyr)

cat("=== COMPREHENSIVE CLUSTERING VALIDATION ===\n")
cat("Testing methodological rigor and robustness\n\n")

# Load the preprocessed data
analysis <- readRDS(here("results", "proper_cooccurrence_analysis.rds"))
stems_with_session <- read_csv(here("data", "focus_group_tokens_preprocessed.csv"), show_col_types = FALSE)

# Get SUD-related tokens
sud_terms_raw <- c(
  "substance", "addiction", "addict", "addicted", "addictive", "dependence", "dependent", "dependency",
  "alcohol", "alcoholism", "alcoholic", "drug", "drugs", "cocaine", "heroin", "opioid", "opiate",
  "marijuana", "cannabis", "methamphetamine", "prescription",
  "recovery", "recovering", "rehabilitation", "rehab", "detox", "detoxification", "treatment",
  "therapy", "counseling", "intervention", "sobriety", "sober", "clean", "abstinence",
  "relapse", "methadone", "suboxone",
  "abuse", "abusing", "struggle", "struggling", "battle", "fighting", "overcome", "overcoming",
  "counselor", "therapist", "specialist", "program", "center", "services", "clinical"
)

sud_terms_stemmed <- tibble(term = sud_terms_raw) %>%
  mutate(term_stem = wordStem(str_to_lower(term), language = "en")) %>%
  pull(term_stem) %>% unique()

# Identify SUD tokens
sud_tokens <- stems_with_session %>%
  filter(str_detect(paste(sud_terms_stemmed, collapse = "|"), word_stem))

cat("📊 DATA VALIDATION:\n")
cat("Total SUD tokens:", nrow(sud_tokens), "\n")
cat("Unique SUD utterances:", n_distinct(sud_tokens$response_id), "\n")
cat("Unique word stems:", n_distinct(sud_tokens$word_stem), "\n\n")

# === VALIDATION TEST 1: DATA SUFFICIENCY ===
cat("🔍 VALIDATION TEST 1: DATA SUFFICIENCY\n")
cat("=====================================\n")

# Check data density
stem_freq <- sud_tokens %>%
  count(word_stem, sort = TRUE) %>%
  filter(n >= 3)  # Minimum frequency threshold

utterance_length <- sud_tokens %>%
  count(response_id) %>%
  pull(n)

cat("Words with ≥3 mentions:", nrow(stem_freq), "\n")
cat("Average tokens per SUD utterance:", round(mean(utterance_length), 1), "\n")
cat("Median tokens per SUD utterance:", median(utterance_length), "\n")
cat("Range of utterance lengths:", range(utterance_length), "\n")

# Data density assessment
n_words <- nrow(stem_freq)
n_utterances <- n_distinct(sud_tokens$response_id)
expected_cooccurrences <- n_words * (n_words - 1) / 2
cat("Possible word pairs:", expected_cooccurrences, "\n")
cat("Data density (utterances/word pairs):", round(n_utterances / expected_cooccurrences, 3), "\n")

if (n_utterances / expected_cooccurrences < 0.1) {
  cat("⚠️ WARNING: Low data density for stable co-occurrence patterns\n")
} else {
  cat("✅ Adequate data density for co-occurrence analysis\n")
}
cat("\n")

# === VALIDATION TEST 2: CO-OCCURRENCE STABILITY ===
cat("🔍 VALIDATION TEST 2: BOOTSTRAP STABILITY\n")
cat("=========================================\n")

# Function to calculate co-occurrence and cluster
cluster_utterances <- function(token_data, min_freq = 3) {
  # Calculate frequencies
  stem_counts <- token_data %>%
    count(word_stem, sort = TRUE) %>%
    filter(n >= min_freq)
  
  if (nrow(stem_counts) < 5) return(NULL)
  
  # Calculate co-occurrence
  word_pairs <- token_data %>%
    filter(word_stem %in% stem_counts$word_stem) %>%
    pairwise_count(word_stem, response_id, sort = TRUE) %>%
    filter(n >= 2)
  
  if (nrow(word_pairs) < 3) return(NULL)
  
  # Create matrix
  cooccur_matrix <- word_pairs %>%
    pivot_wider(names_from = item2, values_from = n, values_fill = 0) %>%
    column_to_rownames("item1") %>%
    as.matrix()
  
  # Make symmetric
  words_common <- intersect(rownames(cooccur_matrix), colnames(cooccur_matrix))
  if (length(words_common) < 3) return(NULL)
  
  cooccur_matrix <- cooccur_matrix[words_common, words_common]
  
  # Calculate distances and cluster
  cooccur_dist <- dist(cooccur_matrix, method = "euclidean")
  hc <- hclust(cooccur_dist, method = "ward.D2")
  
  # Silhouette analysis for optimal k
  sil_scores <- numeric(min(6, nrow(cooccur_matrix)-1))
  for(k in 2:length(sil_scores)+1) {
    if (k <= nrow(cooccur_matrix)) {
      clusters <- cutree(hc, k = k)
      if (length(unique(clusters)) == k) {
        sil <- silhouette(clusters, cooccur_dist)
        sil_scores[k-1] <- mean(sil[, 3])
      }
    }
  }
  
  optimal_k <- which.max(sil_scores) + 1
  max_sil <- max(sil_scores, na.rm = TRUE)
  
  clusters <- cutree(hc, k = optimal_k)
  
  return(list(
    optimal_k = optimal_k,
    silhouette_score = max_sil,
    clusters = clusters,
    words = names(clusters)
  ))
}

# Bootstrap stability test
set.seed(123)
n_bootstrap <- 100
bootstrap_results <- vector("list", n_bootstrap)

cat("Running", n_bootstrap, "bootstrap samples...\n")

for (i in 1:n_bootstrap) {
  # Sample utterances with replacement
  unique_utterances <- unique(sud_tokens$response_id)
  boot_utterances <- sample(unique_utterances, length(unique_utterances), replace = TRUE)
  
  boot_tokens <- sud_tokens %>%
    filter(response_id %in% boot_utterances)
  
  bootstrap_results[[i]] <- cluster_utterances(boot_tokens)
}

# Analyze bootstrap stability
valid_bootstraps <- bootstrap_results[!sapply(bootstrap_results, is.null)]
cat("Valid bootstrap samples:", length(valid_bootstraps), "/", n_bootstrap, "\n")

if (length(valid_bootstraps) > 10) {
  # Optimal k stability
  k_values <- sapply(valid_bootstraps, function(x) x$optimal_k)
  cat("Optimal k distribution:", table(k_values), "\n")
  cat("Most common k:", names(sort(table(k_values), decreasing = TRUE))[1], "\n")
  
  # Silhouette score distribution
  sil_scores <- sapply(valid_bootstraps, function(x) x$silhouette_score)
  cat("Silhouette scores - Mean:", round(mean(sil_scores, na.rm = TRUE), 3), 
      "SD:", round(sd(sil_scores, na.rm = TRUE), 3), "\n")
  cat("Silhouette scores - Range:", round(range(sil_scores, na.rm = TRUE), 3), "\n")
  
  # Word clustering stability
  all_words <- unique(unlist(lapply(valid_bootstraps, function(x) x$words)))
  word_stability <- tibble(word = all_words) %>%
    rowwise() %>%
    mutate(
      appearances = sum(sapply(valid_bootstraps, function(x) word %in% x$words)),
      avg_cluster = mean(sapply(valid_bootstraps, function(x) {
        if (word %in% x$words) x$clusters[word] else NA
      }), na.rm = TRUE)
    ) %>%
    ungroup() %>%
    arrange(desc(appearances))
  
  cat("\nWord stability (top 10):\n")
  print(word_stability %>% slice_head(n = 10))
  
} else {
  cat("⚠️ WARNING: Too few valid bootstrap samples for stability analysis\n")
}

cat("\n")

# === VALIDATION TEST 3: DISTANCE MEASURE SENSITIVITY ===
cat("🔍 VALIDATION TEST 3: DISTANCE MEASURE SENSITIVITY\n")
cat("==================================================\n")

# Test different distance measures on original data
original_result <- cluster_utterances(sud_tokens)

if (!is.null(original_result)) {
  # Get the co-occurrence matrix
  stem_counts <- sud_tokens %>%
    count(word_stem, sort = TRUE) %>%
    filter(n >= 3)
  
  word_pairs <- sud_tokens %>%
    filter(word_stem %in% stem_counts$word_stem) %>%
    pairwise_count(word_stem, response_id, sort = TRUE) %>%
    filter(n >= 2)
  
  cooccur_matrix <- word_pairs %>%
    pivot_wider(names_from = item2, values_from = n, values_fill = 0) %>%
    column_to_rownames("item1") %>%
    as.matrix()
  
  words_common <- intersect(rownames(cooccur_matrix), colnames(cooccur_matrix))
  cooccur_matrix <- cooccur_matrix[words_common, words_common]
  
  # Test different distance measures
  distance_methods <- c("euclidean", "manhattan", "canberra")
  distance_results <- list()
  
  for (method in distance_methods) {
    tryCatch({
      dist_mat <- dist(cooccur_matrix, method = method)
      hc <- hclust(dist_mat, method = "ward.D2")
      
      # Find optimal k
      sil_scores <- numeric(min(6, nrow(cooccur_matrix)-1))
      for(k in 2:length(sil_scores)+1) {
        if (k <= nrow(cooccur_matrix)) {
          clusters <- cutree(hc, k = k)
          if (length(unique(clusters)) == k) {
            sil <- silhouette(clusters, dist_mat)
            sil_scores[k-1] <- mean(sil[, 3])
          }
        }
      }
      
      optimal_k <- which.max(sil_scores) + 1
      max_sil <- max(sil_scores, na.rm = TRUE)
      
      distance_results[[method]] <- list(k = optimal_k, sil = max_sil)
      cat(method, "distance - Optimal k:", optimal_k, "Silhouette:", round(max_sil, 3), "\n")
    }, error = function(e) {
      cat(method, "distance - ERROR:", e$message, "\n")
    })
  }
}

cat("\n")

# === VALIDATION TEST 4: NULL MODEL COMPARISON ===
cat("🔍 VALIDATION TEST 4: NULL MODEL COMPARISON\n")
cat("===========================================\n")

# Create null models with same marginal frequencies
if (!is.null(original_result)) {
  null_silhouettes <- numeric(50)
  
  for (i in 1:50) {
    # Create random co-occurrence matrix with same marginal totals
    word_frequencies <- stem_counts$n
    names(word_frequencies) <- stem_counts$word_stem
    
    # Generate random co-occurrences preserving marginal frequencies
    null_pairs <- expand_grid(
      item1 = names(word_frequencies),
      item2 = names(word_frequencies)
    ) %>%
      filter(item1 != item2) %>%
      rowwise() %>%
      mutate(
        n = rpois(1, sqrt(word_frequencies[item1] * word_frequencies[item2]) / 10)
      ) %>%
      filter(n >= 2)
    
    if (nrow(null_pairs) > 0) {
      null_matrix <- null_pairs %>%
        pivot_wider(names_from = item2, values_from = n, values_fill = 0) %>%
        column_to_rownames("item1") %>%
        as.matrix()
      
      words_common <- intersect(rownames(null_matrix), colnames(null_matrix))
      if (length(words_common) >= 3) {
        null_matrix <- null_matrix[words_common, words_common]
        null_dist <- dist(null_matrix, method = "euclidean")
        
        # Calculate silhouette for k=3
        null_clusters <- cutree(hclust(null_dist, method = "ward.D2"), k = 3)
        if (length(unique(null_clusters)) == 3) {
          null_sil <- silhouette(null_clusters, null_dist)
          null_silhouettes[i] <- mean(null_sil[, 3])
        }
      }
    }
  }
  
  null_silhouettes <- null_silhouettes[null_silhouettes > 0]
  
  if (length(null_silhouettes) > 0) {
    cat("Null model silhouettes - Mean:", round(mean(null_silhouettes), 3), 
        "SD:", round(sd(null_silhouettes), 3), "\n")
    cat("Null model range:", round(range(null_silhouettes), 3), "\n")
    cat("Original silhouette:", round(original_result$silhouette_score, 3), "\n")
    
    # Statistical test
    p_value <- mean(null_silhouettes >= original_result$silhouette_score)
    cat("P-value (original > null):", round(p_value, 3), "\n")
    
    if (p_value < 0.05) {
      cat("✅ Original clustering significantly better than random\n")
    } else {
      cat("⚠️ WARNING: Original clustering not significantly better than random\n")
    }
  }
}

cat("\n")

# === VALIDATION TEST 5: ALTERNATIVE THRESHOLDS ===
cat("🔍 VALIDATION TEST 5: THRESHOLD SENSITIVITY\n")
cat("===========================================\n")

# Test different minimum frequency and co-occurrence thresholds
thresholds <- expand_grid(
  min_freq = c(2, 3, 4, 5),
  min_cooccur = c(1, 2, 3)
)

threshold_results <- thresholds %>%
  rowwise() %>%
  mutate(
    result = list(tryCatch({
      # Filter by thresholds
      stem_counts_thresh <- sud_tokens %>%
        count(word_stem, sort = TRUE) %>%
        filter(n >= min_freq)
      
      word_pairs_thresh <- sud_tokens %>%
        filter(word_stem %in% stem_counts_thresh$word_stem) %>%
        pairwise_count(word_stem, response_id, sort = TRUE) %>%
        filter(n >= min_cooccur)
      
      if (nrow(word_pairs_thresh) < 3 || nrow(stem_counts_thresh) < 5) {
        return(tibble(k = NA, sil = NA, n_words = 0, n_pairs = 0))
      }
      
      # Cluster
      cooccur_matrix_thresh <- word_pairs_thresh %>%
        pivot_wider(names_from = item2, values_from = n, values_fill = 0) %>%
        column_to_rownames("item1") %>%
        as.matrix()
      
      words_common <- intersect(rownames(cooccur_matrix_thresh), colnames(cooccur_matrix_thresh))
      cooccur_matrix_thresh <- cooccur_matrix_thresh[words_common, words_common]
      
      cooccur_dist_thresh <- dist(cooccur_matrix_thresh, method = "euclidean")
      hc_thresh <- hclust(cooccur_dist_thresh, method = "ward.D2")
      
      # Find optimal k
      sil_scores <- numeric(min(6, nrow(cooccur_matrix_thresh)-1))
      for(k in 2:length(sil_scores)+1) {
        if (k <= nrow(cooccur_matrix_thresh)) {
          clusters <- cutree(hc_thresh, k = k)
          if (length(unique(clusters)) == k) {
            sil <- silhouette(clusters, cooccur_dist_thresh)
            sil_scores[k-1] <- mean(sil[, 3])
          }
        }
      }
      
      optimal_k <- which.max(sil_scores) + 1
      max_sil <- max(sil_scores, na.rm = TRUE)
      
      tibble(
        k = optimal_k,
        sil = max_sil,
        n_words = nrow(cooccur_matrix_thresh),
        n_pairs = nrow(word_pairs_thresh)
      )
    }, error = function(e) {
      tibble(k = NA, sil = NA, n_words = 0, n_pairs = 0)
    }))
  ) %>%
  unnest(result)

cat("Threshold sensitivity analysis:\n")
print(threshold_results %>% select(-result) %>% arrange(desc(sil)))

# === FINAL ASSESSMENT ===
cat("\n🎯 VALIDATION SUMMARY\n")
cat("====================\n")

# Save all validation results
validation_summary <- list(
  bootstrap_results = valid_bootstraps,
  distance_sensitivity = if(exists("distance_results")) distance_results else NULL,
  null_comparison = if(exists("null_silhouettes")) null_silhouettes else NULL,
  threshold_sensitivity = threshold_results,
  original_result = original_result,
  data_quality = list(
    n_words = nrow(stem_freq),
    n_utterances = n_distinct(sud_tokens$response_id),
    data_density = n_distinct(sud_tokens$response_id) / (nrow(stem_freq) * (nrow(stem_freq) - 1) / 2)
  )
)

saveRDS(validation_summary, here("results", "clustering_validation_comprehensive.rds"))

cat("✅ Comprehensive validation complete!\n")
cat("Results saved to: results/clustering_validation_comprehensive.rds\n")
cat("\nRecommendations will be based on validation findings...\n") 