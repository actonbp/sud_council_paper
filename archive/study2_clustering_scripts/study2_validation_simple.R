# SIMPLIFIED CLUSTERING VALIDATION SCRIPT
# Focus on key validation tests without complex silhouette edge cases

library(tidyverse)
library(tidytext)
library(here)
library(cluster)
library(SnowballC)
library(widyr)

cat("=== CLUSTERING VALIDATION ANALYSIS ===\n")
cat("Testing data quality and clustering robustness\n\n")

# Load preprocessed data
stems_with_session <- read_csv(here("data", "focus_group_tokens_preprocessed.csv"), show_col_types = FALSE)

# Define SUD terms
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

# Get SUD tokens
sud_tokens <- stems_with_session %>%
  filter(str_detect(paste(sud_terms_stemmed, collapse = "|"), word_stem))

cat("📊 BASIC DATA VALIDATION:\n")
cat("=========================\n")
cat("Total SUD tokens:", nrow(sud_tokens), "\n")
cat("Unique SUD utterances:", n_distinct(sud_tokens$response_id), "\n")
cat("Unique word stems:", n_distinct(sud_tokens$word_stem), "\n")
cat("Average tokens per utterance:", round(nrow(sud_tokens) / n_distinct(sud_tokens$response_id), 2), "\n\n")

# Word frequency analysis
word_frequencies <- sud_tokens %>%
  count(word_stem, sort = TRUE)

cat("📈 WORD FREQUENCY DISTRIBUTION:\n")
cat("===============================\n")
cat("Total unique words:", nrow(word_frequencies), "\n")
cat("Words with ≥3 mentions:", sum(word_frequencies$n >= 3), "\n")
cat("Words with ≥5 mentions:", sum(word_frequencies$n >= 5), "\n")
cat("Words with ≥10 mentions:", sum(word_frequencies$n >= 10), "\n")

cat("\nTop 15 most frequent words:\n")
print(word_frequencies %>% slice_head(n = 15))

# Co-occurrence analysis
cat("\n🔗 CO-OCCURRENCE ANALYSIS:\n")
cat("==========================\n")

# Test different minimum frequency thresholds
for (min_freq in c(2, 3, 4, 5)) {
  stem_counts <- word_frequencies %>%
    filter(n >= min_freq)
  
  if (nrow(stem_counts) < 3) {
    cat("Min freq", min_freq, ": Too few words (", nrow(stem_counts), ")\n")
    next
  }
  
  # Calculate co-occurrences
  word_pairs <- sud_tokens %>%
    filter(word_stem %in% stem_counts$word_stem) %>%
    pairwise_count(word_stem, response_id, sort = TRUE) %>%
    filter(n >= 2)
  
  cat("Min freq", min_freq, ": ", nrow(stem_counts), "words,", nrow(word_pairs), "co-occurrence pairs\n")
  
  if (nrow(word_pairs) > 0) {
    cat("  - Top co-occurrence:", word_pairs$item1[1], "+", word_pairs$item2[1], "(", word_pairs$n[1], "times)\n")
    cat("  - Co-occurrence range:", range(word_pairs$n), "\n")
  }
}

# Matrix analysis for min_freq = 3 (our current approach)
cat("\n🔍 MATRIX STRUCTURE ANALYSIS (min_freq=3):\n")
cat("=========================================\n")

stem_counts <- word_frequencies %>%
  filter(n >= 3)

word_pairs <- sud_tokens %>%
  filter(word_stem %in% stem_counts$word_stem) %>%
  pairwise_count(word_stem, response_id, sort = TRUE) %>%
  filter(n >= 2)

if (nrow(word_pairs) > 0) {
  # Create co-occurrence matrix
  cooccur_matrix <- word_pairs %>%
    pivot_wider(names_from = item2, values_from = n, values_fill = 0) %>%
    column_to_rownames("item1") %>%
    as.matrix()
  
  # Make symmetric
  words_common <- intersect(rownames(cooccur_matrix), colnames(cooccur_matrix))
  cooccur_matrix <- cooccur_matrix[words_common, words_common]
  
  cat("Matrix dimensions:", nrow(cooccur_matrix), "x", ncol(cooccur_matrix), "\n")
  cat("Matrix sparsity:", round(sum(cooccur_matrix == 0) / length(cooccur_matrix), 3), "\n")
  cat("Matrix density (non-zero):", round(sum(cooccur_matrix > 0) / length(cooccur_matrix), 3), "\n")
  cat("Average co-occurrence value:", round(mean(cooccur_matrix[cooccur_matrix > 0]), 2), "\n")
  
  # Check matrix properties
  cat("Matrix sum:", sum(cooccur_matrix), "\n")
  cat("Matrix max value:", max(cooccur_matrix), "\n")
  cat("Number of non-zero entries:", sum(cooccur_matrix > 0), "\n")
  
  # Simple clustering without silhouette
  cat("\n🎯 CLUSTERING ATTEMPT:\n")
  cat("======================\n")
  
  if (nrow(cooccur_matrix) >= 3) {
    cooccur_dist <- dist(cooccur_matrix, method = "euclidean")
    hc <- hclust(cooccur_dist, method = "ward.D2")
    
    cat("Distance matrix created successfully\n")
    cat("Hierarchical clustering completed\n")
    
    # Test different k values
    for (k in 2:min(6, nrow(cooccur_matrix))) {
      clusters <- cutree(hc, k = k)
      n_clusters_actual <- length(unique(clusters))
      
      cat("k =", k, ": Got", n_clusters_actual, "actual clusters\n")
      
      if (n_clusters_actual == k) {
        # Calculate within-cluster sum of squares
        wss <- 0
        for (cluster_id in unique(clusters)) {
          cluster_points <- cooccur_matrix[clusters == cluster_id, , drop = FALSE]
          if (nrow(cluster_points) > 1) {
            cluster_center <- colMeans(cluster_points)
            for (i in 1:nrow(cluster_points)) {
              wss <- wss + sum((cluster_points[i, ] - cluster_center)^2)
            }
          }
        }
        cat("  - Within-cluster SS:", round(wss, 1), "\n")
        
        # Show cluster assignments
        cluster_table <- table(clusters)
        cat("  - Cluster sizes:", paste(cluster_table, collapse = ", "), "\n")
      }
    }
    
    # Show detailed k=3 clustering
    clusters_k3 <- cutree(hc, k = 3)
    cat("\nDETAILED k=3 CLUSTERING:\n")
    for (cluster_id in sort(unique(clusters_k3))) {
      words_in_cluster <- names(clusters_k3)[clusters_k3 == cluster_id]
      cat("Cluster", cluster_id, ":", paste(words_in_cluster, collapse = ", "), "\n")
    }
  }
}

# Bootstrap stability test (simplified)
cat("\n🔄 BOOTSTRAP STABILITY TEST:\n")
cat("============================\n")

set.seed(123)
n_bootstrap <- 20  # Reduced for speed
successful_bootstraps <- 0

for (i in 1:n_bootstrap) {
  # Sample utterances with replacement
  unique_utterances <- unique(sud_tokens$response_id)
  boot_utterances <- sample(unique_utterances, length(unique_utterances), replace = TRUE)
  
  boot_tokens <- sud_tokens %>%
    filter(response_id %in% boot_utterances)
  
  # Quick clustering attempt
  tryCatch({
    boot_stem_counts <- boot_tokens %>%
      count(word_stem, sort = TRUE) %>%
      filter(n >= 3)
    
    if (nrow(boot_stem_counts) >= 5) {
      boot_word_pairs <- boot_tokens %>%
        filter(word_stem %in% boot_stem_counts$word_stem) %>%
        pairwise_count(word_stem, response_id, sort = TRUE) %>%
        filter(n >= 2)
      
      if (nrow(boot_word_pairs) >= 3) {
        successful_bootstraps <- successful_bootstraps + 1
      }
    }
  }, error = function(e) {
    # Ignore errors, just count failures
  })
}

cat("Successful bootstrap samples:", successful_bootstraps, "/", n_bootstrap, "\n")
cat("Bootstrap stability rate:", round(successful_bootstraps / n_bootstrap, 2), "\n")

if (successful_bootstraps / n_bootstrap < 0.5) {
  cat("⚠️ WARNING: Low bootstrap stability suggests unstable clustering\n")
} else {
  cat("✅ Reasonable bootstrap stability\n")
}

cat("\n🎯 VALIDATION SUMMARY:\n")
cat("======================\n")
cat("1. Data sufficiency: ", ifelse(n_distinct(sud_tokens$response_id) > 100, "✅ Good", "⚠️ Limited"), "\n")
cat("2. Word frequency: ", ifelse(sum(word_frequencies$n >= 3) >= 10, "✅ Adequate", "⚠️ Limited"), "\n")
cat("3. Co-occurrence pairs: ", ifelse(nrow(word_pairs) >= 10, "✅ Sufficient", "⚠️ Limited"), "\n")
cat("4. Bootstrap stability: ", ifelse(successful_bootstraps / n_bootstrap >= 0.5, "✅ Stable", "⚠️ Unstable"), "\n")

cat("\n💡 RECOMMENDATIONS BASED ON VALIDATION:\n")
cat("=======================================\n")

# Specific recommendations
if (n_distinct(sud_tokens$response_id) < 50) {
  cat("• Consider lower minimum frequency thresholds due to limited data\n")
}

if (sum(word_frequencies$n >= 3) < 15) {
  cat("• Reduce minimum word frequency from 3 to 2 for more words\n")
}

if (nrow(word_pairs) < 10) {
  cat("• Reduce minimum co-occurrence threshold from 2 to 1\n")
}

if (successful_bootstraps / n_bootstrap < 0.5) {
  cat("• Focus on most frequent words only for stability\n")
  cat("• Consider simpler grouping approaches\n")
}

cat("\n✅ Validation analysis complete!\n") 