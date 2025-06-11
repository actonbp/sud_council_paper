# SINGLETON CLUSTER DIAGNOSIS
# Understanding why "peopl" forms isolated cluster and exploring solutions

library(tidyverse)
library(here)
library(cluster)
library(widyr)

cat("=== SINGLETON CLUSTER DIAGNOSIS ===\n")
cat("Understanding why 'peopl' is isolated\n\n")

# Load the analysis results
results <- readRDS(here("results", "study2_original_approach_improved.rds"))

# Get the co-occurrence matrix and clustering results
cooccur_matrix <- results$cooccurrence_matrix
k3_clusters <- results$alternative_k_options$k3$clusters

cat("📊 CURRENT k=3 CLUSTERING:\n")
cat("=========================\n")
for (cluster_id in sort(unique(k3_clusters))) {
  words_in_cluster <- names(k3_clusters)[k3_clusters == cluster_id]
  cat("Cluster", cluster_id, ":", paste(words_in_cluster, collapse = ", "), "\n")
}

cat("\n🔍 DIAGNOSING 'PEOPL' SINGLETON:\n")
cat("================================\n")

# Analyze why "peopl" is isolated
peopl_row <- cooccur_matrix["peopl", ]
peopl_cooccurrences <- peopl_row[peopl_row > 0]
peopl_cooccurrences <- sort(peopl_cooccurrences, decreasing = TRUE)

cat("'peopl' co-occurs with", length(peopl_cooccurrences), "other words\n")
cat("Top 10 co-occurrences with 'peopl':\n")
for (i in 1:min(10, length(peopl_cooccurrences))) {
  word <- names(peopl_cooccurrences)[i]
  freq <- peopl_cooccurrences[i]
  cluster <- k3_clusters[word]
  cat(sprintf("  %2d. %-10s (freq: %2d, in cluster %d)\n", i, word, freq, cluster))
}

cat("\n📏 DISTANCE ANALYSIS:\n")
cat("====================\n")

# Calculate distances from "peopl" to all other words
dist_matrix <- dist(cooccur_matrix, method = "euclidean")
dist_df <- as.matrix(dist_matrix)
peopl_distances <- dist_df["peopl", ]
peopl_distances <- peopl_distances[names(peopl_distances) != "peopl"]
peopl_distances <- sort(peopl_distances)

cat("Closest words to 'peopl' (by Euclidean distance):\n")
for (i in 1:min(10, length(peopl_distances))) {
  word <- names(peopl_distances)[i]
  distance <- peopl_distances[i]
  cluster <- k3_clusters[word]
  cat(sprintf("  %2d. %-10s (distance: %6.2f, in cluster %d)\n", i, word, distance, cluster))
}

cat("\n🎯 POTENTIAL SOLUTIONS:\n")
cat("======================\n")

# Solution 1: Try different k values
cat("SOLUTION 1: Try k=2 (merge singleton with closest cluster)\n")
k2_clusters <- results$alternative_k_options$k2$clusters
cat("k=2 clustering:\n")
for (cluster_id in sort(unique(k2_clusters))) {
  words_in_cluster <- names(k2_clusters)[k2_clusters == cluster_id]
  cat("  Cluster", cluster_id, ":", paste(head(words_in_cluster, 8), collapse = ", "))
  if (length(words_in_cluster) > 8) cat(" ... +", length(words_in_cluster) - 8, "more")
  cat("\n")
}
cat("k=2 metrics: Silhouette =", round(results$alternative_k_options$k2$silhouette, 3), 
    ", Singletons =", results$alternative_k_options$k2$singleton_clusters, "\n\n")

# Solution 2: Try k=4 to see if it creates more balanced clusters
cat("SOLUTION 2: Try k=4 (allow more clusters to reduce singleton dominance)\n")
if ("k4" %in% names(results$alternative_k_options)) {
  k4_clusters <- results$alternative_k_options$k4$clusters
  cat("k=4 clustering:\n")
  for (cluster_id in sort(unique(k4_clusters))) {
    words_in_cluster <- names(k4_clusters)[k4_clusters == cluster_id]
    cat("  Cluster", cluster_id, ":", paste(head(words_in_cluster, 6), collapse = ", "))
    if (length(words_in_cluster) > 6) cat(" ... +", length(words_in_cluster) - 6, "more")
    cat("\n")
  }
  cat("k=4 metrics: Silhouette =", round(results$alternative_k_options$k4$silhouette, 3), 
      ", Singletons =", results$alternative_k_options$k4$singleton_clusters, "\n\n")
} else {
  cat("k=4 not available in saved results\n\n")
}

# Solution 3: Manual merging based on co-occurrence patterns
cat("SOLUTION 3: Data-driven merging based on co-occurrence patterns\n")
cat("'peopl' could be merged with cluster containing its top co-occurring words:\n")

# Find which cluster contains most of peopl's top co-occurring words
top_cooccur_words <- names(head(peopl_cooccurrences, 5))
cluster_votes <- table(k3_clusters[top_cooccur_words])
target_cluster <- as.numeric(names(cluster_votes)[which.max(cluster_votes)])

cat("Top 5 co-occurring words with 'peopl':", paste(top_cooccur_words, collapse = ", "), "\n")
cat("Most of these are in cluster", target_cluster, "\n")
cat("Merging 'peopl' with cluster", target_cluster, "would create:\n")

merged_clusters <- k3_clusters
merged_clusters["peopl"] <- target_cluster

for (cluster_id in sort(unique(merged_clusters))) {
  words_in_cluster <- names(merged_clusters)[merged_clusters == cluster_id]
  cat("  Merged Cluster", cluster_id, ":", paste(head(words_in_cluster, 8), collapse = ", "))
  if (length(words_in_cluster) > 8) cat(" ... +", length(words_in_cluster) - 8, "more")
  cat("\n")
}

cat("\n💡 RECOMMENDATION:\n")
cat("==================\n")

# Evaluate which solution makes most sense
if (results$alternative_k_options$k2$silhouette > 0.5 && 
    results$alternative_k_options$k2$singleton_clusters == 0) {
  cat("✅ RECOMMENDED: Use k=2\n")
  cat("   - No singleton clusters\n")
  cat("   - Good silhouette score (", round(results$alternative_k_options$k2$silhouette, 3), ")\n")
  cat("   - Simpler, more interpretable structure\n")
} else if (results$alternative_k_options$k3$silhouette > 0.2) {
  cat("⚠️  ALTERNATIVE: Use k=3 with manual merging\n")
  cat("   - Merge 'peopl' with cluster", target_cluster, "\n")
  cat("   - More thematic detail but requires methodological justification\n")
} else {
  cat("❌ PROBLEM: All clustering options have significant issues\n")
  cat("   - May need to reconsider approach entirely\n")
}

cat("\nNEXT STEPS:\n")
cat("Choose your preferred solution and I'll implement it properly!\n") 