# Examine Simple TF-IDF Clustering Results in Detail
# Purpose: Look at what the "winning" method actually produced
# Focus: Are the clusters actually interpretable and meaningful?

library(tidyverse)
library(tidytext)
library(here)
library(glue)
library(cluster)

# Set up paths
data_path <- here("data", "focus_group_substantive.csv")
results_dir <- here("results", "r", "study2_simple_tfidf_detailed")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# Read and preprocess data (same as method comparison)
focus_data <- read_csv(data_path, show_col_types = FALSE)

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

cleaned_data <- focus_data %>%
  mutate(
    cleaned_for_analysis = str_replace_all(cleaned_text, regex(sud_pattern, ignore_case = TRUE), ""),
    cleaned_for_analysis = str_squish(cleaned_for_analysis),
    word_count = str_count(cleaned_for_analysis, "\\w+")
  ) %>%
  filter(word_count >= 5) %>%
  mutate(doc_id = row_number())

cat(glue("Analyzing {nrow(cleaned_data)} utterances\n"))

# Recreate the "winning" Simple TF-IDF approach
cat("\n=== Recreating Simple TF-IDF + K-means ===\n")

# Create TF-IDF matrix
tfidf_matrix <- cleaned_data %>%
  unnest_tokens(word, cleaned_for_analysis) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2) %>%
  count(doc_id, word) %>%
  bind_tf_idf(word, doc_id, n) %>%
  select(doc_id, word, tf_idf) %>%
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%
  column_to_rownames("doc_id") %>%
  as.matrix()

cat(glue("TF-IDF matrix: {nrow(tfidf_matrix)} docs x {ncol(tfidf_matrix)} terms\n"))

# Test different K values and get silhouette scores
silhouette_scores <- map_dbl(2:6, function(k) {
  kmeans_result <- kmeans(tfidf_matrix, centers = k, nstart = 20, iter.max = 100)
  sil_result <- silhouette(kmeans_result$cluster, dist(tfidf_matrix))
  mean(sil_result[, 3])
})

k_values <- 2:6
optimal_k <- k_values[which.max(silhouette_scores)]

cat("\n=== Silhouette Scores by K ===\n")
for(i in seq_along(k_values)) {
  cat(glue("K = {k_values[i]}: Silhouette = {round(silhouette_scores[i], 3)}\n"))
}

cat(glue("\nOptimal K = {optimal_k} (silhouette = {round(max(silhouette_scores), 3)})\n"))

# Final clustering with optimal K
final_clustering <- kmeans(tfidf_matrix, centers = optimal_k, nstart = 25, iter.max = 100)

# Add cluster assignments to data
clustered_data <- cleaned_data %>%
  mutate(cluster = final_clustering$cluster)

# Analyze cluster sizes and characteristics
cluster_summary <- clustered_data %>%
  count(cluster, sort = TRUE) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 1),
    avg_words = map_dbl(cluster, ~mean(clustered_data$word_count[clustered_data$cluster == .x]))
  )

cat("\n=== Cluster Summary ===\n")
print(cluster_summary)

# Get top terms for each cluster using cluster centers
cluster_centers <- final_clustering$centers
vocab <- colnames(tfidf_matrix)

cat("\n=== Top Terms per Cluster (from K-means Centers) ===\n")
for(i in 1:optimal_k) {
  center_weights <- cluster_centers[i, ]
  top_indices <- order(center_weights, decreasing = TRUE)[1:15]
  top_terms <- vocab[top_indices]
  top_weights <- round(center_weights[top_indices], 4)
  
  cluster_size <- sum(final_clustering$cluster == i)
  cat(glue("\n**Cluster {i}** ({cluster_size} utterances):\n"))
  
  for(j in 1:10) {
    cat(glue("  {j}. {top_terms[j]} (weight: {top_weights[j]})\n"))
  }
}

# Alternative analysis: TF-IDF by cluster (not centers)
cat("\n=== Alternative: TF-IDF Analysis by Cluster ===\n")

cluster_tfidf <- clustered_data %>%
  select(cluster, cleaned_for_analysis) %>%
  unnest_tokens(word, cleaned_for_analysis) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2) %>%
  count(cluster, word) %>%
  bind_tf_idf(word, cluster, n) %>%
  arrange(cluster, desc(tf_idf))

for(i in 1:optimal_k) {
  top_cluster_terms <- cluster_tfidf %>%
    filter(cluster == i) %>%
    slice_head(n = 10)
  
  cluster_size <- sum(final_clustering$cluster == i)
  cat(glue("\n**Cluster {i}** ({cluster_size} utterances) - TF-IDF method:\n"))
  
  for(j in 1:nrow(top_cluster_terms)) {
    cat(glue("  {j}. {top_cluster_terms$word[j]} (tf-idf: {round(top_cluster_terms$tf_idf[j], 4)})\n"))
  }
}

# Sample utterances from each cluster
cat("\n=== Sample Utterances per Cluster ===\n")
for(i in 1:optimal_k) {
  cat(glue("\n--- Cluster {i} Sample Utterances ---\n"))
  
  cluster_utterances <- clustered_data %>% 
    filter(cluster == i) %>%
    slice_sample(n = min(3, n()))
  
  for(j in 1:nrow(cluster_utterances)) {
    cat(glue("{j}. {str_trunc(cluster_utterances$cleaned_text[j], 120)}\n"))
  }
}

# Check for term overlap between clusters
cat("\n=== Checking for Term Overlap Between Clusters ===\n")

top_terms_by_cluster <- cluster_tfidf %>%
  group_by(cluster) %>%
  slice_head(n = 10) %>%
  select(cluster, word)

term_overlap <- top_terms_by_cluster %>%
  count(word) %>%
  filter(n > 1) %>%
  arrange(desc(n))

if(nrow(term_overlap) > 0) {
  cat("⚠️ Terms appearing in multiple clusters:\n")
  print(term_overlap)
} else {
  cat("✓ No term overlap in top 10 terms per cluster\n")
}

# Evaluate cluster interpretability
cat("\n=== Cluster Interpretability Assessment ===\n")

# Check if clusters are substantively meaningful
interpretability_scores <- map_dbl(1:optimal_k, function(i) {
  cluster_terms <- cluster_tfidf %>%
    filter(cluster == i) %>%
    slice_head(n = 5) %>%
    pull(word)
  
  # Simple heuristic: do the top 5 terms seem thematically related?
  # This is subjective but we can flag potential issues
  
  # Check for very generic terms
  generic_terms <- c("people", "person", "help", "helping", "work", "working", 
                    "good", "bad", "thing", "things", "way", "ways", "time", "times")
  
  generic_count <- sum(cluster_terms %in% generic_terms)
  interpretability_score <- 1 - (generic_count / length(cluster_terms))
  
  cat(glue("Cluster {i}: {round(interpretability_score, 2)} (generic terms: {generic_count}/5)\n"))
  cat(glue("  Top terms: {paste(cluster_terms, collapse = ', ')}\n"))
  
  return(interpretability_score)
})

avg_interpretability <- mean(interpretability_scores)
cat(glue("\nOverall interpretability score: {round(avg_interpretability, 2)} (higher = better)\n"))

# Save detailed results
write_csv(clustered_data, file.path(results_dir, "simple_tfidf_clusters_detailed.csv"))
write_csv(cluster_tfidf, file.path(results_dir, "cluster_terms_tfidf.csv"))

# Final assessment
cat("\n=== FINAL ASSESSMENT: Simple TF-IDF Method ===\n")
cat(glue("✓ Silhouette Score: {round(max(silhouette_scores), 3)} (cluster separation quality)\n"))
cat(glue("✓ Optimal K: {optimal_k} clusters\n"))
cat(glue("✓ Interpretability: {round(avg_interpretability, 2)} (0-1 scale)\n"))

if(avg_interpretability < 0.6) {
  cat("⚠️ LOW INTERPRETABILITY: Many generic terms in top clusters\n")
  cat("   Consider trying a different method or different preprocessing\n")
} else if(avg_interpretability < 0.8) {
  cat("📊 MODERATE INTERPRETABILITY: Some generic terms present\n")
  cat("   Clusters may need manual interpretation/refinement\n")
} else {
  cat("🎯 HIGH INTERPRETABILITY: Clusters appear thematically coherent\n")
}

if(length(unique(cluster_summary$n)) == 1) {
  cat("✓ Balanced cluster sizes\n")
} else {
  size_ratio <- min(cluster_summary$n) / max(cluster_summary$n)
  if(size_ratio < 0.3) {
    cat("⚠️ UNBALANCED CLUSTERS: Some very small/large clusters\n")
  } else {
    cat("📊 Reasonably balanced cluster sizes\n")
  }
}

cat("\n" %||% glue("Results saved to: {results_dir}\n"))
cat("Review the detailed terms and sample utterances to assess whether\n")
cat("this method produces scientifically meaningful and interpretable topics.\n")