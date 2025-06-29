# Text Embeddings + Clustering with tidymodels/textrecipes
# Purpose: Use textrecipes word embeddings for semantic clustering
# Input: data/focus_group_substantive.csv
# Output: Semantic clusters using tidymodels workflow

# Load required libraries
library(tidyverse)
library(tidymodels)
library(textrecipes)
library(tidytext)
library(here)
library(glue)
library(cluster)
library(umap)

# Set up paths
data_path <- here("data", "focus_group_substantive.csv")
results_dir <- here("results", "r", "study2_textrecipes_clustering")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# Read focus group data
cat("\n=== Reading Focus Group Data ===\n")
focus_data <- read_csv(data_path, show_col_types = FALSE)
cat(glue("Total substantive utterances: {nrow(focus_data)}\n"))

# SUD-specific terms to remove from text
sud_terms <- c(
  "substance", "substances", "counselor", "counselors", "counseling",
  "alcohol", "alcoholic", "alcoholism", "drug", "drugs", "addiction", "addictions",
  "sud", "abuse", "disorder", "disorders", "treatment", "treatments", "recovery",
  "addict", "addicts", "addicted", "mental", "health", "therapist", "therapists", 
  "therapy", "therapies", "psychology", "psychologist", "psychologists", "psych",
  "psychiatric", "psychiatrist", "social", "worker", "workers", "nurse", "nurses",
  "nursing", "clinical", "clinician", "practitioner", "practitioners"
)

# Create regex pattern for SUD terms
sud_pattern <- paste0("\\b(", paste(sud_terms, collapse = "|"), ")\\b")

# Clean text by removing SUD-specific terms
cat("\n=== Cleaning Text for Embedding Analysis ===\n")
cleaned_data <- focus_data %>%
  mutate(
    # Remove SUD terms but keep sentence structure
    cleaned_for_embedding = str_replace_all(cleaned_text, regex(sud_pattern, ignore_case = TRUE), ""),
    # Clean up extra spaces and punctuation
    cleaned_for_embedding = str_squish(cleaned_for_embedding),
    # Keep only utterances with meaningful content
    word_count = str_count(cleaned_for_embedding, "\\w+"),
    # Create a dummy outcome variable for recipe (clustering doesn't need this)
    outcome = 1
  ) %>%
  filter(word_count >= 5) %>%  # Keep utterances with at least 5 words
  mutate(doc_id = row_number())

cat(glue("Utterances after cleaning: {nrow(cleaned_data)}\n"))
cat(glue("Average words per utterance: {round(mean(cleaned_data$word_count), 1)}\n"))

# Check if we have pre-trained embeddings available
# If not available, we'll use a simple approach
cat("\n=== Setting Up Word Embeddings ===\n")

# For this example, we'll create a simple embedding approach
# In practice, you would load pre-trained embeddings like GloVe or fastText
# Let's check if textrecipes has built-in embeddings or create a workaround

# Create a recipe for text preprocessing and embedding
text_recipe <- recipe(outcome ~ cleaned_for_embedding, data = cleaned_data) %>%
  # Tokenize the text
  step_tokenize(cleaned_for_embedding) %>%
  # Remove stopwords
  step_stopwords(cleaned_for_embedding) %>%
  # Remove rare tokens (appear in < 3 documents)
  step_tokenfilter(cleaned_for_embedding, min_times = 3) %>%
  # Use TF-IDF features
  step_tfidf(cleaned_for_embedding)

cat("✓ Text preprocessing recipe created\n")

# Prepare the recipe
cat("\n=== Preparing Text Recipe ===\n")
text_prep <- prep(text_recipe)
cat("✓ Recipe prepared\n")

# Apply the recipe to create features
embedding_features <- bake(text_prep, new_data = cleaned_data)

# Remove the outcome column and get feature matrix
feature_matrix <- embedding_features %>%
  select(-outcome) %>%
  as.matrix()

cat(glue("✓ Feature matrix created: {nrow(feature_matrix)} x {ncol(feature_matrix)}\n"))

# Apply dimensionality reduction if needed (TF-IDF can be high-dimensional)
if(ncol(feature_matrix) > 50) {
  cat("\n=== Applying PCA for Dimensionality Reduction ===\n")
  pca_result <- prcomp(feature_matrix, center = TRUE, scale. = TRUE)
  # Keep components that explain 90% of variance
  cumvar <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
  n_components <- which(cumvar >= 0.90)[1]
  n_components <- min(n_components, 20)  # Cap at 20 components
  
  reduced_features <- pca_result$x[, 1:n_components]
  cat(glue("✓ PCA complete: Reduced to {n_components} components (90% variance)\n"))
} else {
  reduced_features <- feature_matrix
  cat("✓ Using full feature matrix (low dimensional)\n")
}

# Apply UMAP for further reduction and visualization
cat("\n=== Applying UMAP for Clustering Preparation ===\n")
umap_config <- umap.defaults
umap_config$n_neighbors <- min(15, nrow(reduced_features) - 1)
umap_config$min_dist <- 0.1
umap_config$n_components <- 3

umap_result <- umap(reduced_features, config = umap_config)
umap_coords <- as.data.frame(umap_result$layout)
colnames(umap_coords) <- c("UMAP1", "UMAP2", "UMAP3")

cat("✓ UMAP complete\n")

# Test different numbers of clusters
cat("\n=== Testing Cluster Numbers ===\n")
k_values <- 2:6
silhouette_scores <- numeric(length(k_values))
within_ss <- numeric(length(k_values))

for(i in seq_along(k_values)) {
  k <- k_values[i]
  kmeans_result <- kmeans(umap_coords, centers = k, nstart = 25, iter.max = 100)
  
  if(k > 1) {
    sil_score <- cluster::silhouette(kmeans_result$cluster, dist(umap_coords))
    silhouette_scores[i] <- mean(sil_score[, 3])
  } else {
    silhouette_scores[i] <- 0
  }
  
  within_ss[i] <- kmeans_result$tot.withinss
  cat(glue("K = {k}: Silhouette = {round(silhouette_scores[i], 3)}\n"))
}

# Select optimal K
optimal_k <- k_values[which.max(silhouette_scores)]
cat(glue("\n✓ Optimal K = {optimal_k} (silhouette: {round(max(silhouette_scores), 3)})\n"))

# Final clustering
final_kmeans <- kmeans(umap_coords, centers = optimal_k, nstart = 25, iter.max = 100)

# Add cluster assignments to original data
clustered_data <- cleaned_data %>%
  bind_cols(umap_coords) %>%
  mutate(cluster = final_kmeans$cluster)

# Save results
output_data <- clustered_data %>%
  select(doc_id, session_id, participant_id, cleaned_text, 
         cleaned_for_embedding, word_count, cluster, UMAP1, UMAP2, UMAP3)

write_csv(output_data, file.path(results_dir, "textrecipes_clusters.csv"))

# Analyze clusters
cat("\n=== Analyzing Clusters ===\n")
cluster_summary <- clustered_data %>%
  count(cluster) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 1),
    avg_words = map_dbl(cluster, ~mean(clustered_data$word_count[clustered_data$cluster == .x]))
  ) %>%
  arrange(desc(n))

print(cluster_summary)

# Extract characteristic terms using basic tokenization for interpretation
# Use regular tidytext tokenization for TF-IDF analysis
cluster_terms <- clustered_data %>%
  select(cluster, cleaned_for_embedding) %>%
  unnest_tokens(word, cleaned_for_embedding) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2, !str_detect(word, "^\\d+$")) %>%
  count(cluster, word) %>%
  bind_tf_idf(word, cluster, n) %>%
  arrange(cluster, desc(tf_idf))

# Top terms per cluster
top_terms_per_cluster <- cluster_terms %>%
  group_by(cluster) %>%
  slice_max(tf_idf, n = 12) %>%
  summarise(
    top_terms = paste(word, collapse = ", "),
    .groups = "drop"
  )

write_csv(cluster_terms, file.path(results_dir, "cluster_terms_analysis.csv"))

# Display cluster themes
cat("\n=== Cluster Themes (TF-IDF Analysis) ===\n")
for(i in 1:optimal_k) {
  cluster_info <- cluster_summary %>% filter(cluster == i)
  top_terms <- top_terms_per_cluster %>% filter(cluster == i) %>% pull(top_terms)
  
  cat(glue("\n**Cluster {i}**: {cluster_info$n} utterances ({cluster_info$percentage}%)\n"))
  cat(glue("Key terms: {top_terms}\n"))
}

# Sample utterances
cat("\n=== Sample Utterances per Cluster ===\n")
for(i in 1:optimal_k) {
  cat(glue("\n--- Cluster {i} Examples ---\n"))
  cluster_data <- clustered_data %>% filter(cluster == i)
  samples <- cluster_data %>%
    slice_sample(n = min(3, nrow(cluster_data))) %>%
    pull(cleaned_text)
  
  for(j in seq_along(samples)) {
    cat(glue("{j}. {str_trunc(samples[j], 100)}\n"))
  }
}

# Save metrics
metrics_df <- data.frame(
  k = k_values,
  silhouette_score = silhouette_scores,
  within_ss = within_ss
)
write_csv(metrics_df, file.path(results_dir, "clustering_metrics.csv"))

# Summary
summary_data <- data.frame(
  metric = c("Total Utterances", "Optimal K", "Max Silhouette", "Feature Dimensions", "PCA Components"),
  value = c(nrow(clustered_data), optimal_k, round(max(silhouette_scores), 3), 
            ncol(feature_matrix), ncol(reduced_features))
)
write_csv(summary_data, file.path(results_dir, "analysis_summary.csv"))

cat("\n=== Textrecipes Clustering Complete ===\n")
cat(glue("Results saved to: {results_dir}\n"))
cat("\nKey outputs:\n")
cat("- textrecipes_clusters.csv: Utterances with cluster assignments\n")
cat("- cluster_terms_analysis.csv: TF-IDF terms characterizing each cluster\n")
cat("- clustering_metrics.csv: Validation metrics for different K values\n")

cat(glue("\n🎯 Analysis identified {optimal_k} thematic clusters using tidymodels workflow.\n"))
cat("Clusters are based on TF-IDF features processed through textrecipes.\n")