# Sentence Embeddings + Clustering for Study 2 Focus Groups (Python Backend)
# Purpose: Use sentence embeddings via Python sentence-transformers to cluster utterances
# Input: data/focus_group_substantive.csv  
# Output: Semantic clusters with topic interpretation

# Load required libraries
library(tidyverse)
library(tidytext)
library(SnowballC)
library(here)
library(glue)
library(cluster)
library(umap)
library(reticulate)

# Set up paths
data_path <- here("data", "focus_group_substantive.csv")
results_dir <- here("results", "r", "study2_embedding_clustering")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# Read focus group data
cat("\n=== Reading Focus Group Data ===\n")
focus_data <- read_csv(data_path, show_col_types = FALSE)
cat(glue("Total substantive utterances: {nrow(focus_data)}\n"))

# SUD-specific terms to remove from text (same as BTM script)
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

# Clean text by removing SUD-specific terms but preserving sentence structure
cat("\n=== Cleaning Text for Semantic Analysis ===\n")
cleaned_data <- focus_data %>%
  mutate(
    # Remove SUD terms but keep sentence structure
    cleaned_for_embedding = str_replace_all(cleaned_text, regex(sud_pattern, ignore_case = TRUE), ""),
    # Clean up extra spaces
    cleaned_for_embedding = str_squish(cleaned_for_embedding),
    # Keep only utterances with meaningful content after cleaning
    word_count = str_count(cleaned_for_embedding, "\\w+")
  ) %>%
  filter(word_count >= 5) %>%  # Keep utterances with at least 5 words
  mutate(doc_id = row_number())

cat(glue("Utterances after cleaning: {nrow(cleaned_data)}\n"))
cat(glue("Average words per utterance: {round(mean(cleaned_data$word_count), 1)}\n"))

# Check if Python and required packages are available
cat("\n=== Checking Python Environment ===\n")
tryCatch({
  # Try to import sentence-transformers
  st <- import("sentence_transformers")
  numpy <- import("numpy")
  cat("✓ Python sentence-transformers available\n")
}, error = function(e) {
  cat("✗ Python sentence-transformers not available\n")
  cat("Installing via pip...\n")
  py_install("sentence-transformers", pip = TRUE)
  st <<- import("sentence_transformers")
  numpy <<- import("numpy")
})

# Initialize sentence transformer model
cat("\n=== Loading Sentence Transformer Model ===\n")
model <- st$SentenceTransformer('all-MiniLM-L6-v2')
cat("✓ Model loaded: all-MiniLM-L6-v2 (384 dimensions)\n")

# Generate embeddings
cat("\n=== Generating Sentence Embeddings ===\n")
cat(glue("Processing {nrow(cleaned_data)} utterances...\n"))

# Convert text to embeddings
texts <- cleaned_data$cleaned_for_embedding
embeddings <- model$encode(texts)
embedding_matrix <- as.matrix(embeddings)

cat(glue("✓ Embeddings generated: {nrow(embedding_matrix)} x {ncol(embedding_matrix)}\n"))

# Apply UMAP for dimensionality reduction
cat("\n=== Applying UMAP Dimensionality Reduction ===\n")
umap_config <- umap.defaults
umap_config$n_neighbors <- min(15, nrow(embedding_matrix) - 1)
umap_config$min_dist <- 0.1
umap_config$n_components <- 3  # 3D for better clustering

umap_result <- umap(embedding_matrix, config = umap_config)
umap_coords <- as.data.frame(umap_result$layout)
colnames(umap_coords) <- c("UMAP1", "UMAP2", "UMAP3")

cat("✓ UMAP dimensionality reduction complete\n")

# Test different numbers of clusters
cat("\n=== Testing Different Cluster Numbers ===\n")
k_values <- 2:6
silhouette_scores <- numeric(length(k_values))
within_ss <- numeric(length(k_values))

for(i in seq_along(k_values)) {
  k <- k_values[i]
  kmeans_result <- kmeans(umap_coords, centers = k, nstart = 25, iter.max = 100)
  
  # Calculate silhouette score
  if(k > 1) {
    sil_score <- cluster::silhouette(kmeans_result$cluster, dist(umap_coords))
    silhouette_scores[i] <- mean(sil_score[, 3])
  } else {
    silhouette_scores[i] <- 0
  }
  
  within_ss[i] <- kmeans_result$tot.withinss
  
  cat(glue("K = {k}: Silhouette = {round(silhouette_scores[i], 3)}, Within SS = {round(within_ss[i], 1)}\n"))
}

# Select optimal K based on silhouette score
optimal_k <- k_values[which.max(silhouette_scores)]
cat(glue("\n✓ Optimal K = {optimal_k} (silhouette score: {round(max(silhouette_scores), 3)})\n"))

# Final clustering with optimal K
final_kmeans <- kmeans(umap_coords, centers = optimal_k, nstart = 25, iter.max = 100)

# Add cluster assignments to data
clustered_data <- cleaned_data %>%
  bind_cols(umap_coords) %>%
  mutate(cluster = final_kmeans$cluster)

# Save cluster assignments with metadata
output_data <- clustered_data %>%
  select(doc_id, session_id, participant_id, response_id, cleaned_text, 
         cleaned_for_embedding, word_count, cluster, UMAP1, UMAP2, UMAP3)

write_csv(output_data, file.path(results_dir, "sentence_embedding_clusters.csv"))

# Analyze cluster characteristics
cat("\n=== Analyzing Cluster Characteristics ===\n")

cluster_summary <- clustered_data %>%
  count(cluster) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 1),
    avg_words = map_dbl(cluster, ~mean(clustered_data$word_count[clustered_data$cluster == .x]))
  ) %>%
  arrange(desc(n))

print(cluster_summary)

# Extract top characteristic words for each cluster using TF-IDF
cat("\n=== Extracting Cluster Themes via TF-IDF ===\n")

# Tokenize the cleaned text for theme extraction
cluster_words <- clustered_data %>%
  select(doc_id, cluster, cleaned_for_embedding) %>%
  unnest_tokens(word, cleaned_for_embedding) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2, !str_detect(word, "^\\d+$")) %>%
  mutate(word = SnowballC::wordStem(word)) %>%
  filter(nchar(word) >= 3)  # Remove overly stemmed words

# Calculate TF-IDF by cluster
cluster_tfidf <- cluster_words %>%
  count(cluster, word) %>%
  bind_tf_idf(word, cluster, n) %>%
  arrange(cluster, desc(tf_idf))

# Top words per cluster
top_words_per_cluster <- cluster_tfidf %>%
  group_by(cluster) %>%
  slice_max(tf_idf, n = 12) %>%
  summarise(
    top_words = paste(word, collapse = ", "),
    .groups = "drop"
  )

# Save detailed results
write_csv(cluster_tfidf, file.path(results_dir, "cluster_tfidf_analysis.csv"))
write_csv(top_words_per_cluster, file.path(results_dir, "cluster_top_words.csv"))

# Print cluster themes
cat("\n=== Semantic Cluster Themes ===\n")
for(i in 1:optimal_k) {
  cluster_info <- cluster_summary %>% filter(cluster == i)
  top_words <- top_words_per_cluster %>% filter(cluster == i) %>% pull(top_words)
  
  cat(glue("\n**Cluster {i}**: {cluster_info$n} utterances ({cluster_info$percentage}%)\n"))
  cat(glue("Avg words: {round(cluster_info$avg_words, 1)}\n"))
  cat(glue("Key terms: {top_words}\n"))
}

# Sample representative utterances from each cluster
cat("\n=== Representative Utterances per Cluster ===\n")
for(i in 1:optimal_k) {
  cat(glue("\n--- Cluster {i} Examples ---\n"))
  
  # Get utterances closest to cluster center in embedding space
  cluster_utterances <- clustered_data %>% filter(cluster == i)
  cluster_embeddings <- embedding_matrix[clustered_data$cluster == i, ]
  cluster_center <- colMeans(cluster_embeddings)
  
  # Calculate distances to center
  distances <- apply(cluster_embeddings, 1, function(x) sqrt(sum((x - cluster_center)^2)))
  closest_indices <- order(distances)[1:min(3, length(distances))]
  
  representative_utterances <- cluster_utterances[closest_indices, ]$cleaned_text
  
  for(j in seq_along(representative_utterances)) {
    cat(glue("{j}. {str_trunc(representative_utterances[j], 120)}\n"))
  }
}

# Create analysis summary
summary_metrics <- data.frame(
  metric = c("Total Utterances Analyzed", "Optimal K", "Max Silhouette Score", 
             "Embedding Dimensions", "UMAP Components", "Avg Words per Utterance"),
  value = c(nrow(clustered_data), optimal_k, round(max(silhouette_scores), 3),
            ncol(embedding_matrix), 3, round(mean(cleaned_data$word_count), 1))
)

write_csv(summary_metrics, file.path(results_dir, "analysis_summary.csv"))

# Save clustering metrics
clustering_metrics <- data.frame(
  k = k_values,
  silhouette_score = silhouette_scores,
  within_ss = within_ss
)
write_csv(clustering_metrics, file.path(results_dir, "clustering_metrics.csv"))

cat("\n=== Sentence Embedding Analysis Complete ===\n")
cat(glue("Results saved to: {results_dir}\n"))
cat("\nKey outputs:\n")
cat("- sentence_embedding_clusters.csv: Full data with cluster assignments\n")
cat("- cluster_tfidf_analysis.csv: TF-IDF analysis of cluster characteristics\n")
cat("- cluster_top_words.csv: Top words summarizing each cluster\n")
cat("- clustering_metrics.csv: Validation metrics for different K values\n")
cat("- analysis_summary.csv: Overall analysis summary\n")

cat(glue("\n🎯 Semantic analysis identified {optimal_k} distinct thematic clusters.\n"))
cat("These represent semantically coherent groups based on utterance meaning,\n")
cat("not just word co-occurrence patterns. Review representative utterances\n")
cat("and characteristic terms to interpret the underlying motivational themes.\n")