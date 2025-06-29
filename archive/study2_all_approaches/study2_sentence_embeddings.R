# Sentence Embeddings + Clustering for Study 2 Focus Groups
# Purpose: Use sentence embeddings to cluster utterances semantically
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
library(text)  # For sentence embeddings

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

# Initialize text package (this may take a while on first run)
cat("\n=== Initializing Text Package for Embeddings ===\n")
# This will download pre-trained models if not already available
text::textEmbed_initialize()

# Get sentence embeddings
cat("\n=== Generating Sentence Embeddings ===\n")
cat("This may take several minutes depending on the number of utterances...\n")

# Generate embeddings using text package
embeddings <- text::textEmbed(
  texts = cleaned_data$cleaned_for_embedding,
  model = "sentence-transformers/all-MiniLM-L6-v2",  # Fast, good quality model
  device = "cpu",  # Use CPU (change to "cuda" if GPU available)
  tokenizer_parallelism = FALSE
)

# Extract embedding matrix
embedding_matrix <- embeddings$texts

cat(glue("Embedding dimensions: {ncol(embedding_matrix)}\n"))

# Apply UMAP for dimensionality reduction
cat("\n=== Applying UMAP Dimensionality Reduction ===\n")
umap_config <- umap.defaults
umap_config$n_neighbors <- min(15, nrow(embedding_matrix) - 1)
umap_config$min_dist <- 0.1
umap_config$n_components <- 3  # 3D for better clustering

umap_result <- umap(embedding_matrix, config = umap_config)
umap_coords <- as.data.frame(umap_result$layout)
colnames(umap_coords) <- c("UMAP1", "UMAP2", "UMAP3")

# Test different numbers of clusters
cat("\n=== Testing Different Cluster Numbers ===\n")
k_values <- 2:6
silhouette_scores <- numeric(length(k_values))

for(i in seq_along(k_values)) {
  k <- k_values[i]
  kmeans_result <- kmeans(umap_coords, centers = k, nstart = 25, iter.max = 100)
  sil_score <- cluster::silhouette(kmeans_result$cluster, dist(umap_coords))
  silhouette_scores[i] <- mean(sil_score[, 3])
  cat(glue("K = {k}: Silhouette score = {round(silhouette_scores[i], 3)}\n"))
}

# Select optimal K
optimal_k <- k_values[which.max(silhouette_scores)]
cat(glue("\nOptimal K = {optimal_k} (highest silhouette score: {round(max(silhouette_scores), 3)})\n"))

# Final clustering with optimal K
final_kmeans <- kmeans(umap_coords, centers = optimal_k, nstart = 25, iter.max = 100)

# Add cluster assignments to data
clustered_data <- cleaned_data %>%
  bind_cols(umap_coords) %>%
  mutate(cluster = final_kmeans$cluster)

# Save cluster assignments
write_csv(clustered_data, file.path(results_dir, "embedding_cluster_assignments.csv"))

# Analyze cluster characteristics
cat("\n=== Analyzing Cluster Characteristics ===\n")

cluster_summary <- clustered_data %>%
  count(cluster) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 1),
    avg_words = map_dbl(cluster, ~mean(clustered_data$word_count[clustered_data$cluster == .x]))
  )

print(cluster_summary)

# Extract top words for each cluster using TF-IDF
cat("\n=== Extracting Cluster Themes via TF-IDF ===\n")

# Tokenize the cleaned text for theme extraction
cluster_words <- clustered_data %>%
  select(doc_id, cluster, cleaned_for_embedding) %>%
  unnest_tokens(word, cleaned_for_embedding) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2) %>%
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
  slice_max(tf_idf, n = 15) %>%
  summarise(
    top_words = paste(word, collapse = ", "),
    .groups = "drop"
  )

# Save top words
write_csv(cluster_tfidf, file.path(results_dir, "cluster_tfidf_words.csv"))

# Print cluster themes
cat("\n=== Cluster Themes (Top TF-IDF Words) ===\n")
for(i in 1:optimal_k) {
  cluster_info <- cluster_summary %>% filter(cluster == i)
  top_words <- top_words_per_cluster %>% filter(cluster == i) %>% pull(top_words)
  
  cat(glue("\n**Cluster {i}**: {cluster_info$n} utterances ({cluster_info$percentage}%)\n"))
  cat(glue("Top words: {str_trunc(top_words, 80)}\n"))
}

# Sample utterances from each cluster for interpretation
cat("\n=== Sample Utterances per Cluster ===\n")
for(i in 1:optimal_k) {
  cat(glue("\n--- Cluster {i} Sample Utterances ---\n"))
  samples <- clustered_data %>%
    filter(cluster == i) %>%
    slice_sample(n = min(3, n())) %>%
    pull(cleaned_text)
  
  for(j in seq_along(samples)) {
    cat(glue("{j}. {str_trunc(samples[j], 100)}\n"))
  }
}

# Create summary metrics
summary_metrics <- data.frame(
  metric = c("Total Utterances", "Optimal K", "Max Silhouette Score", 
             "Embedding Dimensions", "UMAP Components"),
  value = c(nrow(clustered_data), optimal_k, round(max(silhouette_scores), 3),
            ncol(embedding_matrix), 3)
)

write_csv(summary_metrics, file.path(results_dir, "embedding_analysis_summary.csv"))

# Save silhouette scores for all K values
k_metrics <- data.frame(
  k = k_values,
  silhouette_score = silhouette_scores
)
write_csv(k_metrics, file.path(results_dir, "silhouette_scores.csv"))

cat("\n=== Sentence Embedding Analysis Complete ===\n")
cat(glue("Results saved to: {results_dir}\n"))
cat("\nKey outputs:\n")
cat("- embedding_cluster_assignments.csv: Utterances with cluster assignments and UMAP coordinates\n")
cat("- cluster_tfidf_words.csv: TF-IDF analysis of words per cluster\n")
cat("- silhouette_scores.csv: Cluster validation metrics for different K values\n")
cat("- embedding_analysis_summary.csv: Overall analysis summary\n")

# Final message
cat(glue("\nSemantic clustering identified {optimal_k} distinct themes in focus group discussions.\n"))
cat("Review the sample utterances and top words to interpret the thematic content.\n")