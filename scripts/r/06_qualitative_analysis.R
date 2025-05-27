# 06_qualitative_analysis.R
# This script analyzes qualitative interview data (Study 2) using thematic analysis
# and text embeddings for theme extraction

# Load required packages
library(tidyverse)
library(text2vec)
library(topicmodels)
library(stm)
library(udpipe)
library(quanteda)
library(umap)
library(factoextra)
library(here)

# Configuration
interview_data_dir <- here("data", "interviews")
results_dir <- here("results", "r", "study2_qualitative")
random_state <- 42

# Create results directory
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# --- Load Interview Transcripts ---
cat("Loading interview transcripts...\n")

# This function would load actual interview data
# For this template, we're creating a placeholder function
load_interview_data <- function() {
  # In practice, this would load actual transcript files
  # For example:
  # files <- list.files(interview_data_dir, pattern = "*.txt", full.names = TRUE)
  # interviews <- map_df(files, function(file) {
  #   text <- read_file(file)
  #   participant_id <- gsub(".*?([0-9]+).*", "\\1", basename(file))
  #   tibble(participant_id = participant_id, transcript = text)
  # })
  
  # Placeholder data
  tibble(
    participant_id = paste0("P", 1:10),
    transcript = c(
      "I've always been interested in helping people with addiction. My family has experience with SUD.",
      "The stigma around addiction makes it challenging. I worry about burnout in this field.",
      "I think counseling for substance use requires special training. I'm interested but worried about job security.",
      "Mental health counseling seems more versatile to me than focusing just on substance use.",
      "My psychology classes introduced me to this field. The professor was a former SUD counselor.",
      "I'd like to work in this area but I'm concerned about the emotional toll and relatively low pay.",
      "My exposure to SUD counseling has been limited. I wish there were more courses on it.",
      "The opioid crisis makes this field important, but I'm not sure if I'm emotionally prepared for it.",
      "I've volunteered at a recovery center and found the work meaningful but challenging.",
      "The certification requirements seem complicated. I'm not sure if it's worth the investment."
    ),
    gender = sample(c("Woman", "Man", "Gender Diverse"), 10, replace = TRUE),
    age = sample(18:35, 10, replace = TRUE),
    interest_level = sample(c("Not interested", "Slightly", "Moderately", "Very"), 10, replace = TRUE)
  )
}

# Load or create placeholder data
interview_data <- load_interview_data()
cat(paste0("Loaded ", nrow(interview_data), " interview transcripts\n"))

# --- Preprocess Text Data ---
cat("\nPreprocessing text data...\n")

# Create a corpus
corpus <- corpus(interview_data$transcript, docnames = interview_data$participant_id)
cat(paste0("Created corpus with ", ndoc(corpus), " documents\n"))

# Tokenization and preprocessing
tokens <- tokens(corpus, remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE) %>%
  tokens_remove(stopwords("english")) %>%
  tokens_wordstem()

# Create document-feature matrix
dfm <- dfm(tokens)
cat(paste0("Created document-feature matrix with ", nfeat(dfm), " features\n"))

# Trim sparse terms
dfm_trimmed <- dfm_trim(dfm, min_docfreq = 2)
cat(paste0("Trimmed to ", nfeat(dfm_trimmed), " features occurring in at least 2 documents\n"))

# --- Approach 1: Topic Modeling ---
cat("\nPerforming topic modeling...\n")

# Determine optimal number of topics (k)
# In practice, you would run topic models with different k values and evaluate
find_optimal_k <- function(dfm, max_k = 10) {
  coherence_values <- numeric(max_k - 1)
  for (k in 2:max_k) {
    cat(paste0("Testing model with k = ", k, "...\n"))
    set.seed(random_state)
    
    # For LDA model
    lda_model <- LDA(dfm, k = k, control = list(seed = random_state))
    
    # Calculate coherence or perplexity
    # This is a placeholder - real implementation would calculate actual metrics
    coherence_values[k-1] <- k * -1  # Placeholder
  }
  
  # Return optimal k
  which.max(coherence_values) + 1
}

# Find optimal k (commented out as it's computationally intensive)
# optimal_k <- find_optimal_k(dfm_trimmed)
optimal_k <- 4  # Placeholder, typically determined through evaluation

# Run LDA with optimal k
cat(paste0("\nRunning LDA with ", optimal_k, " topics...\n"))
set.seed(random_state)
lda_model <- LDA(dfm_trimmed, k = optimal_k, control = list(seed = random_state))

# Extract topics
topics <- tidy(lda_model, matrix = "beta")

# Get top terms for each topic
top_terms <- topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  arrange(topic, -beta)

cat("\nTop terms for each topic:\n")
print(top_terms)

# Save topic terms
write_csv(top_terms, file.path(results_dir, "topic_top_terms.csv"))

# Get document topics
document_topics <- tidy(lda_model, matrix = "gamma") %>%
  pivot_wider(names_from = topic, values_from = gamma, names_prefix = "topic_")

# Join with original data
interview_data_with_topics <- interview_data %>%
  mutate(document = row_number()) %>%
  left_join(document_topics, by = "document")

# Save document topic assignments
write_csv(interview_data_with_topics, file.path(results_dir, "document_topics.csv"))

# --- Approach 2: Text Embeddings and Clustering ---
cat("\nGenerating text embeddings and clustering...\n")

# Create GloVe embeddings (in practice, you might use pre-trained embeddings)
tokens_list <- as.list(as.character(interview_data$transcript))
it <- itoken(tokens_list, progressbar = FALSE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 2)
vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5)
glove_model <- GloVe$new(rank = 50, x_max = 10)
set.seed(random_state)
wv <- glove_model$fit_transform(tcm, n_iter = 50)

# Function to get document embeddings by averaging word vectors
get_doc_embedding <- function(doc, word_vectors, vocab) {
  words <- tolower(unlist(strsplit(doc, "\\W+")))
  words <- words[words != ""]
  word_ids <- match(words, vocab$term)
  word_ids <- word_ids[!is.na(word_ids)]
  
  if (length(word_ids) == 0) {
    return(rep(0, ncol(word_vectors)))
  }
  
  doc_vec <- colMeans(word_vectors[word_ids, , drop = FALSE])
  return(doc_vec)
}

# Get document embeddings
doc_embeddings <- map(interview_data$transcript, 
                      ~get_doc_embedding(., wv, vocab)) %>%
  do.call(rbind, .)

# Reduce dimensionality with UMAP for visualization
set.seed(random_state)
umap_result <- umap(doc_embeddings, n_neighbors = 3, min_dist = 0.1, n_components = 2)

# Cluster the embeddings
set.seed(random_state)
kmeans_result <- kmeans(doc_embeddings, centers = optimal_k, nstart = 25)

# Add clusters and UMAP coordinates to data
interview_data_clustered <- interview_data %>%
  mutate(
    cluster = kmeans_result$cluster,
    umap_x = umap_result$layout[, 1],
    umap_y = umap_result$layout[, 2]
  )

# Save clustered data
write_csv(interview_data_clustered, file.path(results_dir, "interview_clusters.csv"))

# Plot clusters
cluster_plot <- ggplot(interview_data_clustered, aes(x = umap_x, y = umap_y, color = factor(cluster))) +
  geom_point(size = 3, alpha = 0.7) +
  labs(
    title = "Interview Clusters Based on Text Embeddings",
    x = "UMAP Dimension 1",
    y = "UMAP Dimension 2",
    color = "Cluster"
  ) +
  theme_minimal()

ggsave(file.path(results_dir, "interview_clusters.png"), cluster_plot, width = 8, height = 6)

# --- Extract Representative Quotes ---
cat("\nExtracting representative quotes for each theme...\n")

# Function to find the most central document in each cluster
find_central_docs <- function(embeddings, clusters, n = 2) {
  cluster_centers <- aggregate(embeddings, by = list(cluster = clusters), FUN = mean)
  
  central_docs <- map_dfr(unique(clusters), function(c) {
    cluster_docs <- which(clusters == c)
    cluster_center <- cluster_centers[cluster_centers$cluster == c, -1]
    
    # Calculate distance from each document to cluster center
    distances <- map_dbl(cluster_docs, function(doc_idx) {
      sqrt(sum((embeddings[doc_idx, ] - as.numeric(cluster_center))^2))
    })
    
    # Get indices of n closest documents
    closest_indices <- cluster_docs[order(distances)[1:min(n, length(distances))]]
    
    tibble(
      cluster = c,
      doc_idx = closest_indices,
      distance = distances[order(distances)[1:min(n, length(distances))]]
    )
  })
  
  return(central_docs)
}

# Find central documents
central_docs <- find_central_docs(doc_embeddings, kmeans_result$cluster)

# Extract representative quotes
representative_quotes <- central_docs %>%
  left_join(interview_data %>% mutate(doc_idx = row_number()), by = "doc_idx") %>%
  group_by(cluster) %>%
  summarize(
    cluster_size = sum(kmeans_result$cluster == first(cluster)),
    representative_quotes = list(transcript),
    participant_ids = list(participant_id)
  )

# Save representative quotes
write_rds(representative_quotes, file.path(results_dir, "representative_quotes.rds"))

# --- Generate Theme Descriptions ---
cat("\nGenerating theme descriptions...\n")

# In practice, you would analyze the terms and quotes to generate these manually
# For this template, we're creating a placeholder function
generate_theme_descriptions <- function(top_terms, representative_quotes) {
  # This would be a manual process in practice
  # Here we're just creating placeholder descriptions
  themes <- tibble(
    cluster = 1:optimal_k,
    theme_name = c(
      "Personal Connection and Motivation",
      "Concerns and Professional Barriers",
      "Education and Exposure",
      "Field Importance and Requirements"
    ),
    theme_description = c(
      "Participants discussing personal connections to SUD and motivations to help others with addiction",
      "Concerns about burnout, stigma, emotional toll, and career stability in SUD counseling",
      "Experiences with education, training, and exposure to SUD counseling",
      "Perspectives on field importance, certification requirements, and professional development"
    )
  )
  
  return(themes)
}

theme_descriptions <- generate_theme_descriptions(top_terms, representative_quotes)
write_csv(theme_descriptions, file.path(results_dir, "theme_descriptions.csv"))

# --- Generate Summary Report ---
cat("\nGenerating qualitative analysis summary...\n")

# Create a markdown summary
summary_md <- c(
  "# Study 2: Qualitative Analysis Summary",
  "",
  paste("## Overview"),
  paste("- Total interviews analyzed:", nrow(interview_data)),
  paste("- Identified themes:", optimal_k),
  "",
  "## Identified Themes"
)

for (i in 1:nrow(theme_descriptions)) {
  theme_info <- theme_descriptions[i, ]
  quotes <- representative_quotes %>% filter(cluster == theme_info$cluster)
  
  theme_summary <- c(
    "",
    paste("### Theme", i, ":", theme_info$theme_name),
    paste(theme_info$theme_description),
    "",
    paste("**Size:** Mentioned by", quotes$cluster_size, "participants"),
    "",
    "**Representative quotes:**"
  )
  
  for (j in seq_along(unlist(quotes$representative_quotes))) {
    quote_text <- unlist(quotes$representative_quotes)[j]
    participant <- unlist(quotes$participant_ids)[j]
    theme_summary <- c(theme_summary, paste0("- \"", quote_text, "\" (", participant, ")"))
  }
  
  summary_md <- c(summary_md, theme_summary)
}

summary_md <- c(
  summary_md,
  "",
  "## Methods",
  "Interview transcripts were analyzed using a combination of:",
  "1. Topic modeling with Latent Dirichlet Allocation (LDA)",
  "2. Text embeddings using GloVe (Global Vectors for Word Representation)",
  "3. Clustering with K-means on document embeddings",
  "4. Manual thematic analysis of representative quotes"
)

write_lines(summary_md, file.path(results_dir, "qualitative_analysis_summary.md"))

cat("\nQualitative analysis completed successfully.\n")