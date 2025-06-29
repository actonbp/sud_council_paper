# Biterm Topic Model (BTM) for Study 2 Focus Groups
# Purpose: Apply BTM topic modeling to short focus group utterances
# Input: data/focus_group_substantive.csv
# Output: BTM topic model results with optimal K selection

# Load required libraries
library(BTM)
library(tidyverse)
library(tidytext)
library(SnowballC)
library(here)
library(glue)

# Set up paths
data_path <- here("data", "focus_group_substantive.csv")
results_dir <- here("results", "r", "study2_btm_modeling")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# Read preprocessed focus group data
cat("\n=== Reading Focus Group Data ===\n")
focus_data <- read_csv(data_path, show_col_types = FALSE)
cat(glue("Total substantive utterances: {nrow(focus_data)}\n"))

# Enhanced stopword list combining tidytext defaults with domain-specific terms
extra_stop <- c(
  # Generic filler words from TF-IDF analysis
  "lot", "yeah", "um", "uh", "dont", "stuff", "thing", "things", 
  "gonna", "gotta", "kinda", "sorta", "basically", "actually",
  "really", "maybe", "probably", "definitely", "obviously",
  
  # Career-related generic terms (too common to be informative)
  "job", "jobs", "work", "working", "career", "careers",
  "field", "fields", "area", "areas", "profession", "professional",
  
  # SUD-specific terms to remove (we're already filtering for these)
  "substance", "substances", "substanc", "counselor", "counselors", "counseling",
  "alcohol", "alcoholic", "alcoholism", "drug", "drugs", "addiction", "addictions",
  "sud", "abuse", "abus", "disorder", "disorders", "use", "user", "users",
  "treatment", "treatments", "recovery", "addict", "addicts", "addicted",
  
  # Mental health career terms to remove (we want factors, not career names)
  "mental", "health", "therapist", "therapists", "therapy", "therapies",
  "psychology", "psychologist", "psychologists", "psych", "psychiatric", "psychiatrist",
  "social", "worker", "workers", "nurse", "nurses", "nursing", "nurs",
  "clinical", "clinician", "practitioner", "practitioners", "provider", "providers",
  
  # Very generic terms that don't add meaning
  "people", "person", "someone", "somebody", "everyone", "everybody",
  "different", "similar", "good", "bad", "better", "worse", "best", "worst",
  
  # Time/quantity terms
  "time", "times", "years", "year", "day", "days", "week", "weeks",
  "one", "two", "three", "first", "second", "third",
  
  # Conversation markers
  "know", "knows", "knowing", "think", "thinking", "thought",
  "feel", "feeling", "felt", "say", "saying", "said",
  "talk", "talking", "talked", "tell", "telling", "told",
  
  # Auxiliary/modal verbs
  "can", "cant", "could", "couldnt", "would", "wouldnt", "wouldn",
  "should", "shouldnt", "will", "wont", "might", "must",
  "got", "get", "gets", "getting", "go", "going", "goes",
  
  # Conversation filler words and generic stems
  "don", "guess", "take", "takes", "taking", "took", "pre", "pretti",
  "didn", "wasn", "couldn", "wouldn", "isn", "aren", "haven", "hasn",
  "ad", "ads", "didn't", "wasn't", "couldn't", "wouldn't", "isn't", "aren't"
)

# Text preprocessing for BTM
cat("\n=== Preprocessing Text for BTM ===\n")

# Tokenize and preprocess
tokens <- focus_data %>%
  mutate(doc_id = row_number()) %>%
  unnest_tokens(word, cleaned_text) %>%
  # Remove standard + custom stopwords
  anti_join(stop_words, by = "word") %>%
  filter(!word %in% extra_stop) %>%
  # Remove numbers and very short words
  filter(!str_detect(word, "^\\d+$"), nchar(word) > 2) %>%
  # Apply tidytext-style stemming with SnowballC
  mutate(word = SnowballC::wordStem(word)) %>%
  # Filter out overly aggressive stems (too short or nonsensical)
  filter(nchar(word) >= 3, !word %in% c("ad", "ads", "didn", "wasn", "couldn", "wouldn"))

# Remove rare terms (appear in < 3 documents)
word_doc_counts <- tokens %>%
  distinct(doc_id, word) %>%
  count(word) %>%
  filter(n >= 3)

tokens_filtered <- tokens %>%
  semi_join(word_doc_counts, by = "word")

cat(glue("Unique tokens after filtering: {n_distinct(tokens_filtered$word)}\n"))
cat(glue("Average tokens per document: {round(nrow(tokens_filtered) / n_distinct(tokens_filtered$doc_id), 1)}\n"))

# Prepare data for BTM (requires specific format)
btm_data <- tokens_filtered %>%
  select(doc_id, word) %>%
  as.data.frame()

# Function to fit BTM and calculate model metrics
fit_btm_model <- function(k, data) {
  cat(glue("\nFitting BTM with K = {k} topics...\n"))
  
  # Fit BTM model
  model <- BTM(
    data, 
    k = k,
    iter = 2000,
    alpha = 50/k,  # Standard Dirichlet prior
    beta = 0.01,   # Promotes sparsity
    trace = 100
  )
  
  # Calculate log-likelihood (higher is better)
  loglik_obj <- logLik(model)
  loglik <- loglik_obj$ll
  
  # Extract theta (topic proportions) for entropy calculation
  # BTM has theta as a vector of overall topic proportions
  theta <- model$theta
  entropy <- -sum(theta * log(theta + 1e-10), na.rm = TRUE)
  
  list(
    k = k,
    model = model,
    loglik = loglik,
    entropy = entropy
  )
}

# Fit models for different K values
cat("\n=== Testing Different Topic Numbers (K) ===\n")
k_values <- 2:8
model_results <- map(k_values, ~fit_btm_model(.x, btm_data))

# Extract metrics for model selection
metrics_df <- model_results %>%
  map_df(~data.frame(
    k = .x$k,
    loglik = .x$loglik,
    entropy = .x$entropy
  ))

# Save metrics
write_csv(metrics_df, file.path(results_dir, "btm_model_metrics.csv"))

# Print metrics
cat("\n=== Model Selection Metrics ===\n")
print(metrics_df)

# Select optimal K based on elbow analysis
# Calculate improvement rates
metrics_df <- metrics_df %>%
  arrange(k) %>%
  mutate(
    loglik_diff = loglik - lag(loglik),
    improvement_pct = (loglik - lag(loglik)) / abs(lag(loglik)) * 100
  )

cat("\n=== Improvement Analysis ===\n")
print(metrics_df %>% select(k, loglik, loglik_diff, improvement_pct))

# Force K=3 for cleaner, non-overlapping topics
# The data shows good improvements up to K=5, but topics start overlapping
optimal_k <- 3

cat(glue("\nSelected K = {optimal_k} for cleaner topic separation\n"))
cat("(Forcing K=3 to avoid topic overlap seen in higher K values)\n")

# Extract best model
best_model <- model_results[[which(k_values == optimal_k)]]$model

# Extract and save topic-term distributions
cat("\n=== Extracting Topic-Term Distributions ===\n")

# Get top terms per topic
terms_per_topic <- 20
topic_terms <- terms(best_model, top_n = terms_per_topic)

# Convert to tidy format
topic_terms_df <- map_df(1:optimal_k, function(topic) {
  topic_df <- topic_terms[[topic]]
  data.frame(
    topic = topic,
    rank = 1:nrow(topic_df),
    term = topic_df$token,
    probability = topic_df$probability,
    stringsAsFactors = FALSE
  )
})

# Save topic terms
write_csv(topic_terms_df, file.path(results_dir, "btm_top_terms_per_topic.csv"))

# Print top 10 terms per topic
cat("\n=== Top Terms per Topic ===\n")
topic_terms_df %>%
  filter(rank <= 10) %>%
  group_by(topic) %>%
  summarise(
    top_terms = paste(term, collapse = ", ")
  ) %>%
  print()

# Extract document-topic probabilities
cat("\n=== Document-Topic Probabilities ===\n")

# Predict topic probabilities for each document
doc_topics <- predict(best_model, newdata = btm_data)

# Get the document IDs that have predictions
predicted_doc_ids <- as.integer(rownames(doc_topics))

# Create tidy format with document metadata
doc_topic_df <- focus_data %>%
  mutate(doc_id = row_number()) %>%
  filter(doc_id %in% predicted_doc_ids) %>%
  select(doc_id, session_id, participant_id, cleaned_text) %>%
  bind_cols(as.data.frame(doc_topics)) %>%
  pivot_longer(
    cols = starts_with("V"),
    names_to = "topic_temp",
    values_to = "probability"
  ) %>%
  mutate(topic = as.integer(str_extract(topic_temp, "\\d+"))) %>%
  select(-topic_temp)

# Save document-topic probabilities
write_csv(doc_topic_df, file.path(results_dir, "btm_document_topic_probs.csv"))

# Summary statistics
topic_summary <- doc_topic_df %>%
  group_by(topic) %>%
  summarise(
    avg_probability = mean(probability),
    n_dominant = sum(probability > 0.5),
    n_present = sum(probability > 0.1)
  ) %>%
  arrange(desc(avg_probability))

cat("\n=== Topic Distribution Summary ===\n")
print(topic_summary)

# Create model summary
model_summary <- data.frame(
  metric = c("Optimal K", "Log-likelihood", "Entropy", "Iterations", 
             "Alpha", "Beta", "Total Documents", "Vocabulary Size"),
  value = c(optimal_k, metrics_df$loglik[metrics_df$k == optimal_k], metrics_df$entropy[metrics_df$k == optimal_k],
            2000, 50/optimal_k, 0.01, n_distinct(btm_data$doc_id), n_distinct(btm_data$word))
)

# Save model summary
write_csv(model_summary, file.path(results_dir, "btm_model_summary.csv"))

# Print completion message
cat("\n=== BTM Topic Modeling Complete ===\n")
cat(glue("Results saved to: {results_dir}\n"))
cat("\nKey outputs:\n")
cat("- btm_model_metrics.csv: Model selection metrics for K=2-8\n")
cat("- btm_top_terms_per_topic.csv: Top 20 terms per topic\n")
cat("- btm_document_topic_probs.csv: Document-topic probability matrix\n")
cat("- btm_model_summary.csv: Model configuration and metrics\n")

# Save the model object for future use
saveRDS(best_model, file.path(results_dir, glue("btm_model_k{optimal_k}.rds")))
cat(glue("\nModel object saved as: btm_model_k{optimal_k}.rds\n"))