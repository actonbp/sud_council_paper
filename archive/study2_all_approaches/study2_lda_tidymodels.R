# Enhanced LDA Topic Modeling with Tidymodels Integration
# Purpose: Clean topic separation using topicmodels::LDA with tidytext/tidymodels workflow
# Input: data/focus_group_substantive.csv
# Output: Clean LDA topics with coherence metrics and validation

# Load required libraries
library(tidyverse)
library(tidytext)
library(tidymodels)
library(topicmodels)
library(ldatuning)
library(SnowballC)
library(here)
library(glue)

# Set up paths
data_path <- here("data", "focus_group_substantive.csv")
results_dir <- here("results", "r", "study2_lda_tidymodels")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# Read preprocessed focus group data
cat("\n=== Reading Focus Group Data ===\n")
focus_data <- read_csv(data_path, show_col_types = FALSE)
cat(glue("Total substantive utterances: {nrow(focus_data)}\n"))

# Enhanced stopword list (using BTM preprocessing approach)
extra_stop <- c(
  # Generic filler words from BTM analysis
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

# Text preprocessing following BTM approach
cat("\n=== Preprocessing Text (BTM-style) ===\n")

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
  mutate(word_stem = SnowballC::wordStem(word)) %>%
  # Filter out overly aggressive stems (too short or nonsensical)
  filter(nchar(word_stem) >= 3, !word_stem %in% c("ad", "ads", "didn", "wasn", "couldn", "wouldn"))

# Remove rare terms (appear in < 3 documents) for clean topics
word_doc_counts <- tokens %>%
  distinct(doc_id, word_stem) %>%
  count(word_stem) %>%
  filter(n >= 3)

tokens_filtered <- tokens %>%
  semi_join(word_doc_counts, by = "word_stem")

cat(glue("Unique tokens after filtering: {n_distinct(tokens_filtered$word_stem)}\n"))
cat(glue("Average tokens per document: {round(nrow(tokens_filtered) / n_distinct(tokens_filtered$doc_id), 1)}\n"))

# Create word counts for document-term matrix
word_counts <- tokens_filtered %>%
  count(doc_id, word_stem) %>%
  filter(n > 0)  # Ensure positive counts

cat(glue("Documents with tokens: {n_distinct(word_counts$doc_id)}\n"))
cat(glue("Total word-document pairs: {nrow(word_counts)}\n"))

# Create document-term matrix
cat("\n=== Creating Document-Term Matrix ===\n")
focus_dtm <- word_counts %>%
  cast_dtm(doc_id, word_stem, n)

cat(glue("DTM dimensions: {nrow(focus_dtm)} documents × {ncol(focus_dtm)} terms\n"))

# Ensure we have enough data for meaningful modeling
if (ncol(focus_dtm) < 10) {
  stop("Too few terms remaining (", ncol(focus_dtm), "). Consider relaxing filtering criteria.")
}

# Function to calculate topic coherence (simplified version)
calculate_coherence <- function(model, dtm, top_n = 10) {
  # Extract topic-term matrix
  topic_terms <- tidy(model, matrix = "beta") %>%
    group_by(topic) %>%
    slice_max(beta, n = top_n) %>%
    pull(term)
  
  # Simple coherence approximation based on term co-occurrence
  # Higher coherence = terms appear together more often
  coherence_scores <- numeric(model@k)
  
  for (i in 1:model@k) {
    topic_start <- (i-1) * top_n + 1
    topic_end <- i * top_n
    topic_terms_i <- topic_terms[topic_start:topic_end]
    
    # Calculate pairwise term co-occurrence in documents
    term_cooccur <- 0
    term_pairs <- 0
    
    for (j in seq_along(topic_terms_i)) {
      for (k in seq_along(topic_terms_i)) {
        if (j < k) {
          term_pairs <- term_pairs + 1
          # Count documents containing both terms
          docs_with_both <- sum(as.matrix(dtm[, topic_terms_i[j]]) > 0 & 
                               as.matrix(dtm[, topic_terms_i[k]]) > 0)
          if (docs_with_both > 0) {
            term_cooccur <- term_cooccur + 1
          }
        }
      }
    }
    
    coherence_scores[i] <- if (term_pairs > 0) term_cooccur / term_pairs else 0
  }
  
  mean(coherence_scores)
}

# Model fitting and evaluation
cat("\n=== Fitting LDA Models with Evaluation Metrics ===\n")

# Test k values 2, 3, 4 as requested
k_values <- 2:4
model_results <- tibble(k = k_values)

# Fit models and calculate metrics
model_results <- model_results %>%
  mutate(
    model = map(k, ~{
      cat(glue("Fitting LDA model with k = {.x}...\n"))
      LDA(focus_dtm, 
          k = .x, 
          method = "Gibbs",
          control = list(
            seed = 1234,
            iter = 2000,
            burnin = 200,
            thin = 10,
            alpha = 50/.x,  # Symmetric Dirichlet prior
            delta = 0.1     # Beta parameter for term distributions
          ))
    }),
    perplexity = map_dbl(model, ~perplexity(.x, focus_dtm)),
    coherence = map_dbl(model, ~calculate_coherence(.x, focus_dtm))
  )

# Display metrics
cat("\n=== Model Comparison Metrics ===\n")
metrics_summary <- model_results %>%
  select(k, perplexity, coherence) %>%
  mutate(
    perplexity = round(perplexity, 2),
    coherence = round(coherence, 3)
  )

print(metrics_summary)

# Model selection: prioritize coherence for clean separation
# Lower perplexity is better, higher coherence is better
model_results <- model_results %>%
  mutate(
    # Normalize metrics for comparison (0-1 scale)
    perp_norm = (max(perplexity) - perplexity) / (max(perplexity) - min(perplexity)),
    coh_norm = (coherence - min(coherence)) / (max(coherence) - min(coherence)),
    # Combined score favoring coherence for clean topics
    combined_score = 0.3 * perp_norm + 0.7 * coh_norm
  )

optimal_k <- model_results$k[which.max(model_results$combined_score)]
optimal_model <- model_results$model[[which.max(model_results$combined_score)]]

cat(glue("\n=== Model Selection Results ===\n"))
cat(glue("Selected k = {optimal_k} (prioritizing topic coherence)\n"))
cat(glue("Best perplexity: {round(model_results$perplexity[model_results$k == optimal_k], 2)}\n"))
cat(glue("Best coherence: {round(model_results$coherence[model_results$k == optimal_k], 3)}\n"))

# Extract topic information using tidytext
cat("\n=== Extracting Topic Information ===\n")

# Topic-term probabilities (beta)
topic_terms <- tidy(optimal_model, matrix = "beta")

# Document-topic probabilities (gamma)
topic_documents <- tidy(optimal_model, matrix = "gamma") %>%
  # Join back with original document metadata
  left_join(
    focus_data %>% mutate(document = as.character(row_number())),
    by = "document"
  )

# Top terms per topic for interpretation
top_terms_per_topic <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 15) %>%
  ungroup() %>%
  arrange(topic, desc(beta))

# Display top terms for each topic
cat("\n=== Top Terms per Topic ===\n")
for (i in 1:optimal_k) {
  topic_terms_i <- top_terms_per_topic %>%
    filter(topic == i) %>%
    slice_head(n = 10) %>%
    pull(term)
  
  cat(glue("Topic {i}: {paste(topic_terms_i, collapse = ', ')}\n"))
}

# Topic prevalence analysis
topic_prevalence <- topic_documents %>%
  group_by(topic) %>%
  summarise(
    avg_gamma = mean(gamma),
    median_gamma = median(gamma),
    n_dominant = sum(gamma > 0.5),
    n_present = sum(gamma > 0.2),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_gamma))

cat("\n=== Topic Prevalence Summary ===\n")
print(topic_prevalence)

# Save results
cat("\n=== Saving Results ===\n")

# Model comparison metrics
write_csv(metrics_summary, file.path(results_dir, "lda_model_comparison.csv"))

# Topic-term probabilities
write_csv(topic_terms, file.path(results_dir, "lda_topic_term_probabilities.csv"))

# Document-topic probabilities
write_csv(topic_documents, file.path(results_dir, "lda_document_topic_probabilities.csv"))

# Top terms per topic
write_csv(top_terms_per_topic, file.path(results_dir, "lda_top_terms_per_topic.csv"))

# Topic prevalence
write_csv(topic_prevalence, file.path(results_dir, "lda_topic_prevalence.csv"))

# Model summary
model_summary <- tibble(
  analysis_date = Sys.Date(),
  approach = "Enhanced LDA with coherence-based selection",
  optimal_k = optimal_k,
  perplexity = model_results$perplexity[model_results$k == optimal_k],
  coherence = model_results$coherence[model_results$k == optimal_k],
  n_documents = nrow(focus_dtm),
  n_terms = ncol(focus_dtm),
  preprocessing_approach = "BTM-style with enhanced stopwords",
  gibbs_iterations = 2000,
  burnin = 200,
  alpha = 50/optimal_k,
  delta = 0.1
)

write_csv(model_summary, file.path(results_dir, "lda_model_summary.csv"))

# Save model object for future use
saveRDS(optimal_model, file.path(results_dir, glue("lda_model_k{optimal_k}.rds")))

# Print completion message
cat("\n=== Enhanced LDA Topic Modeling Complete ===\n")
cat(glue("Results saved to: {results_dir}\n"))
cat("\nKey outputs:\n")
cat("- lda_model_comparison.csv: Perplexity and coherence metrics for k=2-4\n")
cat("- lda_topic_term_probabilities.csv: Full topic-term probability matrix\n")
cat("- lda_document_topic_probabilities.csv: Document-topic assignments\n")
cat("- lda_top_terms_per_topic.csv: Top 15 terms per topic\n")
cat("- lda_topic_prevalence.csv: Topic distribution summary\n")
cat("- lda_model_summary.csv: Model configuration and final metrics\n")
cat(glue("- lda_model_k{optimal_k}.rds: Fitted model object\n"))

cat("\n=== Methodology Summary ===\n")
cat("✓ BTM-style preprocessing with enhanced stopword filtering\n")
cat("✓ Document-term matrix with minimum 3-document term frequency\n")
cat("✓ Gibbs sampling LDA with 2000 iterations, 200 burnin\n")
cat("✓ Coherence-prioritized model selection for clean topic separation\n")
cat("✓ Comprehensive evaluation metrics (perplexity + coherence)\n")
cat(glue("✓ Final model: k={optimal_k} topics with {ncol(focus_dtm)} terms across {nrow(focus_dtm)} documents\n"))