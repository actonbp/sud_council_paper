# =============================================================================
# Study 2: Iterative Parameter Tuning for Topic Quality
# SUD Counseling Career Research Project
# =============================================================================
# Easy parameter adjustment and re-running for topic refinement
# Date: June 12, 2025
# =============================================================================

library(tidymodels)
library(textrecipes)
library(tidytext)
library(here)
library(glue)
library(readr)
library(SnowballC)
library(topicmodels)
library(stringr)

cat("=== ITERATIVE TOPIC TUNING ===\n")
cat("Quick parameter adjustment and re-analysis\n\n")

# =============================================================================
# 🎛️ PARAMETER ADJUSTMENT SECTION - EDIT THESE VALUES
# =============================================================================

cat("🎛️ CURRENT TUNING PARAMETERS\n")
cat("============================\n")

# EDIT THESE BASED ON QUALITY ASSESSMENT RECOMMENDATIONS:

# Topic count range
k_values <- 2:3              # Back to 2-3 topics - best perplexity scores + appropriate for small dataset
                            # Let's work with what the data wants

# Vocabulary size options  
max_tokens_values <- c(12, 15, 18)   # Reduce vocabulary - smaller vocabularies for cleaner topics
                                    # Was 15-25, now trying 12-18

# Frequency threshold options
min_freq_values <- c(5, 6)          # Increase frequency threshold to filter rare terms
                                   # Was 4-5, now trying 5-6

# Additional stopwords (FOCUSED - only the worst offenders)
additional_stopwords <- c(
  # MAJOR OVERLAPPING TERMS - these appear in every single topic:
  "someth", "help", "interest", # These are the biggest problems
  "some", "something",  # Add these too since they become "someth"
  
  # Very generic discourse terms:
  "also", "lot", "kind", "much", "well", "thing", "way", "make", # Add "make" - too generic
  
  # Conversation fillers that aren't meaningful:
  "yeah", "ok", "yes", "no", "maybe", "like", "realli", "pretti", 
  
  # Over-stemmed terms that aren't interpretable:
  "abl", "littl", # Too stemmed to be useful
  
  # Time/generic descriptors:
  "time", "year", "day", "good", "bad", "big", "small",
  
  # Generic action words that appeared in both topics:
  "mean", "take"  # These were appearing in both topics but not meaningful
  
  # NOTE: Keeping terms like "support", "person", "famili", "school" 
  # These might be meaningful if they cluster differently
)

cat(glue("🔢 Testing k values: {paste(k_values, collapse = ', ')}\n"))
cat(glue("📚 Testing max_tokens: {paste(max_tokens_values, collapse = ', ')}\n"))
cat(glue("🔍 Testing min_freq: {paste(min_freq_values, collapse = ', ')}\n"))
cat(glue("🛑 Additional stopwords: {length(additional_stopwords)} terms\n"))

# =============================================================================
# Load data and setup
# =============================================================================

# Load the substantive data 
if (!file.exists(here("data", "focus_group_substantive.csv"))) {
  stop("❌ focus_group_substantive.csv not found. Run study2_data_preparation.R first.")
}

substantive_data <- read_csv(here("data", "focus_group_substantive.csv"), 
                            show_col_types = FALSE)

cat(glue("\n✅ Loaded {nrow(substantive_data)} substantive utterances\n"))

# =============================================================================
# Enhanced stopword list
# =============================================================================

# Base comprehensive stopwords
sud_terms <- c(
  "substance", "substances", "substanc", "addiction", "addict", "addicted", 
  "drug", "drugs", "alcohol", "alcoholic", "abuse", "abusing",
  "counselor", "counselors", "counseling", "therapy", "therapist", "therapists",
  "treatment", "recover", "recovery", "rehab", "rehabilitation",
  "mental", "health", "psychology", "psychologist", "psychiatric", "psychiatrist",
  "social", "worker", "clinical", "clinician",
  "career", "careers", "job", "jobs", "work", "working", "field", "fields",
  "profession", "professional", "area", "areas"
)

conversation_terms <- c(
  "um", "uh", "like", "know", "yeah", "okay", "right", "guess", "maybe",
  "actually", "probably", "definitely", "obviously", "basically", "literally",
  "kinda", "sorta", "gonna", "wanna", "pretty", "really", "just",
  "think", "thought", "feel", "feeling", "say", "said", "talk", "talking",
  "tell", "telling", "see", "look", "go", "going", "get", "getting",
  "come", "coming", "want", "wanted", "need", "needed",
  "good", "bad", "better", "worse", "best", "different", "similar",
  "thing", "things", "stuff", "way", "ways", "time", "times",
  "people", "person", "someone", "everybody", "everyone",
  "can", "cant", "could", "couldnt", "would", "wouldnt", "should", "shouldnt",
  "will", "wont", "might", "must", "got", "don", "didn", "wasn", "couldn",
  "wouldn", "isn", "aren", "haven", "hasn"
)

# Combine with additional stopwords
comprehensive_stopwords <- bind_rows(
  get_stopwords("en"),
  tibble(word = sud_terms, lexicon = "sud_specific"),
  tibble(word = conversation_terms, lexicon = "focus_group"),
  tibble(word = additional_stopwords, lexicon = "iterative_custom")
) %>%
  distinct(word, .keep_all = TRUE)

cat(glue("🛑 Total stopwords: {nrow(comprehensive_stopwords)} terms\n"))

# =============================================================================
# Create tuning grid
# =============================================================================

tune_grid <- expand_grid(
  k = k_values,
  max_tokens = max_tokens_values,
  min_freq = min_freq_values
)

cat(glue("\n🔄 Testing {nrow(tune_grid)} parameter combinations\n"))
print(tune_grid)

# =============================================================================
# Simplified analysis for quick iteration
# =============================================================================

cat("\n🚀 Running quick analysis...\n")

# For this iteration, let's use a simpler approach that works reliably
# We'll create a basic LDA analysis without the complex workflow

cat("🔧 Using simplified LDA approach for quick iteration...\n")

# Process text data manually for LDA
processed_data <- substantive_data %>%
  select(response_id, cleaned_text) %>%
  unnest_tokens(word, cleaned_text) %>%
  anti_join(comprehensive_stopwords, by = "word") %>%
  filter(!str_detect(word, "^\\d+$"), nchar(word) > 2) %>%
  mutate(word = SnowballC::wordStem(word)) %>%
  filter(nchar(word) >= 3)  # Back to 3 characters - 4 was too restrictive

# Test each parameter combination manually
results_list <- list()

for (i in 1:nrow(tune_grid)) {
  params <- tune_grid[i, ]
  cat(glue("Testing: k={params$k}, max_tokens={params$max_tokens}, min_freq={params$min_freq}\n"))
  
  # Apply current parameters
  word_counts <- processed_data %>%
    count(response_id, word, sort = TRUE) %>%
    group_by(word) %>%
    filter(n() >= params$min_freq) %>%
    ungroup() %>%
    group_by(word) %>%
    summarise(total_freq = sum(n), .groups = "drop") %>%
    slice_max(total_freq, n = params$max_tokens) %>%
    pull(word)
  
  # Filter to selected vocabulary
  doc_word_counts <- processed_data %>%
    filter(word %in% word_counts) %>%
    count(response_id, word, sort = TRUE)
  
  if (nrow(doc_word_counts) < 50) {
    cat("  ⚠️ Too few terms, skipping...\n")
    next
  }
  
  # Create DTM
  dtm <- doc_word_counts %>%
    cast_dtm(response_id, word, n)
  
  if (nrow(dtm) < 10) {
    cat("  ⚠️ Too few documents, skipping...\n")
    next
  }
  
  # Run LDA
  lda_model <- LDA(dtm, k = params$k, control = list(seed = 1234))
  
  # Extract results
  topic_terms <- tidy(lda_model, matrix = "beta")
  perplexity_score <- perplexity(lda_model)
  
  results_list[[i]] <- list(
    params = params,
    perplexity = perplexity_score,
    topic_terms = topic_terms,
    model = lda_model
  )
  
  cat(glue("  ✓ Perplexity: {round(perplexity_score, 1)}\n"))
}

# Find best result
perplexities <- map_dbl(results_list, ~.x$perplexity)
best_idx <- which.min(perplexities)
best_result <- results_list[[best_idx]]
best_params <- best_result$params

# =============================================================================
# Quick results preview
# =============================================================================

cat("\n📊 TUNING RESULTS PREVIEW\n")
cat("=========================\n")

if (length(results_list) == 0) {
  cat("❌ No valid results - try reducing parameter constraints\n")
  stop("No valid parameter combinations produced results")
}

# Show all results
tuning_summary <- map_dfr(results_list, function(result) {
  tibble(
    k = result$params$k,
    max_tokens = result$params$max_tokens,
    min_freq = result$params$min_freq,
    perplexity = result$perplexity
  )
}) %>%
  arrange(perplexity) %>%
  mutate(rank = row_number())

cat("All parameter combinations (ranked by perplexity):\n")
print(tuning_summary %>% 
      mutate(perplexity = round(perplexity, 1)))

cat(glue("\n🏆 Best parameters: k={best_params$k}, max_tokens={best_params$max_tokens}, min_freq={best_params$min_freq}\n"))

# =============================================================================
# Extract topics for quick review
# =============================================================================

cat("\n🔍 QUICK TOPIC PREVIEW\n")
cat("=====================\n")

# Extract topics from best model
topic_terms <- best_result$topic_terms

# Show top terms for each topic
for (topic_num in unique(topic_terms$topic)) {
  cat(glue("\n📋 TOPIC {topic_num} - Top 8 terms:\n"))
  
  top_terms <- topic_terms %>%
    filter(topic == topic_num) %>%
    slice_max(beta, n = 8) %>%
    mutate(prob_pct = round(beta * 100, 1))
  
  for (i in 1:nrow(top_terms)) {
    term_info <- top_terms[i, ]
    cat(glue("   {i}. {term_info$term} ({term_info$prob_pct}%)\n"))
  }
}

# =============================================================================
# Quick quality check
# =============================================================================

cat("\n⚡ QUICK QUALITY CHECK\n")
cat("=====================\n")

# Check for overlap
if (length(unique(topic_terms$topic)) > 1) {
  topic_top_terms <- topic_terms %>%
    group_by(topic) %>%
    slice_max(beta, n = 8) %>%
    group_split()
  
  # Simple overlap check for top terms
  overlaps <- character()
  topics <- unique(topic_terms$topic)
  
  for (i in 1:(length(topics)-1)) {
    for (j in (i+1):length(topics)) {
      terms1 <- topic_top_terms[[i]]$term
      terms2 <- topic_top_terms[[j]]$term
      shared <- intersect(terms1, terms2)
      if (length(shared) > 0) {
        overlaps <- c(overlaps, glue("Topics {topics[i]} & {topics[j]}: {paste(shared, collapse = ', ')}"))
      }
    }
  }
  
  if (length(overlaps) > 0) {
    cat("🟡 Term overlaps detected:\n")
    for (overlap in overlaps) {
      cat(glue("   {overlap}\n"))
    }
  } else {
    cat("✅ No obvious term overlaps detected\n")
  }
} else {
  cat("ℹ️ Only one topic - no overlap to check\n")
}

# Check for generic terms
generic_terms <- c("feel", "think", "know", "say", "go", "get", "want", "need", 
                  "good", "bad", "way", "thing", "people", "person", "realli", "pretti")

generic_count <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 8) %>%
  summarize(generic_in_top = sum(term %in% generic_terms), .groups = "drop")

for (i in 1:nrow(generic_count)) {
  topic_generic <- generic_count[i, ]
  if (topic_generic$generic_in_top > 3) {
    cat(glue("🔴 Topic {topic_generic$topic}: {topic_generic$generic_in_top}/8 generic terms - needs more filtering\n"))
  } else if (topic_generic$generic_in_top > 1) {
    cat(glue("🟡 Topic {topic_generic$topic}: {topic_generic$generic_in_top}/8 generic terms - acceptable\n"))
  } else {
    cat(glue("✅ Topic {topic_generic$topic}: {topic_generic$generic_in_top}/8 generic terms - good specificity\n"))
  }
}

# =============================================================================
# Save results and next steps
# =============================================================================

results_dir <- here("results", "r", "study2_iterative")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

# Save quick results
write_csv(topic_terms, file.path(results_dir, "quick_topic_terms.csv"))
write_csv(tuning_summary, file.path(results_dir, "quick_tuning_summary.csv"))
write_csv(best_params, file.path(results_dir, "quick_best_params.csv"))

cat("\n💾 Quick results saved to results/r/study2_iterative/\n")

# =============================================================================
# Decision point
# =============================================================================

cat("\n🤔 DECISION POINT\n")
cat("=================\n")
cat("Review the topic previews above. Do they look:\n")
cat("✅ GOOD: Distinct, interpretable, specific terms → Proceed with full analysis\n")
cat("🟡 OKAY: Some issues but promising → Minor parameter adjustments\n") 
cat("🔴 BAD: Generic, overlapping, unclear → Major parameter adjustments needed\n")

cat("\n📝 TO ADJUST PARAMETERS:\n")
cat("1. Edit the parameter values at the top of this script\n")
cat("2. Add problematic terms to additional_stopwords\n")
cat("3. Re-run this script\n")
cat("4. Repeat until topics look good\n")

cat("\n📝 WHEN TOPICS LOOK GOOD:\n")
cat("1. Run study2_tidymodels_analysis.R with the good parameters\n")
cat("2. Run study2_topic_quality_assessment.R for detailed analysis\n")
cat("3. Generate visualizations and manuscript tables\n")

cat(glue("\n🎯 Current best: k={best_params$k}, max_tokens={best_params$max_tokens}, min_freq={best_params$min_freq}\n"))
cat("Copy these values to the main analysis script when ready!\n")