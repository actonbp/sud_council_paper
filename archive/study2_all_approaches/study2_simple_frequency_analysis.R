# =============================================================================
# Study 2: Simple Frequency Analysis of SUD Career Discussion
# SUD Counseling Career Research Project
# =============================================================================
# Simple, straightforward analysis of most common terms in SUD career discussions
# Perfect for 1 impact factor journal - no need to overcomplicate
# Date: June 12, 2025
# =============================================================================

library(tidyverse)
library(tidytext)
library(here)
library(glue)
library(readr)
library(SnowballC)
library(stringr)

cat("=== SIMPLE FREQUENCY ANALYSIS ===\n")
cat("Straightforward analysis of SUD career discussion themes\n\n")

# =============================================================================
# 1. Load data
# =============================================================================

# Load the substantive data
if (!file.exists(here("data", "focus_group_substantive.csv"))) {
  stop("❌ focus_group_substantive.csv not found. Run study2_data_preparation.R first.")
}

substantive_data <- read_csv(here("data", "focus_group_substantive.csv"), 
                            show_col_types = FALSE)

cat(glue("✅ Loaded {nrow(substantive_data)} substantive utterances\n"))

# =============================================================================
# 2. Create focused stopword list - only remove the obvious ones
# =============================================================================

cat("\n🛑 Creating focused stopword list...\n")

# Only remove the most obvious non-meaningful terms
focused_stopwords <- bind_rows(
  get_stopwords("en"),
  tibble(word = c(
    # Only the most generic conversation terms:
    "um", "uh", "like", "know", "yeah", "okay", "right", "guess", "maybe",
    "actually", "probably", "definitely", "really", "pretty", "just",
    
    # Only the most generic action words:
    "go", "get", "come", "see", "look", "say", "tell", "talk",
    
    # Only clearly over-stemmed terms:
    "abl", "littl", # These are "able" and "little" but uninterpretable
    
    # Basic SUD terms that don't add meaning (we're already filtering FOR these):
    "substance", "substances", "substanc", "addiction", "addict",
    "counselor", "counselors", "counseling", "therapy", "therapist",
    
    # Time and very generic descriptors:
    "time", "year", "day", "thing", "way"
  ), lexicon = "focused_custom")
) %>%
  distinct(word, .keep_all = TRUE)

cat(glue("📝 Focused stopwords: {nrow(focused_stopwords)} terms\n"))

# =============================================================================
# 3. Process text and identify SUD-relevant discussions
# =============================================================================

cat("\n🔍 Processing text and identifying SUD discussions...\n")

# SUD-relevant terms for identifying relevant utterances
sud_terms <- c(
  # Direct SUD terms
  "substance", "addiction", "addict", "drug", "alcohol", "abuse",
  # Treatment/counseling terms  
  "counseling", "therapy", "therapist", "counselor", "treatment", "recovery",
  # Career/field terms
  "career", "job", "work", "field", "profession", "major"
)

# Tokenize and process
tokens <- substantive_data %>%
  select(response_id, cleaned_text) %>%
  unnest_tokens(word, cleaned_text) %>%
  mutate(word_stem = wordStem(word, language = "english")) %>%
  # Keep both original and stemmed for analysis
  mutate(
    word_clean = str_to_lower(word),
    is_sud_term = word_stem %in% wordStem(sud_terms, language = "english")
  )

# Identify SUD-relevant utterances
sud_utterances <- tokens %>%
  group_by(response_id) %>%
  summarize(
    mentions_sud = any(is_sud_term),
    .groups = "drop"
  ) %>%
  filter(mentions_sud) %>%
  pull(response_id)

cat(glue("🎯 Found {length(sud_utterances)} utterances mentioning SUD/career topics\n"))
cat(glue("📊 Detection rate: {round(length(sud_utterances)/nrow(substantive_data)*100, 1)}%\n"))

# =============================================================================
# 4. Analyze word frequencies in SUD discussions
# =============================================================================

cat("\n📊 Analyzing word frequencies in SUD career discussions...\n")

# Filter to SUD utterances and clean
sud_word_freq <- tokens %>%
  filter(response_id %in% sud_utterances) %>%
  # Remove stopwords
  anti_join(focused_stopwords, by = "word") %>%
  # Basic filtering
  filter(
    !str_detect(word, "^\\d+$"),  # Remove numbers
    nchar(word) >= 3,             # Remove very short words
    str_detect(word, "^[a-zA-Z]") # Keep only alphabetic
  ) %>%
  # Use stemmed words for consistency
  count(word_stem, sort = TRUE) %>%
  # Only keep words that appear multiple times
  filter(n >= 3) %>%
  # Get back to readable format by finding most common original word for each stem
  left_join(
    tokens %>%
      filter(response_id %in% sud_utterances) %>%
      count(word_stem, word_clean) %>%
      group_by(word_stem) %>%
      slice_max(n, n = 1, with_ties = FALSE) %>%
      select(word_stem, example_word = word_clean),
    by = "word_stem"
  ) %>%
  # Calculate percentages
  mutate(
    percentage = round(n / sum(n) * 100, 1),
    rank = row_number()
  ) %>%
  select(rank, word_stem, example_word, frequency = n, percentage)

cat(glue("✅ Analyzed {nrow(sud_word_freq)} meaningful terms\n"))

# =============================================================================
# 5. Display results
# =============================================================================

cat("\n📋 TOP TERMS IN SUD CAREER DISCUSSIONS\n")
cat("=====================================\n")

# Show top 20 terms
top_terms <- sud_word_freq %>% slice_head(n = 20)

cat("Rank | Term | Frequency | Percentage\n")
cat("-----|------|-----------|----------\n")

for (i in 1:nrow(top_terms)) {
  term_info <- top_terms[i, ]
  cat(glue("{sprintf('%2d', term_info$rank)} | {sprintf('%-12s', term_info$example_word)} | {sprintf('%6d', term_info$frequency)} | {sprintf('%5.1f', term_info$percentage)}%\n"))
}

# =============================================================================
# 6. Analyze term categories for interpretation
# =============================================================================

cat("\n🎯 THEMATIC ANALYSIS OF TOP TERMS\n")
cat("=================================\n")

# Categorize terms for interpretation
term_categories <- sud_word_freq %>%
  slice_head(n = 30) %>%
  mutate(
    category = case_when(
      example_word %in% c("family", "parent", "parents", "mom", "dad", "brother", "sister") ~ "Family/Personal",
      example_word %in% c("help", "helping", "support", "care", "caring") ~ "Helping Orientation", 
      example_word %in% c("school", "education", "learn", "learning", "study", "student") ~ "Education/Learning",
      example_word %in% c("people", "person", "someone", "individual", "client") ~ "People Focus",
      example_word %in% c("experience", "experiences", "background", "history") ~ "Personal Experience",
      example_word %in% c("interest", "interested", "motivation", "motivated") ~ "Interest/Motivation",
      example_word %in% c("hard", "difficult", "challenge", "challenging", "struggle") ~ "Challenges/Difficulty",
      example_word %in% c("feel", "feeling", "feelings", "emotion", "emotional") ~ "Emotional Processing",
      example_word %in% c("money", "financial", "salary", "pay", "income") ~ "Financial Considerations",
      TRUE ~ "Other"
    )
  )

# Summarize by category
category_summary <- term_categories %>%
  group_by(category) %>%
  summarize(
    term_count = n(),
    total_frequency = sum(frequency),
    total_percentage = sum(percentage),
    example_terms = paste(example_word[1:min(5, n())], collapse = ", "),
    .groups = "drop"
  ) %>%
  arrange(desc(total_percentage))

cat("Theme Categories (Top 30 terms):\n\n")

for (i in 1:nrow(category_summary)) {
  cat_info <- category_summary[i, ]
  cat(glue("**{cat_info$category}** ({cat_info$total_percentage}% of discourse)\n"))
  cat(glue("   Terms: {cat_info$example_terms}\n"))
  cat(glue("   {cat_info$term_count} terms, {cat_info$total_frequency} total mentions\n\n"))
}

# =============================================================================
# 7. Save results
# =============================================================================

cat("\n💾 Saving results...\n")

# Set up results directory
results_dir <- here("results", "r", "study2_simple_frequency")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

# Save detailed word frequencies
write_csv(sud_word_freq, file.path(results_dir, "word_frequencies.csv"))

# Save category analysis  
write_csv(category_summary, file.path(results_dir, "thematic_categories.csv"))

# Save analysis metadata
analysis_metadata <- tibble(
  analysis_date = Sys.Date(),
  total_utterances = nrow(substantive_data),
  sud_utterances = length(sud_utterances),
  detection_rate_pct = round(length(sud_utterances)/nrow(substantive_data)*100, 1),
  unique_terms_analyzed = nrow(sud_word_freq),
  total_word_tokens = sum(sud_word_freq$frequency),
  stopwords_used = nrow(focused_stopwords),
  methodology = "Simple frequency analysis of SUD career discussions"
)

write_csv(analysis_metadata, file.path(results_dir, "analysis_metadata.csv"))

# Create manuscript summary
manuscript_summary <- tibble(
  statistic = c(
    "Total substantive utterances",
    "SUD career-relevant utterances", 
    "Detection rate",
    "Unique meaningful terms",
    "Total word tokens analyzed",
    "Most frequent term",
    "Top category"
  ),
  value = c(
    as.character(nrow(substantive_data)),
    as.character(length(sud_utterances)),
    glue("{round(length(sud_utterances)/nrow(substantive_data)*100, 1)}%"),
    as.character(nrow(sud_word_freq)),
    as.character(sum(sud_word_freq$frequency)),
    glue("{top_terms$example_word[1]} ({top_terms$percentage[1]}%)"),
    glue("{category_summary$category[1]} ({category_summary$total_percentage[1]}%)")
  )
)

write_csv(manuscript_summary, file.path(results_dir, "manuscript_summary.csv"))

cat("📁 Results saved to:", results_dir, "\n")
cat("Files created:\n")
cat("   - word_frequencies.csv (detailed term analysis)\n")
cat("   - thematic_categories.csv (theme groupings)\n") 
cat("   - analysis_metadata.csv (analysis specifications)\n")
cat("   - manuscript_summary.csv (key statistics for paper)\n")

# =============================================================================
# 8. Final summary
# =============================================================================

cat(glue("\n🏁 SIMPLE FREQUENCY ANALYSIS COMPLETE!\n"))
cat("=====================================\n")
cat(glue("✅ Analyzed {length(sud_utterances)} SUD career discussions ({round(length(sud_utterances)/nrow(substantive_data)*100, 1)}% of utterances)\n"))
cat(glue("✅ Identified {nrow(sud_word_freq)} meaningful terms\n"))
cat(glue("✅ Top theme: {category_summary$category[1]} ({category_summary$total_percentage[1]}% of discourse)\n"))
cat(glue("✅ Most frequent term: {top_terms$example_word[1]} ({top_terms$percentage[1]}%)\n"))

cat("\n🎯 KEY INSIGHT FOR MANUSCRIPT:\n")
cat("SUD counseling career discussions center on themes of ")
cat(tolower(category_summary$category[1]))
if (nrow(category_summary) > 1) {
  cat(glue(", {tolower(category_summary$category[2])}"))
  if (nrow(category_summary) > 2) {
    cat(glue(", and {tolower(category_summary$category[3])}"))
  }
}
cat(".\n")

cat("\n📝 MANUSCRIPT INTEGRATION:\n")
cat("Replace complex topic modeling section with:\n")
cat("- Simple frequency analysis methodology\n") 
cat("- Clear thematic categories based on most frequent terms\n")
cat("- Straightforward interpretation aligned with Study 1 findings\n")

cat("\n✅ Ready for manuscript integration - much simpler and cleaner approach!\n")