
# Proper Co-occurrence Analysis Following smltar/tidytext Principles
# Uses comprehensively preprocessed tokens for robust topic analysis

library(tidyverse)
library(tidytext)
library(here)

cat("=== PROPER SUD TOPIC CO-OCCURRENCE ANALYSIS ===\n")
cat("Using smltar/tidytext preprocessed data\n\n")

# Load preprocessed data
preprocessed_tokens <- read_csv(here("data", "focus_group_tokens_preprocessed.csv"), show_col_types = FALSE)
preprocessed_utterances <- read_csv(here("data", "focus_group_comprehensive_preprocessed.csv"), show_col_types = FALSE)
preprocessing_metadata <- readRDS(here("data", "preprocessing_metadata.rds"))

cat("📊 PREPROCESSING SUMMARY:\n")
cat("- Original utterances:", preprocessing_metadata$original_utterances, "\n")
cat("- Final meaningful tokens:", preprocessing_metadata$final_tokens, "\n")
cat("- Stemming reduction:", preprocessing_metadata$stemming_reduction_pct, "%\n")
cat("- SUD utterances detected:", preprocessing_metadata$sud_utterances_detected, 
    "(", preprocessing_metadata$sud_detection_percentage, "%)\n\n")

# Extract SUD-related utterances using proper detection
sud_utterances <- preprocessed_utterances %>%
  filter(mentions_sud_stems == TRUE) %>%
  select(response_id, session_id.x, Speaker.x, cleaned_text)

cat("🎯 SUD UTTERANCE ANALYSIS:\n")
cat("Analyzing", nrow(sud_utterances), "SUD-related utterances\n")
cat("Represents", round(nrow(sud_utterances)/nrow(preprocessed_utterances)*100, 1), 
    "% of substantive content\n\n")

# Get properly preprocessed tokens for SUD utterances
sud_tokens <- preprocessed_tokens %>%
  filter(response_id %in% sud_utterances$response_id) %>%
  select(response_id, word_original, word_stem)

cat("📝 TOKEN ANALYSIS:\n")
cat("SUD-related tokens:", nrow(sud_tokens), "\n")
cat("Unique original words:", length(unique(sud_tokens$word_original)), "\n")
cat("Unique stems:", length(unique(sud_tokens$word_stem)), "\n\n")

# ANALYSIS 1: Stem-based word frequencies (most robust)
cat("🔍 ANALYSIS 1: Stem-based Word Frequencies\n")

stem_freq <- sud_tokens %>%
  count(word_stem, sort = TRUE) %>%
  filter(n >= 2)  # Stems appearing at least twice

cat("Top stems co-occurring with SUD discussions:\n")
print(head(stem_freq, 20))

# ANALYSIS 2: Original word frequencies (for interpretability)
cat("\n📖 ANALYSIS 2: Original Word Frequencies\n")

# Get most common original word for each frequent stem
original_word_freq <- sud_tokens %>%
  filter(word_stem %in% stem_freq$word_stem) %>%
  count(word_stem, word_original, sort = TRUE) %>%
  group_by(word_stem) %>%
  slice_max(n, n = 1, with_ties = FALSE) %>%  # Most common original word per stem
  ungroup() %>%
  left_join(stem_freq, by = "word_stem", suffix = c("_original", "_stem")) %>%
  select(word_stem, word_original, n_stem) %>%
  arrange(desc(n_stem))

cat("Top original words (by stem frequency):\n")
print(head(original_word_freq, 20))

# ANALYSIS 3: Thematic clustering using stems
cat("\n🎯 ANALYSIS 3: Thematic Clusters (Stem-based)\n")

# Career/professional stems
career_stems <- stem_freq %>%
  filter(str_detect(word_stem, "career|work|job|profession|field|train|educat|school|degre|learn|class|student"))

if(nrow(career_stems) > 0) {
  cat("\n🎯 CAREER/PROFESSIONAL stems:\n")
  career_words <- career_stems %>%
    left_join(original_word_freq, by = "word_stem") %>%
    select(word_stem, word_original, frequency = n) %>%
    arrange(desc(frequency))
  print(career_words)
}

# Personal experience stems
personal_stems <- stem_freq %>%
  filter(str_detect(word_stem, "person|experi|famili|friend|life|stori|feel|emot|own|met|know"))

if(nrow(personal_stems) > 0) {
  cat("\n👤 PERSONAL EXPERIENCE stems:\n")
  personal_words <- personal_stems %>%
    left_join(original_word_freq, by = "word_stem") %>%
    select(word_stem, word_original, frequency = n) %>%
    arrange(desc(frequency))
  print(personal_words)
}

# Helping/service stems
helping_stems <- stem_freq %>%
  filter(str_detect(word_stem, "help|support|care|serv|assist|counsel|treat|recover|client|patient"))

if(nrow(helping_stems) > 0) {
  cat("\n🤝 HELPING/SERVICE stems:\n")
  helping_words <- helping_stems %>%
    left_join(original_word_freq, by = "word_stem") %>%
    select(word_stem, word_original, frequency = n) %>%
    arrange(desc(frequency))
  print(helping_words)
}

# Challenge/barrier stems
challenge_stems <- stem_freq %>%
  filter(str_detect(word_stem, "difficult|hard|challeng|problem|issu|struggl|tough|barrier|stigma|scar"))

if(nrow(challenge_stems) > 0) {
  cat("\n⚠️ CHALLENGE/BARRIER stems:\n")
  challenge_words <- challenge_stems %>%
    left_join(original_word_freq, by = "word_stem") %>%
    select(word_stem, word_original, frequency = n) %>%
    arrange(desc(frequency))
  print(challenge_words)
}

# Interest/motivation stems
interest_stems <- stem_freq %>%
  filter(str_detect(word_stem, "interest|want|like|enjoy|passion|motivat|excit|appeal|drawn|attract|compassion"))

if(nrow(interest_stems) > 0) {
  cat("\n💡 INTEREST/MOTIVATION stems:\n")
  interest_words <- interest_stems %>%
    left_join(original_word_freq, by = "word_stem") %>%
    select(word_stem, word_original, frequency = n) %>%
    arrange(desc(frequency))
  print(interest_words)
}

# People/relationships stems
people_stems <- stem_freq %>%
  filter(str_detect(word_stem, "peopl|person|famili|friend|mom|dad|parent|child|kid|brother|sister|client|patient"))

if(nrow(people_stems) > 0) {
  cat("\n👥 PEOPLE/RELATIONSHIPS stems:\n")
  people_words <- people_stems %>%
    left_join(original_word_freq, by = "word_stem") %>%
    select(word_stem, word_original, frequency = n) %>%
    arrange(desc(frequency))
  print(people_words)
}

# ANALYSIS 4: Session-level patterns
cat("\n📊 ANALYSIS 4: Session-level Patterns\n")

# Join with utterances to get session info
sud_tokens_with_session <- sud_tokens %>%
  left_join(sud_utterances %>% select(response_id, session_id.x), by = "response_id")

session_patterns <- sud_tokens_with_session %>%
  group_by(session_id.x) %>%
  summarise(
    n_tokens = n(),
    n_unique_stems = n_distinct(word_stem),
    top_stem = names(sort(table(word_stem), decreasing = TRUE))[1],
    .groups = "drop"
  ) %>%
  arrange(desc(n_tokens))

cat("SUD discussion patterns by session:\n")
print(session_patterns)

# ANALYSIS 5: Comprehensive thematic summary
cat("\n📋 ANALYSIS 5: Comprehensive Thematic Summary\n")

theme_summary <- tibble(
  theme = c("Career/Professional", "Personal Experience", "Helping/Service", 
           "Challenge/Barrier", "Interest/Motivation", "People/Relationships"),
  unique_stems = c(nrow(career_stems), nrow(personal_stems), nrow(helping_stems),
                  nrow(challenge_stems), nrow(interest_stems), nrow(people_stems)),
  total_mentions = c(
    if(nrow(career_stems) > 0) sum(career_stems$n) else 0,
    if(nrow(personal_stems) > 0) sum(personal_stems$n) else 0,
    if(nrow(helping_stems) > 0) sum(helping_stems$n) else 0,
    if(nrow(challenge_stems) > 0) sum(challenge_stems$n) else 0,
    if(nrow(interest_stems) > 0) sum(interest_stems$n) else 0,
    if(nrow(people_stems) > 0) sum(people_stems$n) else 0
  )
) %>%
  mutate(
    percentage_of_sud_tokens = round(total_mentions / nrow(sud_tokens) * 100, 1)
  ) %>%
  arrange(desc(total_mentions))

cat("Theme prevalence in SUD discussions:\n")
print(theme_summary)

# Save comprehensive analysis results
proper_cooccurrence_analysis <- list(
  preprocessing_summary = preprocessing_metadata,
  sud_utterances_count = nrow(sud_utterances),
  sud_tokens_count = nrow(sud_tokens),
  stem_frequencies = stem_freq,
  original_word_frequencies = original_word_freq,
  thematic_clusters = list(
    career = if(exists("career_words")) career_words else tibble(),
    personal = if(exists("personal_words")) personal_words else tibble(),
    helping = if(exists("helping_words")) helping_words else tibble(),
    challenges = if(exists("challenge_words")) challenge_words else tibble(),
    interest = if(exists("interest_words")) interest_words else tibble(),
    people = if(exists("people_words")) people_words else tibble()
  ),
  session_patterns = session_patterns,
  theme_summary = theme_summary
)

saveRDS(proper_cooccurrence_analysis, here("results", "proper_cooccurrence_analysis.rds"))

cat("\n💾 ANALYSIS SAVED:\n")
cat("File: results/proper_cooccurrence_analysis.rds\n\n")

cat("✅ PROPER CO-OCCURRENCE ANALYSIS COMPLETE!\n")
cat("Following smltar/tidytext best practices:\n")
cat("✓ Used Porter-stemmed tokens for robust analysis\n")
cat("✓ Proper preprocessing pipeline (tokenization → stopwords → stemming)\n")
cat("✓ Conservative SUD detection (", preprocessing_metadata$sud_detection_percentage, "% of content)\n")
cat("✓ Stem-based thematic clustering\n")
cat("✓ Preserved original words for interpretability\n")
cat("✓ Session-level pattern analysis\n")