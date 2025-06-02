# Proper Text Preprocessing Following smltar/tidytext Principles
# Implements comprehensive preprocessing for focus group analysis

library(tidyverse)
library(tidytext)
library(SnowballC)
library(here)

# Install stopwords if not available, otherwise use tidytext stopwords
if(require(stopwords, quietly = TRUE)) {
  stopwords_available <- TRUE
} else {
  stopwords_available <- FALSE
  cat("Note: stopwords package not available, using tidytext stop_words only\n")
}

cat("=== COMPREHENSIVE TEXT PREPROCESSING (smltar/tidytext) ===\n")

# Load substantive data
substantive_data <- read_csv(here("data", "focus_group_substantive.csv"), show_col_types = FALSE)

cat("Original data:", nrow(substantive_data), "utterances\n")

# STEP 1: Comprehensive Tokenization
cat("\n🔤 STEP 1: Tokenization\n")

# Tokenize with tidytext unnest_tokens() - follows smltar principles
tokens_raw <- substantive_data %>%
  select(response_id, session_id, Speaker, cleaned_text) %>%
  unnest_tokens(word, cleaned_text, 
                token = "words",           # Word-level tokenization
                strip_punct = TRUE,       # Remove punctuation
                strip_numeric = FALSE)    # Keep numbers for now

cat("Raw tokens:", nrow(tokens_raw), "total words\n")

# STEP 2: Multi-source Stopword Removal  
cat("\n🛑 STEP 2: Stopword Removal\n")

# Combine multiple stopword sources for robustness (smltar best practice)
if(stopwords_available) {
  stop_words_comprehensive <- bind_rows(
    tibble(word = stopwords("en", source = "snowball"), lexicon = "snowball"),
    tibble(word = stopwords("en", source = "stopwords-iso"), lexicon = "iso"),
    stop_words,
    
    # Custom focus group stopwords
    tibble(word = c("um", "uh", "like", "know", "yeah", "okay", "right", 
                   "maybe", "actually", "probably", "guess", "kinda", "sorta",
                   "gonna", "wanna", "basically", "literally", "obviously"),
           lexicon = "focus_group")
  ) %>%
    distinct(word, .keep_all = TRUE)
} else {
  # Use tidytext stopwords only
  stop_words_comprehensive <- bind_rows(
    stop_words,
    
    # Custom focus group stopwords
    tibble(word = c("um", "uh", "like", "know", "yeah", "okay", "right", 
                   "maybe", "actually", "probably", "guess", "kinda", "sorta",
                   "gonna", "wanna", "basically", "literally", "obviously"),
           lexicon = "focus_group")
  ) %>%
    distinct(word, .keep_all = TRUE)
}

cat("Comprehensive stopwords:", nrow(stop_words_comprehensive), "words from", 
    length(unique(stop_words_comprehensive$lexicon)), "sources\n")

# Remove stopwords
tokens_no_stop <- tokens_raw %>%
  anti_join(stop_words_comprehensive, by = "word") %>%
  filter(str_length(word) >= 3,               # Remove very short words
         !str_detect(word, "^\\d+$"),          # Remove pure numbers
         str_detect(word, "^[a-zA-Z]"))        # Keep only alphabetic words

cat("After stopword removal:", nrow(tokens_no_stop), "meaningful tokens\n")

# STEP 3: Stemming using SnowballC (Porter stemmer)
cat("\n🌱 STEP 3: Stemming\n")

tokens_stemmed <- tokens_no_stop %>%
  mutate(
    word_original = word,
    word_stem = wordStem(word, language = "en")  # Porter stemming
  ) %>%
  filter(str_length(word_stem) >= 2)  # Remove very short stems

cat("Unique original words:", length(unique(tokens_no_stop$word)), "\n")
cat("Unique stems:", length(unique(tokens_stemmed$word_stem)), "\n")
cat("Stemming reduction:", round((1 - length(unique(tokens_stemmed$word_stem))/length(unique(tokens_no_stop$word)))*100, 1), "%\n")

# STEP 4: Create SUD Term Stems for Detection
cat("\n🎯 STEP 4: SUD Term Preprocessing\n")

# Comprehensive SUD terminology (from previous analysis)
sud_terms_raw <- c(
  # Core addiction terms
  "substance", "addiction", "addict", "addicted", "addictive", "dependence", "dependent", "dependency",
  
  # Specific substances  
  "alcohol", "alcoholism", "alcoholic", "drug", "drugs", "cocaine", "heroin", "opioid", "opiate",
  "marijuana", "cannabis", "methamphetamine", "prescription",
  
  # Treatment/recovery
  "recovery", "recovering", "rehabilitation", "rehab", "detox", "detoxification", "treatment",
  "therapy", "counseling", "intervention", "sobriety", "sober", "clean", "abstinence",
  "relapse", "methadone", "suboxone",
  
  # Problem framing
  "abuse", "abusing", "struggle", "struggling", "battle", "fighting", "overcome", "overcoming",
  
  # Professional context
  "counselor", "therapist", "specialist", "program", "center", "services", "clinical"
)

# Apply same preprocessing to SUD terms
sud_terms_stemmed <- tibble(term = sud_terms_raw) %>%
  mutate(
    term_clean = str_to_lower(term),
    term_stem = wordStem(term_clean, language = "en")
  ) %>%
  pull(term_stem) %>%
  unique()

cat("SUD terms - Original:", length(sud_terms_raw), "Stems:", length(sud_terms_stemmed), "\n")

# STEP 5: Utterance-Level SUD Detection with Proper Preprocessing
cat("\n🔍 STEP 5: Utterance-Level SUD Detection\n")

# Reconstruct utterances with stems for detection
utterance_stems <- tokens_stemmed %>%
  group_by(response_id, session_id, Speaker) %>%
  summarise(
    stems_combined = paste(word_stem, collapse = " "),
    original_words = paste(word_original, collapse = " "),
    n_tokens = n(),
    .groups = "drop"
  )

# SUD detection using stemmed terms
utterance_stems <- utterance_stems %>%
  mutate(
    mentions_sud_stems = map_lgl(stems_combined, ~{
      any(str_detect(.x, paste0("\\b", sud_terms_stemmed, "\\b", collapse = "|")))
    })
  )

# Join back with original data
comprehensive_results <- substantive_data %>%
  left_join(utterance_stems, by = "response_id") %>%
  replace_na(list(mentions_sud_stems = FALSE))

sud_count <- sum(comprehensive_results$mentions_sud_stems)
sud_pct <- round(sud_count / nrow(comprehensive_results) * 100, 1)

cat("SUD utterances detected:", sud_count, "(", sud_pct, "% of substantive content)\n")

# STEP 6: Save Comprehensive Preprocessing Results  
cat("\n💾 STEP 6: Save Results\n")

# Save preprocessed tokens
write_csv(tokens_stemmed, here("data", "focus_group_tokens_preprocessed.csv"))

# Save utterance-level results
write_csv(comprehensive_results, here("data", "focus_group_comprehensive_preprocessed.csv"))

# Save analysis metadata
preprocessing_metadata <- list(
  original_utterances = nrow(substantive_data),
  raw_tokens = nrow(tokens_raw),
  tokens_after_stopwords = nrow(tokens_no_stop),
  final_tokens = nrow(tokens_stemmed),
  unique_original_words = length(unique(tokens_no_stop$word)),
  unique_stems = length(unique(tokens_stemmed$word_stem)),
  stemming_reduction_pct = round((1 - length(unique(tokens_stemmed$word_stem))/length(unique(tokens_no_stop$word)))*100, 1),
  sud_terms_original = length(sud_terms_raw),
  sud_terms_stemmed = length(sud_terms_stemmed),
  sud_utterances_detected = sud_count,
  sud_detection_percentage = sud_pct,
  stopword_sources = unique(stop_words_comprehensive$lexicon)
)

saveRDS(preprocessing_metadata, here("data", "preprocessing_metadata.rds"))

cat("\nFiles saved:\n")
cat("- focus_group_tokens_preprocessed.csv (", nrow(tokens_stemmed), "tokens)\n")
cat("- focus_group_comprehensive_preprocessed.csv (", nrow(comprehensive_results), "utterances)\n")
cat("- preprocessing_metadata.rds (analysis metadata)\n")

cat("\n✅ COMPREHENSIVE TEXT PREPROCESSING COMPLETE!\n")
cat("Following smltar/tidytext best practices:\n")
cat("✓ Proper unnest_tokens() word tokenization\n")
cat("✓ Multi-source stopword removal (", length(unique(stop_words_comprehensive$lexicon)), "sources)\n")
cat("✓ Porter stemming with SnowballC\n")
cat("✓ Robust SUD term detection\n")
cat("✓ Preserved utterance-level structure\n")