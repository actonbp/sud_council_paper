#!/usr/bin/env Rscript
# 01_score_utterances.R ------------------------------------------------------
# Purpose: tokenise focus-group transcripts, apply certainty/uncertainty
#          lexicon, and output utterance-level scores.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(spacyr)
  library(stringr)
})

# ---------------------------------------------------------------------------
# 1. Load data ---------------------------------------------------------------
# ---------------------------------------------------------------------------

data_dir <- 'data/study2'
files <- list.files(data_dir, pattern = '\\.(csv|CSV)$', full.names = TRUE)
if (length(files) == 0) stop('No focus-group CSV files found in ', data_dir)

read_focus <- function(path) {
  read_csv(path, show_col_types = FALSE) %>%
    mutate(session = basename(path))
}

df_raw <- map_dfr(files, read_focus)

# Remove moderator lines (2–3 letter speaker codes, e.g., LR) ----------------
df <- df_raw %>%
  mutate(Speaker = as.character(Speaker)) %>%
  filter(!str_detect(Speaker, '^[A-Z]{2,3}$')) %>%
  filter(!is.na(Text))

message('Loaded ', nrow(df), ' utterances (participants only).')

# ---------------------------------------------------------------------------
# 2. Initialise spaCy --------------------------------------------------------
# ---------------------------------------------------------------------------

# Flag for spacy availability
use_spacy <- FALSE

# Try initialise spaCy; if fails, keep FALSE
tryCatch({
  spacy_initialize(model = "en_core_web_sm", refresh_settings = FALSE)
  use_spacy <- TRUE
}, error = function(e) {
  message('spaCy not available; falling back to token = "words"')
})

# Choose tokenizer
tok_fun <- if (use_spacy) "spacyr_lemmatize" else "words"

# ---------------------------------------------------------------------------
# 3A. Define stop-word/filler/domain lists ----------------------------------
# ---------------------------------------------------------------------------

data("stop_words", package = "tidytext")

filler_words  <- c("uh", "um", "like", "yeah", "you", "know", "kinda", "sorta", "okay", "right")
domain_terms  <- c("counselor", "counselors", "counseling", "therapist", "therapy",
                   "mental", "health", "substance", "addiction", "field")

custom_stop   <- unique(c(stop_words$word, filler_words, domain_terms))

# helper to clean tokens -----------------------------------------------------
clean_tok <- function(x) {
  x <- str_replace_all(x, "[0-9]+", "")           # remove numbers
  x <- str_replace_all(x, "[^a-z'\\-]", "")      # keep letters / apostrophe / hyphen
  str_trim(x)
}

# ---------------------------------------------------------------------------
# 3. Tokenise & lemmatise ----------------------------------------------------
# ---------------------------------------------------------------------------

if (use_spacy) {
  # Use spaCy to get lemmas directly
  parsed <- spacy_parse(df$Text, lemma = TRUE, pos = FALSE, entity = FALSE)
  tidy_tokens <- parsed %>%
    mutate(doc_id = as.integer(str_remove(doc_id, "text"))) %>%
    mutate(session = df$session[doc_id],
           Speaker = df$Speaker[doc_id],
           Text    = df$Text[doc_id]) %>%
    rename(lemma = lemma) %>%
    select(session, Speaker, Text, lemma) %>%
    mutate(lemma = clean_tok(lemma)) %>%
    filter(
      nchar(lemma) >= 3,
      lemma != "",
      !lemma %in% custom_stop,
      str_detect(lemma, "^[a-z][a-z'\\-]+$")
    )
} else {
  tidy_tokens <- df %>%
    select(session, Speaker, Text) %>%
    unnest_tokens(output = "lemma", input = Text, token = tok_fun,
                  to_lower = TRUE, drop = FALSE) %>%
    mutate(lemma = clean_tok(lemma)) %>%
    filter(
      nchar(lemma) >= 3,
      lemma != "",
      !lemma %in% custom_stop,
      str_detect(lemma, "^[a-z][a-z'\\-]+$")
    )
}

# ---------------------------------------------------------------------------
# 4. Apply lexicon -----------------------------------------------------------
# ---------------------------------------------------------------------------

lexicon <- read_rds('results/study2/expanded_lexicon.rds')

scored <- tidy_tokens %>%
  inner_join(lexicon, by = c('lemma' = 'term')) %>%
  count(session, Speaker, Text, category) %>%
  pivot_wider(names_from = category, values_from = n, values_fill = 0) %>%
  mutate(total_hits = certain + uncertain,
         certainty_score = ifelse(total_hits == 0, 0,
                                  (certain - uncertain) / total_hits))

# Keep all utterances (even zero-hit ones) -----------------------------------

all_utts <- df %>%
  select(session, Speaker, Text) %>%
  distinct() %>%
  left_join(scored, by = c('session', 'Speaker', 'Text')) %>%
  mutate(across(c(certain, uncertain, total_hits, certainty_score),
                ~replace_na(., 0)))

# ---------------------------------------------------------------------------
# 5. Save outputs ------------------------------------------------------------
# ---------------------------------------------------------------------------

dir.create('results/study2', showWarnings = FALSE)

write_csv(all_utts, 'results/study2/dictionary_scores.csv')

summary_tab <- all_utts %>%
  summarise(
    utterances      = n(),
    with_hits       = sum(total_hits > 0),
    pct_with_hits   = round(100 * with_hits / utterances, 1),
    mean_score      = mean(certainty_score),
    sd_score        = sd(certainty_score)
  )
write_csv(summary_tab, 'results/study2/dictionary_summary.csv')

message('✅ Scoring complete.  Output written to results/study2/')

# ---------------------------------------------------------------------------
# 6. Clean up ----------------------------------------------------------------
# ---------------------------------------------------------------------------
if (use_spacy) try(spacy_finalize(), silent = TRUE) 