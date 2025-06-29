#!/usr/bin/env Rscript
# 01_score_utterances.R ------------------------------------------------------
# Purpose: tokenise focus-group transcripts, apply certainty/uncertainty
#          lexicon, and output utterance-level scores.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(spacyr)
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
tryCatch({
  spacy_initialize(model = "en_core_web_sm", refresh_settings = FALSE)
}, error = function(e) {
  message('spaCy not available; falling back to tidy tokenisation')
})

# ---------------------------------------------------------------------------
# 3. Tokenise & lemmatise ----------------------------------------------------
# ---------------------------------------------------------------------------

tidy_tokens <- df %>%
  select(session, Speaker, Text) %>%
  unnest_tokens(output = "lemma", input = Text, token = "spacyr_lemmatize",
                to_lower = TRUE, drop = FALSE) %>%
  filter(str_detect(lemma, '^[a-z][a-z\'\-]+$'))

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
try(spacy_finalize(), silent = TRUE) 