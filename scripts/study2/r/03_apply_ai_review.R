#!/usr/bin/env Rscript
# 03_apply_ai_review.R -------------------------------------------------------
# Purpose: Apply AI manual review decisions to lexicon
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
})

lex <- read_csv('results/study2/expanded_lexicon_review.csv', show_col_types = FALSE)

# Terms to remove (AI manual review) -----------------------------------------
remove_terms <- c(
  # Misspellings & incorrect forms
  "proceded", "deciced", "choosed", "agreeed", "certaint", "definate", 
  "definitve", "defintive", "definative", "determinded", "detemined",
  "resovled", "cleare", "choosen", "chosed", "selcted", "commmitted",
  "comitted", "amybe", "prehaps", "probabably", "probablly", "proably",
  "proabably", "possiblly", "posibly", "posssibly", "possibily", "possiby",
  "possibley", "hesistant", "dubt", "guesss", "quess", "feare", "uncertainity",
  
  # Awkward forms
  "decideds", "sures", "definites", "determinated", "determinates",
  "determinating", "determinator", "undecideds", "guessers",
  
  # Non-standard
  "forsure",
  
  # Wrong meaning
  "detered",  # means discouraged, not determined
  "undoubtedly", "undoubtably",  # actually mean certain!
  "unworried", "unconcerned", "undoubtable",  # actually mean certain!
  
  # Off-topic
  "nervous-system", "neuro-muscular"
)

# Override specific keeps ----------------------------------------------------
keep_terms <- c("Perhaps")  # Valid uncertainty term

# Apply decisions ------------------------------------------------------------
lex <- lex %>%
  mutate(
    eval = case_when(
      term %in% remove_terms ~ "remove",
      term %in% keep_terms ~ "keep",
      TRUE ~ eval
    ),
    review_source = "AI_reviewed"
  )

# Save reviewed lexicon ------------------------------------------------------
write_csv(lex, 'results/study2/expanded_lexicon_ai_reviewed.csv')

# Create final clean lexicon -------------------------------------------------
final_lex <- lex %>%
  filter(eval == "keep") %>%
  select(term, category)

write_csv(final_lex, 'results/study2/expanded_lexicon_final.csv')

# Summary stats --------------------------------------------------------------
n_removed <- sum(lex$eval == "remove")
n_kept <- sum(lex$eval == "keep")

message(sprintf("✅ AI Review Complete:
  • Removed: %d terms
  • Kept: %d terms
  • Final lexicon: %d terms
  
Files written:
  - expanded_lexicon_ai_reviewed.csv (with review details)
  - expanded_lexicon_final.csv (clean version for use)", 
  n_removed, n_kept, nrow(final_lex))) 