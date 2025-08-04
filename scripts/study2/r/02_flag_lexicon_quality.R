#!/usr/bin/env Rscript
# 02_flag_lexicon_quality.R --------------------------------------------------
# Purpose: heuristic pass to flag questionable lexicon entries for manual
#          review. Adds a column `eval` with values "keep" or "remove".
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
})

lex <- read_csv('results/study2/expanded_lexicon.csv', show_col_types = FALSE)

should_remove <- function(term) {
  # heuristic rules --------------------------------------------------------
  if (nchar(term) < 4)                         return(TRUE)
  if (str_detect(term, "[A-Z]"))             return(TRUE)  # contains capitals
  if (str_detect(term, "[0-9]"))             return(TRUE)
  if (str_detect(term, "[[:punct:]&&[^\\-']]")) return(TRUE) # punctuation except hyphen/apostrophe
  if (str_detect(term, "(^-|--|\\.$|,$)"))    return(TRUE)  # leading dash, double dash, trailing punct
  FALSE
}

lex <- lex %>%
  mutate(eval = if_else(map_lgl(term, should_remove), 'remove', 'keep'))

write_csv(lex, 'results/study2/expanded_lexicon_review.csv')

message('✅ Review file written: results/study2/expanded_lexicon_review.csv') 