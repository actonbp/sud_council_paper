# Robustness Check: Broader Mental Health Context Analysis
# Appendix analysis comparing SUD-specific vs broader mental health scope

library(tidyverse)
library(tidytext)
library(SnowballC)
library(here)

cat("=== ROBUSTNESS CHECK: BROADER MENTAL HEALTH CONTEXT ===\n")
cat("Comparing conservative SUD-only vs inclusive mental health detection\n\n")

# Load preprocessed data
preprocessed_utterances <- read_csv(here("data", "focus_group_comprehensive_preprocessed.csv"), show_col_types = FALSE)

# APPROACH 1: Conservative SUD-only (current approach)
conservative_sud <- sum(preprocessed_utterances$mentions_sud_stems)
conservative_pct <- round(conservative_sud / nrow(preprocessed_utterances) * 100, 1)

cat("📊 APPROACH 1: Conservative SUD-only Detection\n")
cat("Utterances:", conservative_sud, "(", conservative_pct, "%)\n")
cat("Terms: SUD-specific terminology only\n\n")

# APPROACH 2: Broader Mental Health Context
cat("📊 APPROACH 2: Broader Mental Health Context\n")

# Broader mental health terms (for robustness check)
broader_mh_terms <- c(
  # Core SUD terms (same as conservative)
  "substance", "addiction", "addict", "addicted", "addictive", "dependence", "dependent", "dependency",
  "alcohol", "alcoholism", "alcoholic", "drug", "drugs", "cocaine", "heroin", "opioid", "opiate",
  "marijuana", "cannabis", "methamphetamine", "prescription",
  "recovery", "recovering", "rehabilitation", "rehab", "detox", "detoxification", "treatment",
  "therapy", "counseling", "intervention", "sobriety", "sober", "clean", "abstinence",
  "relapse", "methadone", "suboxone",
  "abuse", "abusing", "struggle", "struggling", "battle", "fighting", "overcome", "overcoming",
  "counselor", "therapist", "specialist", "program", "center", "services", "clinical",
  
  # Additional mental health context terms
  "mental", "health", "psychological", "psychiatric", "psychology", "psychiatrist",
  "medical", "medicine", "medication", "patient", "patients", "provider", "providers",
  "healthcare", "clinic", "hospital", "professional", "diagnosis", "therapeutic"
)

# Apply stemming to broader terms
broader_stems <- unique(wordStem(broader_mh_terms, language = "en"))

cat("Broader terms count:", length(broader_mh_terms), "→", length(broader_stems), "stems\n")

# Create broader detection using preprocessed tokens
preprocessed_tokens <- read_csv(here("data", "focus_group_tokens_preprocessed.csv"), show_col_types = FALSE)

# Reconstruct utterances with broader detection
broader_utterance_detection <- preprocessed_tokens %>%
  group_by(response_id) %>%
  summarise(
    mentions_broader_mh = any(word_stem %in% broader_stems),
    .groups = "drop"
  )

# Join with original data
broader_results <- preprocessed_utterances %>%
  left_join(broader_utterance_detection, by = "response_id") %>%
  replace_na(list(mentions_broader_mh = FALSE))

broader_count <- sum(broader_results$mentions_broader_mh)
broader_pct <- round(broader_count / nrow(broader_results) * 100, 1)

cat("Utterances:", broader_count, "(", broader_pct, "%)\n")
cat("Additional utterances captured:", broader_count - conservative_sud, "\n\n")

# COMPARISON ANALYSIS
cat("🔍 ROBUSTNESS COMPARISON\n")
cat("Conservative SUD approach:", conservative_sud, "utterances (", conservative_pct, "%)\n")
cat("Broader MH approach:", broader_count, "utterances (", broader_pct, "%)\n")
cat("Difference:", broader_count - conservative_sud, "utterances (", round((broader_pct - conservative_pct), 1), "% points)\n\n")

# Session-level comparison
session_comparison <- broader_results %>%
  group_by(session_id.x) %>%
  summarise(
    total_utterances = n(),
    sud_only = sum(mentions_sud_stems),
    broader_mh = sum(mentions_broader_mh),
    sud_pct = round(sud_only / total_utterances * 100, 1),
    broader_pct = round(broader_mh / total_utterances * 100, 1),
    difference_pct = broader_pct - sud_pct,
    .groups = "drop"
  ) %>%
  arrange(desc(total_utterances))

cat("📋 SESSION-LEVEL COMPARISON\n")
print(session_comparison)

# Example utterances only in broader context
broader_only <- broader_results %>%
  filter(mentions_broader_mh == TRUE & mentions_sud_stems == FALSE) %>%
  select(cleaned_text) %>%
  slice_head(n = 5)

cat("\n📝 EXAMPLES: Broader MH context (not SUD-specific)\n")
for(i in 1:nrow(broader_only)) {
  cat(i, ". ", str_trunc(broader_only$cleaned_text[i], 120), "\n")
}

# Save robustness check results
robustness_results <- list(
  conservative_approach = list(
    count = conservative_sud,
    percentage = conservative_pct,
    terms_used = "SUD-specific only"
  ),
  broader_approach = list(
    count = broader_count,
    percentage = broader_pct,
    terms_used = "SUD + general mental health"
  ),
  comparison = list(
    additional_utterances = broader_count - conservative_sud,
    percentage_point_difference = broader_pct - conservative_pct
  ),
  session_comparison = session_comparison,
  broader_only_examples = broader_only
)

saveRDS(robustness_results, here("results", "robustness_check_mental_health.rds"))

cat("\n💾 ROBUSTNESS CHECK COMPLETE!\n")
cat("Results saved to: results/robustness_check_mental_health.rds\n")
cat("\n🎯 RECOMMENDATION: Use conservative SUD-only approach for main analysis\n")
cat("- More conceptually precise\n")
cat("- Directly addresses SUD counseling interest\n")
cat("- Broader analysis available for sensitivity testing\n")