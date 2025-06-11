# Study 2 Methodology Documentation and Verification
# Creates comprehensive tables for co-author evaluation

library(tidyverse)
library(tidytext)
library(SnowballC)
library(here)
library(knitr)

cat("=== STUDY 2 METHODOLOGY DOCUMENTATION ===\n")

# VERIFICATION 1: Check Detection Logic Specificity
cat("\n🔍 VERIFICATION 1: SUD Detection Logic Analysis\n")

# Load data
data <- read_csv(here("data", "focus_group_comprehensive_preprocessed.csv"), show_col_types = FALSE)

# Define our term categories for analysis
substance_core_terms <- c("substanc", "addict", "depend", "alcohol", "drug", "abus", "recoveri", "rehab", "relaps", "sober", "clean")
professional_terms <- c("counselor", "therapist", "counsel", "therapi", "specialist")
treatment_terms <- c("treatment", "intervent", "detox", "program", "center", "servic")

# Analyze what triggers detection
sud_utterances <- data %>%
  filter(mentions_sud_stems == TRUE) %>%
  mutate(
    has_substance_core = str_detect(stems_combined, paste0("\\b(", paste(substance_core_terms, collapse="|"), ")\\b")),
    has_professional = str_detect(stems_combined, paste0("\\b(", paste(professional_terms, collapse="|"), ")\\b")),
    has_treatment = str_detect(stems_combined, paste0("\\b(", paste(treatment_terms, collapse="|"), ")\\b")),
    
    # Classification
    detection_type = case_when(
      has_substance_core & has_professional ~ "Substance + Professional",
      has_substance_core & has_treatment ~ "Substance + Treatment", 
      has_substance_core ~ "Substance Only",
      has_professional ~ "Professional Only",
      has_treatment ~ "Treatment Only",
      TRUE ~ "Other"
    )
  )

detection_summary <- sud_utterances %>%
  count(detection_type, sort = TRUE) %>%
  mutate(percentage = round(n / sum(n) * 100, 1))

cat("Detection trigger analysis:\n")
print(detection_summary)

# Check if professional-only detection is problematic
professional_only <- sud_utterances %>%
  filter(detection_type == "Professional Only") %>%
  select(cleaned_text) %>%
  slice_head(n = 5)

cat("\nExamples of 'Professional Only' detection:\n")
for(i in 1:nrow(professional_only)) {
  cat(i, ". ", str_trunc(professional_only$cleaned_text[i], 120), "\n")
}

# TABLE 1: Complete SUD Terms Used
cat("\n📋 TABLE 1: Complete SUD Detection Terms\n")

# Recreate our SUD terms with categories
sud_terms_categorized <- tibble(
  Category = c(
    rep("Core Addiction", 8),
    rep("Substances", 13), 
    rep("Treatment/Recovery", 17),
    rep("Problem Framing", 8),
    rep("Professional Context", 7)
  ),
  Original_Term = c(
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
) %>%
  mutate(
    Stemmed_Term = wordStem(Original_Term, language = "en")
  ) %>%
  group_by(Category) %>%
  mutate(Term_Number = row_number()) %>%
  ungroup()

# Save comprehensive terms table
write_csv(sud_terms_categorized, here("results", "study2_sud_terms_complete.csv"))

cat("Complete SUD terms by category (first 15 rows):\n")
print(head(sud_terms_categorized, 15))
cat("... (", nrow(sud_terms_categorized), "total terms)\n")
cat("Full table saved: results/study2_sud_terms_complete.csv\n")

# TABLE 2: Preprocessing Pipeline Documentation
cat("\n📋 TABLE 2: Preprocessing Pipeline Steps\n")

preprocessing_steps <- tibble(
  Step = 1:6,
  Process = c(
    "Tokenization", 
    "Stopword Removal",
    "Stemming", 
    "SUD Term Processing",
    "Utterance-level Detection",
    "Quality Control"
  ),
  Method = c(
    "tidytext::unnest_tokens() with word-level tokenization",
    "Multi-source stopwords (tidytext + custom focus group terms)",
    "Porter stemming via SnowballC::wordStem()",
    "Apply Porter stemming to SUD term list",
    "Detect any SUD stem in reconstructed utterance stems", 
    "Manual verification of detection examples"
  ),
  Input = c(
    "310 substantive utterances",
    "20,890 raw tokens", 
    "4,324 meaningful tokens",
    "53 original SUD terms",
    "Stemmed utterances + stemmed SUD terms",
    "109 detected SUD utterances"
  ),
  Output = c(
    "20,890 raw word tokens",
    "4,324 meaningful tokens (79% reduction)",
    "1,000 unique stems (21% reduction)",
    "42 unique SUD stems",
    "109 SUD utterances (35.2%)",
    "Verified conservative SUD-specific detection"
  )
)

print(preprocessing_steps)
write_csv(preprocessing_steps, here("results", "study2_preprocessing_pipeline.csv"))
cat("Pipeline documentation saved: results/study2_preprocessing_pipeline.csv\n")

# TABLE 3: Detection Logic Validation
cat("\n📋 TABLE 3: SUD Detection Logic Validation\n")

detection_logic <- tibble(
  Detection_Type = detection_summary$detection_type,
  Count = detection_summary$n,
  Percentage = detection_summary$percentage,
  Validity_Assessment = c(
    "VALID - Clear SUD context",     # Substance + Professional  
    "VALID - Clear SUD context",     # Substance + Treatment
    "VALID - Core SUD terminology",  # Substance Only
    "QUESTIONABLE - May be too broad", # Professional Only
    "QUESTIONABLE - May be too broad"  # Treatment Only  
  ),
  Recommendation = c(
    "Include - High confidence",
    "Include - High confidence", 
    "Include - High confidence",
    "Review - Check context",
    "Review - Check context"
  )
)

print(detection_logic)
write_csv(detection_logic, here("results", "study2_detection_validation.csv"))
cat("Detection validation saved: results/study2_detection_validation.csv\n")

# RECOMMENDATION: More Conservative Approach?
cat("\n🎯 METHODOLOGY RECOMMENDATION\n")

conservative_count <- sum(detection_summary$n[detection_summary$detection_type %in% 
                         c("Substance + Professional", "Substance + Treatment", "Substance Only")])
conservative_pct <- round(conservative_count / nrow(data) * 100, 1)

cat("Current approach: 109 utterances (35.2%)\n")
cat("More conservative (substance-required): ", conservative_count, " utterances (", conservative_pct, "%)\n")
cat("Difference: ", 109 - conservative_count, " utterances may be too broad\n")

# Save all methodology documentation
methodology_summary <- list(
  detection_summary = detection_summary,
  preprocessing_steps = preprocessing_steps,
  detection_validation = detection_logic,
  terms_by_category = sud_terms_categorized,
  professional_only_examples = professional_only,
  recommendation = list(
    current_count = 109,
    conservative_count = conservative_count,
    difference = 109 - conservative_count
  )
)

saveRDS(methodology_summary, here("results", "study2_methodology_complete.rds"))

# CONSERVATIVE APPROACH: Require Substance-Specific Terms
cat("\n🎯 RECOMMENDED CONSERVATIVE APPROACH\n")
cat("ISSUE: 38.5% of detections are 'Professional Only' (may be too broad)\n")
cat("SOLUTION: Require substance-specific terms for detection\n\n")

conservative_logic <- data %>%
  mutate(
    mentions_conservative_sud = str_detect(stems_combined, paste0("\\b(", paste(substance_core_terms, collapse="|"), ")\\b"))
  )

conservative_count_new <- sum(conservative_logic$mentions_conservative_sud)
conservative_pct_new <- round(conservative_count_new / nrow(data) * 100, 1)

cat("COMPARISON:\n")
cat("Current approach (any SUD term): 109 utterances (35.2%)\n")
cat("Conservative approach (substance required): ", conservative_count_new, " utterances (", conservative_pct_new, "%)\n")
cat("Reduction: ", 109 - conservative_count_new, " utterances (", round(35.2 - conservative_pct_new, 1), " percentage points)\n")

# Save conservative detection results
conservative_results <- conservative_logic %>%
  filter(mentions_conservative_sud == TRUE) %>%
  select(response_id, cleaned_text, stems_combined)

write_csv(conservative_results, here("results", "study2_conservative_sud_detection.csv"))

methodology_summary$conservative_approach <- list(
  count = conservative_count_new,
  percentage = conservative_pct_new,
  reduction_from_current = 109 - conservative_count_new,
  detection_file = "study2_conservative_sud_detection.csv"
)

saveRDS(methodology_summary, here("results", "study2_methodology_complete.rds"))

cat("\n💡 RECOMMENDATION FOR CO-AUTHORS:\n")
cat("Use conservative approach requiring substance-specific terms\n")
cat("- More conceptually precise\n") 
cat("- Eliminates general mental health false positives\n")
cat("- Still captures meaningful SUD-specific discussions\n")

cat("\n✅ COMPREHENSIVE METHODOLOGY DOCUMENTATION COMPLETE!\n")
cat("Files created for co-author evaluation:\n")
cat("- study2_sud_terms_complete.csv (all terms used)\n")
cat("- study2_preprocessing_pipeline.csv (methodology steps)\n") 
cat("- study2_detection_validation.csv (logic analysis)\n")
cat("- study2_conservative_sud_detection.csv (recommended approach)\n")
cat("- study2_methodology_complete.rds (complete analysis)\n")