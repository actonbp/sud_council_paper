# =============================================================================
# Study 2: Topic Quality Assessment and Iterative Refinement
# SUD Counseling Career Research Project
# =============================================================================
# Reviews topic quality and provides recommendations for parameter adjustment
# Date: June 12, 2025
# =============================================================================

library(tidyverse)
library(here)
library(glue)
library(knitr)

cat("=== TOPIC QUALITY ASSESSMENT ===\n")
cat("Analyzing topic coherence, distinctiveness, and interpretability\n\n")

# =============================================================================
# 1. Load results and check if they exist
# =============================================================================

results_dir <- here("results", "r", "study2_tidymodels")

# Check if analysis has been run
required_files <- c("topic_term_probabilities.csv", "model_metadata.csv", "best_parameters.csv")
missing_files <- required_files[!file.exists(file.path(results_dir, required_files))]

if (length(missing_files) > 0) {
  cat("❌ Missing results files. Run study2_tidymodels_analysis.R first.\n")
  cat("Missing:", paste(missing_files, collapse = ", "), "\n")
  stop("Cannot proceed without analysis results.")
}

# Load results
topic_terms <- read_csv(file.path(results_dir, "topic_term_probabilities.csv"), show_col_types = FALSE)
model_metadata <- read_csv(file.path(results_dir, "model_metadata.csv"), show_col_types = FALSE)
best_params <- read_csv(file.path(results_dir, "best_parameters.csv"), show_col_types = FALSE)

if (file.exists(file.path(results_dir, "document_topic_probabilities.csv"))) {
  doc_topics <- read_csv(file.path(results_dir, "document_topic_probabilities.csv"), show_col_types = FALSE)
} else {
  doc_topics <- NULL
}

cat(glue("✅ Loaded results for {best_params$k} topics\n"))
cat(glue("📊 Parameters: max_tokens={best_params$max_tokens}, min_freq={best_params$min_freq}\n\n"))

# =============================================================================
# 2. Topic Quality Assessment Functions
# =============================================================================

# Function to assess topic overlap
assess_topic_overlap <- function(topic_terms, top_n = 10) {
  # Get top terms for each topic
  top_terms_by_topic <- topic_terms %>%
    group_by(topic) %>%
    slice_max(beta, n = top_n) %>%
    select(topic, term) %>%
    group_split()
  
  # Calculate overlap between all pairs of topics
  overlap_results <- expand_grid(
    topic1 = unique(topic_terms$topic),
    topic2 = unique(topic_terms$topic)
  ) %>%
    filter(topic1 < topic2) %>%
    rowwise() %>%
    mutate(
      terms1 = list(top_terms_by_topic[[topic1]]$term),
      terms2 = list(top_terms_by_topic[[topic2]]$term),
      overlap_count = length(intersect(terms1, terms2)),
      overlap_pct = overlap_count / top_n * 100,
      shared_terms = paste(intersect(terms1, terms2), collapse = ", ")
    ) %>%
    select(topic1, topic2, overlap_count, overlap_pct, shared_terms)
  
  return(overlap_results)
}

# Function to assess topic coherence (simplified)
assess_topic_coherence <- function(topic_terms, top_n = 8) {
  coherence_scores <- topic_terms %>%
    group_by(topic) %>%
    slice_max(beta, n = top_n) %>%
    summarize(
      top_term_prob = max(beta),
      avg_top_prob = mean(beta),
      prob_spread = max(beta) - min(beta),
      coherence_score = top_term_prob * avg_top_prob, # Simple coherence metric
      .groups = "drop"
    ) %>%
    mutate(
      coherence_rating = case_when(
        coherence_score > 0.15 ~ "High",
        coherence_score > 0.08 ~ "Medium", 
        TRUE ~ "Low"
      )
    )
  
  return(coherence_scores)
}

# Function to check for generic/meaningless terms
check_generic_terms <- function(topic_terms, top_n = 10) {
  generic_indicators <- c(
    "feel", "think", "know", "say", "go", "get", "want", "need", "come", "see",
    "good", "bad", "nice", "well", "way", "thing", "stuff", "lot", "bit",
    "time", "year", "day", "work", "job", "person", "people",
    "realli", "pretti", "veri", "kind", "sort", "guess", "maybe"
  )
  
  generic_analysis <- topic_terms %>%
    group_by(topic) %>%
    slice_max(beta, n = top_n) %>%
    summarize(
      generic_terms = sum(term %in% generic_indicators),
      generic_pct = generic_terms / top_n * 100,
      top_terms = paste(term[1:min(8, n())], collapse = ", "),
      .groups = "drop"
    ) %>%
    mutate(
      quality_flag = case_when(
        generic_pct > 50 ~ "🔴 Too Generic",
        generic_pct > 30 ~ "🟡 Some Generic Terms",
        TRUE ~ "✅ Specific Terms"
      )
    )
  
  return(generic_analysis)
}

# =============================================================================
# 3. Run Quality Assessments
# =============================================================================

cat("🔍 RUNNING TOPIC QUALITY ASSESSMENTS...\n\n")

# 1. Topic Overlap Analysis
cat("1️⃣ TOPIC OVERLAP ANALYSIS\n")
cat("=========================\n")
overlap_results <- assess_topic_overlap(topic_terms)

if (nrow(overlap_results) > 0) {
  cat("Topic pair overlaps (shared terms in top 10):\n")
  overlap_summary <- overlap_results %>%
    arrange(desc(overlap_pct))
  
  for (i in 1:nrow(overlap_summary)) {
    row <- overlap_summary[i, ]
    cat(glue("Topics {row$topic1} & {row$topic2}: {row$overlap_count} shared terms ({round(row$overlap_pct, 1)}%)\n"))
    if (row$overlap_count > 0) {
      cat(glue("  Shared: {row$shared_terms}\n"))
    }
  }
  
  max_overlap <- max(overlap_results$overlap_pct)
  cat(glue("\n📊 Maximum overlap: {round(max_overlap, 1)}%\n"))
  
  if (max_overlap > 40) {
    cat("🔴 HIGH OVERLAP DETECTED - Topics are too similar\n")
  } else if (max_overlap > 25) {
    cat("🟡 MODERATE OVERLAP - Some similarity between topics\n")
  } else {
    cat("✅ LOW OVERLAP - Topics are well-separated\n")
  }
} else {
  cat("Only one topic found - no overlap to assess\n")
}

cat(paste("\n", strrep("=", 50), "\n\n", sep = ""))

# 2. Topic Coherence Analysis  
cat("2️⃣ TOPIC COHERENCE ANALYSIS\n")
cat("===========================\n")
coherence_results <- assess_topic_coherence(topic_terms)

cat("Topic coherence scores:\n")
for (i in 1:nrow(coherence_results)) {
  row <- coherence_results[i, ]
  cat(glue("Topic {row$topic}: {row$coherence_rating} coherence (score: {round(row$coherence_score, 3)})\n"))
  cat(glue("  Top term prob: {round(row$top_term_prob, 3)}, Avg: {round(row$avg_top_prob, 3)}\n"))
}

avg_coherence <- mean(coherence_results$coherence_score)
cat(glue("\n📊 Average coherence: {round(avg_coherence, 3)}\n"))

if (avg_coherence > 0.12) {
  cat("✅ GOOD COHERENCE - Topics are well-defined\n")
} else if (avg_coherence > 0.08) {
  cat("🟡 MODERATE COHERENCE - Topics could be clearer\n")
} else {
  cat("🔴 LOW COHERENCE - Topics are poorly defined\n")
}

cat("\n" %R% strrep("=", 50) %R% "\n\n")

# 3. Generic Terms Analysis
cat("3️⃣ GENERIC TERMS ANALYSIS\n")
cat("=========================\n")
generic_results <- check_generic_terms(topic_terms)

cat("Generic term assessment:\n")
for (i in 1:nrow(generic_results)) {
  row <- generic_results[i, ]
  cat(glue("Topic {row$topic}: {row$quality_flag} ({row$generic_terms}/{10} generic)\n"))
  cat(glue("  Top terms: {row$top_terms}\n"))
}

avg_generic <- mean(generic_results$generic_pct)
cat(glue("\n📊 Average generic percentage: {round(avg_generic, 1)}%\n"))

if (avg_generic > 40) {
  cat("🔴 TOO MANY GENERIC TERMS - Need more aggressive filtering\n")
} else if (avg_generic > 25) {
  cat("🟡 SOME GENERIC TERMS - Consider additional filtering\n")
} else {
  cat("✅ SPECIFIC TERMS - Good vocabulary filtering\n")
}

cat("\n" %R% strrep("=", 50) %R% "\n\n")

# =============================================================================
# 4. Overall Assessment and Recommendations
# =============================================================================

cat("4️⃣ OVERALL ASSESSMENT & RECOMMENDATIONS\n")
cat("========================================\n")

# Calculate overall quality score
quality_score <- 0
quality_issues <- character()

# Overlap penalty
if (length(overlap_results) > 0) {
  max_overlap <- max(overlap_results$overlap_pct)
  if (max_overlap > 40) {
    quality_score <- quality_score - 3
    quality_issues <- c(quality_issues, "High topic overlap")
  } else if (max_overlap > 25) {
    quality_score <- quality_score - 1
    quality_issues <- c(quality_issues, "Moderate topic overlap")
  } else {
    quality_score <- quality_score + 2
  }
}

# Coherence bonus/penalty
if (avg_coherence > 0.12) {
  quality_score <- quality_score + 2
} else if (avg_coherence < 0.08) {
  quality_score <- quality_score - 2
  quality_issues <- c(quality_issues, "Low topic coherence")
}

# Generic terms penalty
if (avg_generic > 40) {
  quality_score <- quality_score - 3
  quality_issues <- c(quality_issues, "Too many generic terms")
} else if (avg_generic > 25) {
  quality_score <- quality_score - 1
  quality_issues <- c(quality_issues, "Some generic terms")
} else {
  quality_score <- quality_score + 1
}

# Overall rating
overall_rating <- case_when(
  quality_score >= 3 ~ "🟢 GOOD - Topics are publication-ready",
  quality_score >= 0 ~ "🟡 FAIR - Topics need minor improvements", 
  quality_score >= -3 ~ "🟠 POOR - Topics need significant improvements",
  TRUE ~ "🔴 BAD - Topics are not usable, major changes needed"
)

cat(glue("Overall Quality Score: {quality_score}\n"))
cat(glue("Rating: {overall_rating}\n\n"))

if (length(quality_issues) > 0) {
  cat("Main Issues Identified:\n")
  for (issue in quality_issues) {
    cat(glue("• {issue}\n"))
  }
  cat("\n")
}

# =============================================================================
# 5. Parameter Adjustment Recommendations
# =============================================================================

cat("5️⃣ PARAMETER ADJUSTMENT RECOMMENDATIONS\n")
cat("=======================================\n")

current_k <- best_params$k
current_max_tokens <- best_params$max_tokens
current_min_freq <- best_params$min_freq

cat(glue("Current parameters: k={current_k}, max_tokens={current_max_tokens}, min_freq={current_min_freq}\n\n"))

recommendations <- character()

# K recommendations
if (length(overlap_results) > 0 && max(overlap_results$overlap_pct) > 40) {
  if (current_k > 2) {
    recommendations <- c(recommendations, glue("• Reduce k to {current_k - 1} (too much overlap)"))
  } else {
    recommendations <- c(recommendations, "• Increase filtering instead of reducing k (already at minimum)")
  }
}

if (avg_coherence < 0.08 && current_k > 2) {
  recommendations <- c(recommendations, glue("• Reduce k to {current_k - 1} (low coherence)"))
}

# Vocabulary size recommendations  
if (avg_generic > 40) {
  new_max_tokens <- max(10, current_max_tokens - 5)
  recommendations <- c(recommendations, glue("• Reduce max_tokens to {new_max_tokens} (too many generic terms)"))
}

if (avg_coherence < 0.08) {
  new_max_tokens <- max(10, current_max_tokens - 5) 
  recommendations <- c(recommendations, glue("• Reduce max_tokens to {new_max_tokens} (improve coherence)"))
}

# Frequency threshold recommendations
if (avg_generic > 30) {
  new_min_freq <- current_min_freq + 1
  recommendations <- c(recommendations, glue("• Increase min_freq to {new_min_freq} (filter rare terms)"))
}

if (length(recommendations) == 0) {
  cat("✅ Current parameters seem appropriate. Topics may be ready for publication.\n")
} else {
  cat("Recommended parameter changes:\n")
  for (rec in recommendations) {
    cat(rec, "\n")
  }
}

cat("\n")

# =============================================================================
# 6. Next Steps
# =============================================================================

cat("6️⃣ NEXT STEPS\n")
cat("=============\n")

if (quality_score >= 3) {
  cat("🎉 Topics look good! Proceed with:\n")
  cat("1. Run study2_tidymodels_visualizations.R\n")
  cat("2. Assign meaningful names to topics\n") 
  cat("3. Update manuscript with results\n")
} else if (quality_score >= 0) {
  cat("🔧 Minor improvements needed:\n")
  cat("1. Consider implementing 1-2 recommended parameter changes\n")
  cat("2. Re-run analysis with adjusted parameters\n")
  cat("3. Re-assess topic quality\n")
} else {
  cat("🚨 Major improvements needed:\n")
  cat("1. Implement recommended parameter changes\n")
  cat("2. Consider additional stopword filtering\n")
  cat("3. Re-run analysis and reassess\n")
  cat("4. May need to iterate several times\n")
}

cat("\n📋 To adjust parameters, edit the tune_grid in study2_tidymodels_analysis.R\n")
cat("📋 Then re-run this assessment script to check improvements\n")

# =============================================================================
# 7. Save assessment results
# =============================================================================

# Save detailed results
assessment_summary <- tibble(
  assessment_date = Sys.Date(),
  overall_quality_score = quality_score,
  overall_rating = str_remove(overall_rating, "^[🟢🟡🟠🔴] "),
  max_topic_overlap_pct = ifelse(length(overlap_results) > 0, max(overlap_results$overlap_pct), 0),
  avg_coherence_score = avg_coherence,
  avg_generic_pct = avg_generic,
  current_k = current_k,
  current_max_tokens = current_max_tokens,
  current_min_freq = current_min_freq,
  main_issues = paste(quality_issues, collapse = "; "),
  recommendations = paste(recommendations, collapse = "; ")
)

write_csv(assessment_summary, file.path(results_dir, "topic_quality_assessment.csv"))

# Save detailed topic analysis
detailed_analysis <- generic_results %>%
  left_join(coherence_results, by = "topic") %>%
  arrange(topic)

write_csv(detailed_analysis, file.path(results_dir, "detailed_topic_analysis.csv"))

if (length(overlap_results) > 0) {
  write_csv(overlap_results, file.path(results_dir, "topic_overlap_analysis.csv"))
}

cat("\n💾 Assessment results saved to:\n")
cat("   - topic_quality_assessment.csv\n")
cat("   - detailed_topic_analysis.csv\n")
if (length(overlap_results) > 0) {
  cat("   - topic_overlap_analysis.csv\n")
}

cat("\n🏁 TOPIC QUALITY ASSESSMENT COMPLETE!\n")
cat("Review the recommendations above and adjust parameters if needed.\n")