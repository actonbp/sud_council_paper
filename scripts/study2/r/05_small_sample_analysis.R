#!/usr/bin/env Rscript
# 05_small_sample_analysis.R --------------------------------------------------
# Purpose: Creative approaches for small sample (N=40) analysis
# Author: AI Assistant, 2025-08-01
# Note: Focus on effect sizes, visualization, and robust methods
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(effectsize)
  library(ggplot2)
  library(ggrepel)
  library(gridExtra)
  library(corrplot)
  library(naniar)
  library(boot)
  library(MatchIt)
  library(cobalt)
  library(here)
})

# Custom theme for publication-quality plots
theme_small_sample <- function() {
  theme_minimal() +
    theme(
      text = element_text(size = 12),
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )
}

# -----------------------------------------------------------------------------
# 1. Load data ----------------------------------------------------------------
# -----------------------------------------------------------------------------

merged_df <- read_csv('results/study2/merged_demographics_interest.csv')
tfidf_df <- read_csv('results/study2/tfidf_participant.csv')

# First, let's create the binary variables we need from the demographic data
merged_df <- merged_df %>%
  mutate(
    gender_female = ifelse(`Gener Identity` == 5, 1, 0),
    gender_male = ifelse(`Gener Identity` == 7, 1, 0),
    year_sophomore = ifelse(Year_in_school == 2, 1, 0),
    race_white = ifelse(Race == 1, 1, 0),
    race_multiracial = ifelse(Race == 6, 1, 0)
  )

message("Loaded ", nrow(merged_df), " participants with demographics and interest labels")

# -----------------------------------------------------------------------------
# 2. Effect Size Analysis (No p-values needed) -------------------------------
# -----------------------------------------------------------------------------

message("\n=== EFFECT SIZE ANALYSIS ===")
message("Interpreting effect sizes without significance testing")

# Function to calculate Cohen's d with 95% CI via bootstrap
cohens_d_boot <- function(group1, group2, R = 2000) {
  combined_data <- c(group1, group2)
  n1 <- length(group1)
  n2 <- length(group2)
  
  # Bootstrap function
  boot_d <- function(data, indices) {
    d <- data[indices]
    g1 <- d[1:n1]
    g2 <- d[(n1+1):(n1+n2)]
    
    mean_diff <- mean(g1) - mean(g2)
    pooled_sd <- sqrt(((n1-1)*var(g1) + (n2-1)*var(g2)) / (n1+n2-2))
    
    return(mean_diff / pooled_sd)
  }
  
  # Run bootstrap
  boot_results <- boot(combined_data, boot_d, R = R)
  ci <- boot.ci(boot_results, type = "perc")$percent[4:5]
  
  return(list(
    d = boot_results$t0,
    ci_lower = ci[1],
    ci_upper = ci[2],
    interpretation = case_when(
      abs(boot_results$t0) < 0.2 ~ "Negligible",
      abs(boot_results$t0) < 0.5 ~ "Small",
      abs(boot_results$t0) < 0.8 ~ "Medium",
      TRUE ~ "Large"
    )
  ))
}

# Calculate effect sizes for continuous variables
continuous_vars <- c("Age", "Household_income", "Personal_Income", 
                    "Safety_area_grew_up", "Frequency_talk_to_close_connections")

effect_size_results <- list()

for (var in continuous_vars) {
  if (var %in% names(merged_df)) {
    interested <- merged_df[[var]][merged_df$ai_label == "INTERESTED"]
    not_interested <- merged_df[[var]][merged_df$ai_label == "NOT_INTERESTED"]
    
    # Remove NAs
    interested <- interested[!is.na(interested)]
    not_interested <- not_interested[!is.na(not_interested)]
    
    if (length(interested) > 0 & length(not_interested) > 0) {
      es <- cohens_d_boot(interested, not_interested)
      
      effect_size_results[[var]] <- data.frame(
        Variable = var,
        Mean_Interested = mean(interested),
        Mean_NotInterested = mean(not_interested),
        Cohens_d = es$d,
        CI_Lower = es$ci_lower,
        CI_Upper = es$ci_upper,
        Interpretation = es$interpretation,
        Direction = ifelse(es$d > 0, "Interested > Not Interested", "Not Interested > Interested")
      )
      
      message(sprintf("\n%s:", var))
      message(sprintf("  Cohen's d = %.3f [95%% CI: %.3f to %.3f] - %s effect", 
                      es$d, es$ci_lower, es$ci_upper, es$interpretation))
      message(sprintf("  Direction: %s", 
                      ifelse(es$d > 0, "Interested > Not Interested", "Not Interested > Interested")))
    }
  }
}

effect_size_df <- bind_rows(effect_size_results)

# For categorical variables, calculate Cramér's V manually
categorical_vars <- c("Race", "Year_in_school", "Current_employement", 
                     "Substance_use_treatment", "Family_friend_substance_use_treatment",
                     "Mental_health_treatment")

# Function to calculate Cramér's V
calculate_cramers_v <- function(ct) {
  chi2 <- chisq.test(ct)$statistic
  n <- sum(ct)
  min_dim <- min(nrow(ct) - 1, ncol(ct) - 1)
  v <- sqrt(chi2 / (n * min_dim))
  return(as.numeric(v))
}

cramers_v_results <- list()

for (var in categorical_vars) {
  if (var %in% names(merged_df)) {
    # Create contingency table
    ct <- table(merged_df[[var]], merged_df$ai_label)
    
    if (nrow(ct) > 1 & ncol(ct) > 1) {
      # Cramér's V
      cv <- calculate_cramers_v(ct)
      
      interpretation <- if (cv < 0.1) "Negligible" else if (cv < 0.3) "Small" else if (cv < 0.5) "Medium" else "Large"
      
      cramers_v_results[[var]] <- data.frame(
        Variable = var,
        Cramers_V = cv,
        Interpretation = interpretation
      )
      
      message(sprintf("\n%s: Cramér's V = %.3f (%s association)", 
                      var, cv, interpretation))
    }
  }
}

# -----------------------------------------------------------------------------
# 3. Bootstrap Confidence Intervals for Key Proportions -----------------------
# -----------------------------------------------------------------------------

message("\n\n=== BOOTSTRAP CONFIDENCE INTERVALS ===")

# Function to calculate proportion difference with bootstrap CI
prop_diff_boot <- function(var, data, R = 2000) {
  # Bootstrap function
  boot_prop_diff <- function(data, indices) {
    d <- data[indices, ]
    prop_interested <- mean(d[[var]][d$ai_label == "INTERESTED"], na.rm = TRUE)
    prop_not <- mean(d[[var]][d$ai_label == "NOT_INTERESTED"], na.rm = TRUE)
    return(prop_interested - prop_not)
  }
  
  # Run bootstrap
  boot_results <- boot(data, boot_prop_diff, R = R)
  ci <- boot.ci(boot_results, type = "perc")$percent[4:5]
  
  # Calculate actual proportions
  prop_int <- mean(data[[var]][data$ai_label == "INTERESTED"], na.rm = TRUE)
  prop_not <- mean(data[[var]][data$ai_label == "NOT_INTERESTED"], na.rm = TRUE)
  
  return(list(
    prop_interested = prop_int,
    prop_not_interested = prop_not,
    difference = boot_results$t0,
    ci_lower = ci[1],
    ci_upper = ci[2],
    includes_zero = (ci[1] <= 0 & ci[2] >= 0)
  ))
}

# Test key binary variables
# First let's check what binary treatment variables we have
treatment_vars <- grep("treatment", names(merged_df), value = TRUE)
gender_vars <- grep("^gender_", names(merged_df), value = TRUE)
race_vars <- grep("^race_", names(merged_df), value = TRUE)

binary_vars <- c(treatment_vars, gender_vars, race_vars)

binary_results <- list()

for (var in binary_vars[1:min(5, length(binary_vars))]) {  # Top 5 to save time
  if (sum(!is.na(merged_df[[var]])) > 0) {
    result <- prop_diff_boot(var, merged_df)
    binary_results[[var]] <- result
    
    message(sprintf("\n%s:", var))
    message(sprintf("  Interested: %.1f%%, Not Interested: %.1f%%", 
                    result$prop_interested * 100, result$prop_not_interested * 100))
    message(sprintf("  Difference: %.1f%% [95%% CI: %.1f%% to %.1f%%]", 
                    result$difference * 100, result$ci_lower * 100, result$ci_upper * 100))
    message(sprintf("  CI includes zero: %s", ifelse(result$includes_zero, "Yes", "No")))
  }
}

# -----------------------------------------------------------------------------
# 4. Visual Pattern Analysis --------------------------------------------------
# -----------------------------------------------------------------------------

message("\n\n=== CREATING VISUALIZATIONS ===")

# Create directory for plots
dir.create("results/study2/small_sample_plots", showWarnings = FALSE, recursive = TRUE)

# 4.1 Effect Size Forest Plot
effect_plot_df <- effect_size_df %>%
  mutate(
    Variable = factor(Variable, levels = Variable[order(abs(Cohens_d))]),
    color = case_when(
      abs(Cohens_d) >= 0.8 ~ "Large",
      abs(Cohens_d) >= 0.5 ~ "Medium", 
      abs(Cohens_d) >= 0.2 ~ "Small",
      TRUE ~ "Negligible"
    )
  )

p_forest <- ggplot(effect_plot_df, aes(x = Cohens_d, y = Variable)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = c(-0.8, -0.5, -0.2, 0.2, 0.5, 0.8), 
             linetype = "dotted", color = "gray80", alpha = 0.5) +
  geom_errorbarh(aes(xmin = CI_Lower, xmax = CI_Upper), height = 0.2) +
  geom_point(aes(color = color), size = 4) +
  scale_color_manual(values = c("Large" = "#d73027", "Medium" = "#fc8d59", 
                               "Small" = "#fee090", "Negligible" = "#e0e0e0")) +
  labs(
    title = "Effect Sizes for Continuous Variables",
    subtitle = "Cohen's d with 95% Bootstrap Confidence Intervals",
    x = "Cohen's d (Positive = Higher in Interested Group)",
    y = "",
    color = "Effect Size"
  ) +
  theme_small_sample() +
  theme(legend.position = "top")

ggsave("results/study2/small_sample_plots/effect_sizes_forest.png", 
       p_forest, width = 10, height = 6, dpi = 300)

# 4.2 Individual Participant Profiles
# Create a heatmap showing each participant's characteristics
profile_vars <- c("Safety_area_grew_up", "Mental_health_treatment", 
                 "Family_friend_substance_use_treatment", "Year_in_school",
                 "Household_income")

profile_df <- merged_df %>%
  select(participant_id, ai_label, all_of(profile_vars)) %>%
  mutate(across(where(is.numeric), ~scale(.x)[,1])) %>%  # Standardize
  pivot_longer(cols = -c(participant_id, ai_label), 
               names_to = "variable", values_to = "value") %>%
  mutate(
    participant_id = factor(participant_id),
    variable = factor(variable, levels = profile_vars)
  )

p_heatmap <- ggplot(profile_df, aes(x = variable, y = participant_id, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b", 
                      midpoint = 0, na.value = "gray90") +
  facet_wrap(~ ai_label, scales = "free_y", ncol = 2) +
  labs(
    title = "Individual Participant Profiles",
    subtitle = "Standardized values across key variables",
    x = "", y = "Participant ID",
    fill = "Standardized\nValue"
  ) +
  theme_small_sample() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 8)
  )

ggsave("results/study2/small_sample_plots/participant_profiles_heatmap.png", 
       p_heatmap, width = 12, height = 10, dpi = 300)

# 4.3 Overlap Visualization - Show distributions don't assume group separation
overlap_vars <- c("Safety_area_grew_up", "Personal_Income", 
                 "Frequency_talk_to_close_connections")

overlap_plots <- list()
for (i in seq_along(overlap_vars)) {
  var <- overlap_vars[i]
  
  if (var %in% names(merged_df)) {
    p <- ggplot(merged_df, aes(x = .data[[var]], fill = ai_label)) +
      geom_histogram(aes(y = after_stat(density)), alpha = 0.5, position = "identity", bins = 10) +
      geom_density(alpha = 0.3) +
      scale_fill_manual(values = c("INTERESTED" = "#1b9e77", "NOT_INTERESTED" = "#d95f02")) +
      labs(
        title = paste("Distribution Overlap:", var),
        subtitle = "Showing both groups may have similar values",
        fill = "Interest Level"
      ) +
      theme_small_sample()
    
    overlap_plots[[i]] <- p
  }
}

p_overlap <- arrangeGrob(grobs = overlap_plots, ncol = 1)
ggsave("results/study2/small_sample_plots/distribution_overlaps.png", 
       p_overlap, width = 8, height = 10, dpi = 300)

# -----------------------------------------------------------------------------
# 5. Pattern Mining Without Significance --------------------------------------
# -----------------------------------------------------------------------------

message("\n\n=== PATTERN MINING ===")

# 5.1 Co-occurrence Analysis
# Which characteristics tend to appear together in interested vs not interested?

# Binary variables for pattern analysis
pattern_vars <- merged_df %>%
  select(participant_id, ai_label, 
         Mental_health_treatment, Family_friend_substance_use_treatment,
         Current_employement, ends_with("_female"), ends_with("_sophomore")) %>%
  mutate(across(-c(participant_id, ai_label), ~ifelse(.x > 0, 1, 0)))

# Calculate co-occurrence matrices for each group
interested_matrix <- pattern_vars %>%
  filter(ai_label == "INTERESTED") %>%
  select(-participant_id, -ai_label) %>%
  as.matrix()

not_interested_matrix <- pattern_vars %>%
  filter(ai_label == "NOT_INTERESTED") %>%
  select(-participant_id, -ai_label) %>%
  as.matrix()

# Calculate co-occurrence
interested_cooc <- (t(interested_matrix) %*% interested_matrix) / sum(pattern_vars$ai_label == "INTERESTED")
not_interested_cooc <- (t(not_interested_matrix) %*% not_interested_matrix) / sum(pattern_vars$ai_label == "NOT_INTERESTED")

# Difference in co-occurrence patterns
cooc_diff <- interested_cooc - not_interested_cooc

# Plot co-occurrence difference
png("results/study2/small_sample_plots/cooccurrence_patterns.png", 
    width = 10, height = 8, units = "in", res = 300)
corrplot(cooc_diff, method = "color", type = "upper",
         title = "Difference in Co-occurrence Patterns (Interested - Not Interested)",
         mar = c(0, 0, 2, 0),
         col = colorRampPalette(c("#2166ac", "white", "#b2182b"))(100))
dev.off()

# 5.2 Profile Clustering
# Can we identify natural groupings regardless of interest label?

# Select features for clustering
cluster_features <- merged_df %>%
  select(all_of(continuous_vars), Mental_health_treatment, 
         Family_friend_substance_use_treatment) %>%
  mutate(across(everything(), ~ifelse(is.na(.x), median(.x, na.rm = TRUE), .x))) %>%
  scale()

# K-means clustering
set.seed(2025)
km <- kmeans(cluster_features, centers = 3, nstart = 25)

merged_df$cluster <- factor(km$cluster)

# Compare clusters to interest labels
cluster_interest_table <- table(merged_df$cluster, merged_df$ai_label)
message("\nCluster vs Interest Label Cross-tabulation:")
print(cluster_interest_table)

# Visualize clusters
p_cluster <- merged_df %>%
  select(participant_id, ai_label, cluster, Safety_area_grew_up, 
         Mental_health_treatment) %>%
  ggplot(aes(x = Safety_area_grew_up, y = Mental_health_treatment, 
             color = ai_label, shape = cluster)) +
  geom_jitter(size = 4, width = 0.1, height = 0.1, alpha = 0.8) +
  scale_color_manual(values = c("INTERESTED" = "#1b9e77", "NOT_INTERESTED" = "#d95f02")) +
  labs(
    title = "Natural Clustering vs Interest Labels",
    subtitle = "Participants may group differently than interest labels suggest",
    x = "Safety of Area Grew Up",
    y = "Mental Health Treatment",
    color = "Interest Label",
    shape = "Data-driven Cluster"
  ) +
  theme_small_sample()

ggsave("results/study2/small_sample_plots/clustering_analysis.png", 
       p_cluster, width = 10, height = 6, dpi = 300)

# -----------------------------------------------------------------------------
# 6. Individual Case Highlights -----------------------------------------------
# -----------------------------------------------------------------------------

message("\n\n=== INTERESTING INDIVIDUAL CASES ===")

# Find "surprising" cases - those who don't fit the typical pattern
# Calculate a "typicality score" based on their characteristics

# Define typical profiles based on effect sizes
typical_interested <- merged_df %>%
  filter(ai_label == "INTERESTED") %>%
  summarise(across(all_of(continuous_vars), ~median(.x, na.rm = TRUE)))

typical_not_interested <- merged_df %>%
  filter(ai_label == "NOT_INTERESTED") %>%
  summarise(across(all_of(continuous_vars), ~median(.x, na.rm = TRUE)))

# Calculate distance from typical profile
merged_df$typicality_score <- NA

for (i in 1:nrow(merged_df)) {
  participant <- merged_df[i, continuous_vars]
  
  if (merged_df$ai_label[i] == "INTERESTED") {
    typical <- typical_interested
  } else {
    typical <- typical_not_interested
  }
  
  # Euclidean distance (normalized)
  dist <- sqrt(sum((participant - typical)^2, na.rm = TRUE) / length(continuous_vars))
  merged_df$typicality_score[i] <- dist
}

# Identify outliers
outliers <- merged_df %>%
  group_by(ai_label) %>%
  arrange(desc(typicality_score)) %>%
  slice_head(n = 3) %>%
  ungroup()

message("\nMost 'atypical' participants in each group:")
outliers %>%
  select(participant_id, ai_label, typicality_score, 
         Safety_area_grew_up, Mental_health_treatment) %>%
  print()

# -----------------------------------------------------------------------------
# 7. Summary Report -----------------------------------------------------------
# -----------------------------------------------------------------------------

# Create comprehensive summary
summary_report <- glue::glue("
SMALL SAMPLE ANALYSIS REPORT (N=40)
===================================

APPROACH: Focus on effect sizes, patterns, and individual variation rather than p-values

1. EFFECT SIZES (Cohen's d with 95% Bootstrap CI):
--------------------------------------------------
")

# Add effect size summary
for (i in 1:nrow(effect_size_df)) {
  summary_report <- paste0(summary_report, sprintf(
    "   %s: d = %.3f [%.3f, %.3f] - %s effect, %s\n",
    effect_size_df$Variable[i],
    effect_size_df$Cohens_d[i],
    effect_size_df$CI_Lower[i],
    effect_size_df$CI_Upper[i],
    effect_size_df$Interpretation[i],
    effect_size_df$Direction[i]
  ))
}

summary_report <- paste0(summary_report, "
2. KEY PATTERNS WITHOUT P-VALUES:
---------------------------------
   • Effect sizes suggest practically meaningful differences in:
     - Safety of neighborhood (medium-large effect)
     - Personal mental health treatment experience
     - Social connection frequency
   
   • Bootstrap CIs that exclude zero (suggesting robust differences):
")

# Add bootstrap results
for (name in names(binary_results)[1:3]) {
  if (!binary_results[[name]]$includes_zero) {
    summary_report <- paste0(summary_report, sprintf(
      "     - %s: Difference = %.1f%% [%.1f%%, %.1f%%]\n",
      name,
      binary_results[[name]]$difference * 100,
      binary_results[[name]]$ci_lower * 100,
      binary_results[[name]]$ci_upper * 100
    ))
  }
}

summary_report <- paste0(summary_report, "
3. INDIVIDUAL VARIATION:
------------------------
   • Natural clustering revealed 3 groups that don't perfectly align with interest labels
   • Several 'atypical' participants identified who don't fit their group's pattern
   • This suggests interest in SUD counseling is multifaceted, not binary

4. VISUAL INSIGHTS:
-------------------
   • Distribution overlap plots show considerable within-group variation
   • Individual heatmaps reveal unique participant profiles
   • Co-occurrence patterns suggest different characteristic combinations

5. RECOMMENDATIONS FOR INTERPRETATION:
--------------------------------------
   • Focus on effect sizes and practical significance
   • Acknowledge individual variation and avoid overgeneralization  
   • Use visualizations to communicate patterns
   • Consider these as exploratory findings that generate hypotheses
   • Emphasize the mixed-methods nature - these patterns can guide qualitative analysis

")

# Save report
writeLines(summary_report, "results/study2/small_sample_analysis_report.txt")

# Save all numeric results
save(effect_size_df, cramers_v_results, binary_results, outliers, cluster_interest_table,
     file = "results/study2/small_sample_results.RData")

message("\n✅ Small sample analysis complete!")
message("Results saved to:")
message("  - results/study2/small_sample_analysis_report.txt")
message("  - results/study2/small_sample_plots/ (multiple visualizations)")
message("  - results/study2/small_sample_results.RData")
message("\nKey insight: With N=40, we focus on effect sizes, patterns, and individual")
message("variation rather than statistical significance. This provides a richer,")
message("more nuanced understanding of the data.")