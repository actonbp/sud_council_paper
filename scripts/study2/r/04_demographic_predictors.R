#!/usr/bin/env Rscript
# 04_demographic_predictors.R
# Purpose: Analyze demographic predictors of SUD counseling interest
# Author: AI Assistant
# Date: 2025-08-01

# Load libraries
library(tidyverse)
library(tidymodels)
library(gtsummary)
library(broom)
library(car)
library(effectsize)
library(here)

# Set up paths
results_dir <- here("results", "study2")
data_dir <- here("data")

# Load merged data from Python analysis
merged_df <- read_csv(file.path(results_dir, "merged_demographics_interest.csv"))

# Clean and prepare data
analysis_df <- merged_df %>%
  mutate(
    # Binary outcome
    interested_binary = ifelse(ai_label == "INTERESTED", 1, 0),
    
    # Recode categorical variables
    gender_female = ifelse(`Gener Identity` == 5, 1, 0),
    gender_male = ifelse(`Gener Identity` == 7, 1, 0),
    gender_other = ifelse(`Gener Identity` == 6, 1, 0),
    
    race_white = ifelse(Race == 1, 1, 0),
    race_black = ifelse(Race == 2, 1, 0),
    race_asian = ifelse(Race == 4, 1, 0),
    race_multiracial = ifelse(Race == 6, 1, 0),
    race_other = ifelse(Race %in% c(5, 7), 1, 0),
    
    year_freshman = ifelse(Year_in_school == 1, 1, 0),
    year_sophomore = ifelse(Year_in_school == 2, 1, 0),
    
    employed = ifelse(Current_employement == 1, 1, 0),
    
    parent_college = ifelse(Parent_highest_level_education >= 4, 1, 0),
    
    urban_origin = ifelse(Area_grew_up == 4, 1, 0),
    suburban_origin = ifelse(Area_grew_up == 2, 1, 0),
    
    su_treatment_personal = ifelse(Substance_use_treatment > 1, 1, 0),
    su_treatment_family = ifelse(Family_friend_substance_use_treatment > 1, 1, 0),
    mh_treatment_personal = ifelse(Mental_health_treatment > 1, 1, 0),
    
    # Continuous variables
    age_scaled = scale(Age)[,1],
    household_income_scaled = scale(Household_income)[,1],
    personal_income_scaled = scale(Personal_Income)[,1],
    safety_origin_scaled = scale(Safety_area_grew_up)[,1],
    social_connection_scaled = scale(Frequency_talk_to_close_connections)[,1]
  )

# Create summary table
cat("=== DESCRIPTIVE STATISTICS BY INTEREST GROUP ===\n\n")

# Categorical variables table
cat_vars <- c("gender_female", "gender_male", "race_white", "race_black", 
              "race_asian", "race_multiracial", "year_freshman", "year_sophomore",
              "employed", "parent_college", "su_treatment_family", "mh_treatment_personal")

cat_labels <- list(
  gender_female = "Female",
  gender_male = "Male", 
  race_white = "White",
  race_black = "Black",
  race_asian = "Asian",
  race_multiracial = "Multiracial",
  year_freshman = "Freshman",
  year_sophomore = "Sophomore",
  employed = "Currently Employed",
  parent_college = "Parent Has College Degree",
  su_treatment_family = "Family/Friend SU Treatment",
  mh_treatment_personal = "Personal MH Treatment"
)

# Create summary table using gtsummary
tbl_summary <- analysis_df %>%
  select(interested_binary, all_of(cat_vars), Age, Household_income, 
         Safety_area_grew_up, Frequency_talk_to_close_connections) %>%
  mutate(interested_binary = factor(interested_binary, 
                                   levels = c(0, 1), 
                                   labels = c("Not Interested", "Interested"))) %>%
  tbl_summary(
    by = interested_binary,
    label = c(cat_labels,
             Age = "Age",
             Household_income = "Household Income",
             Safety_area_grew_up = "Safety of Area Grew Up",
             Frequency_talk_to_close_connections = "Frequency Talk to Close Connections"),
    statistic = list(
      all_continuous() ~ "{mean} ({sd})",
      all_categorical() ~ "{n} ({p}%)"
    )
  ) %>%
  add_p() %>%
  add_overall() %>%
  bold_p()

# Print summary table
print(tbl_summary)

# Save summary table
tbl_summary %>%
  as_gt() %>%
  gt::gtsave(filename = file.path(results_dir, "demographic_summary_table.html"))

# Logistic regression models
cat("\n\n=== LOGISTIC REGRESSION MODELS ===\n\n")

# Model 1: Demographics only
model_demo <- glm(
  interested_binary ~ gender_female + gender_male + 
    race_black + race_asian + race_multiracial + race_other +
    year_sophomore + parent_college + age_scaled,
  data = analysis_df,
  family = binomial()
)

cat("Model 1: Demographics Only\n")
summary(model_demo)

# Model 2: Add experience variables
model_experience <- glm(
  interested_binary ~ gender_female + gender_male + 
    race_black + race_asian + race_multiracial + race_other +
    year_sophomore + parent_college + age_scaled +
    su_treatment_family + mh_treatment_personal,
  data = analysis_df,
  family = binomial()
)

cat("\n\nModel 2: Demographics + Experience\n")
summary(model_experience)

# Model 3: Add social/environmental factors
model_full <- glm(
  interested_binary ~ gender_female + gender_male + 
    race_black + race_asian + race_multiracial + race_other +
    year_sophomore + parent_college + age_scaled +
    su_treatment_family + mh_treatment_personal +
    safety_origin_scaled + social_connection_scaled,
  data = analysis_df,
  family = binomial()
)

cat("\n\nModel 3: Full Model\n")
summary(model_full)

# Extract odds ratios and confidence intervals
or_demo <- tidy(model_demo, exponentiate = TRUE, conf.int = TRUE)
or_experience <- tidy(model_experience, exponentiate = TRUE, conf.int = TRUE)
or_full <- tidy(model_full, exponentiate = TRUE, conf.int = TRUE)

# Create odds ratio table
or_table <- or_full %>%
  filter(term != "(Intercept)") %>%
  mutate(
    OR_CI = sprintf("%.2f (%.2f-%.2f)", estimate, conf.low, conf.high),
    p_value = sprintf("%.3f", p.value),
    term_label = case_when(
      term == "gender_female" ~ "Female (vs Other)",
      term == "gender_male" ~ "Male (vs Other)",
      term == "race_black" ~ "Black (vs White)",
      term == "race_asian" ~ "Asian (vs White)",
      term == "race_multiracial" ~ "Multiracial (vs White)",
      term == "race_other" ~ "Other Race (vs White)",
      term == "year_sophomore" ~ "Sophomore (vs Freshman)",
      term == "parent_college" ~ "Parent College Degree",
      term == "age_scaled" ~ "Age (per SD)",
      term == "su_treatment_family" ~ "Family/Friend SU Treatment",
      term == "mh_treatment_personal" ~ "Personal MH Treatment",
      term == "safety_origin_scaled" ~ "Safety Area Grew Up (per SD)",
      term == "social_connection_scaled" ~ "Social Connection (per SD)",
      TRUE ~ term
    )
  ) %>%
  select(Variable = term_label, `OR (95% CI)` = OR_CI, `p-value` = p_value)

cat("\n\n=== ODDS RATIOS FROM FULL MODEL ===\n")
print(or_table, n = Inf)

# Save odds ratio table
write_csv(or_table, file.path(results_dir, "demographic_odds_ratios.csv"))

# Model comparison
cat("\n\n=== MODEL COMPARISON ===\n")
anova(model_demo, model_experience, model_full, test = "Chisq")

# Calculate pseudo R-squared values
cat("\n\nPseudo R-squared values:\n")
cat("Model 1 (Demographics):", round(1 - model_demo$deviance/model_demo$null.deviance, 3), "\n")
cat("Model 2 (+ Experience):", round(1 - model_experience$deviance/model_experience$null.deviance, 3), "\n")
cat("Model 3 (Full):", round(1 - model_full$deviance/model_full$null.deviance, 3), "\n")

# Check for multicollinearity
cat("\n\n=== MULTICOLLINEARITY CHECK (VIF) ===\n")
vif_values <- vif(model_full)
print(vif_values)

# Effect size calculations for significant predictors
cat("\n\n=== EFFECT SIZES ===\n")

# For binary predictors in full model
binary_vars <- c("su_treatment_family", "mh_treatment_personal")

for (var in binary_vars) {
  # Calculate Cohen's h for proportions
  prop_interested <- mean(analysis_df[[var]][analysis_df$interested_binary == 1])
  prop_not_interested <- mean(analysis_df[[var]][analysis_df$interested_binary == 0])
  
  # Cohen's h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
  cohens_h <- 2 * (asin(sqrt(prop_interested)) - asin(sqrt(prop_not_interested)))
  
  cat(sprintf("\n%s:\n", var))
  cat(sprintf("  Proportion in Interested: %.3f\n", prop_interested))
  cat(sprintf("  Proportion in Not Interested: %.3f\n", prop_not_interested))
  cat(sprintf("  Cohen's h: %.3f\n", cohens_h))
}

# Save all results to RData file
save(analysis_df, model_demo, model_experience, model_full, 
     or_table, tbl_summary,
     file = file.path(results_dir, "demographic_analysis_results.RData"))

cat("\n\n✅ Analysis complete! Results saved to:", results_dir, "\n")