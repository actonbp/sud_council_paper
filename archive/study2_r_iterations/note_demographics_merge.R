#!/usr/bin/env Rscript
# NOTE: Instructions for merging demographic data with LLM labels
# Author: AI Assistant, 2025-08-01

# This file contains instructions for when you have the secure demographic file

# STEP 1: Load the clean dataset with LLM labels
library(dplyr)
library(readr)

# Load the few-shot labeled data (50-50 split)
labeled_data <- read_csv("results/study2/clean_participant_dataset_fewshot.csv")

# STEP 2: Load your secure demographic file
# demographics <- read_csv("path/to/secure_demographics.csv")
# Make sure it has a 'participant_id' column that matches the IDs in labeled_data

# STEP 3: Merge the datasets
# final_dataset <- labeled_data %>%
#   left_join(demographics, by = "participant_id")

# STEP 4: Create descriptive statistics by interest group
# descriptives <- final_dataset %>%
#   group_by(ai_label) %>%
#   summarise(
#     n = n(),
#     # Demographics
#     age_mean = mean(age, na.rm = TRUE),
#     age_sd = sd(age, na.rm = TRUE),
#     gender_female_pct = mean(gender == "Female", na.rm = TRUE) * 100,
#     race_white_pct = mean(race == "White", na.rm = TRUE) * 100,
#     # Academic
#     year_mean = mean(academic_year, na.rm = TRUE),
#     gpa_mean = mean(gpa, na.rm = TRUE),
#     # Psychology/counseling background
#     psych_major_pct = mean(major == "Psychology", na.rm = TRUE) * 100,
#     prior_counseling_pct = mean(prior_counseling_experience == "Yes", na.rm = TRUE) * 100,
#     # Family/personal experience
#     family_sud_pct = mean(family_sud_history == "Yes", na.rm = TRUE) * 100,
#     personal_therapy_pct = mean(personal_therapy == "Yes", na.rm = TRUE) * 100
#   )

# STEP 5: Statistical tests for group differences
# t.test(age ~ ai_label, data = final_dataset)
# chisq.test(final_dataset$ai_label, final_dataset$gender)
# etc.

# STEP 6: Create publication-ready table
# library(gt)
# descriptives %>%
#   gt() %>%
#   tab_header(
#     title = "Table X",
#     subtitle = "Demographic Characteristics by SUD Counseling Interest"
#   ) %>%
#   fmt_number(columns = contains("mean"), decimals = 2) %>%
#   fmt_number(columns = contains("sd"), decimals = 2) %>%
#   fmt_number(columns = contains("pct"), decimals = 1)

print("This script provides a template for demographic analysis once you have the secure data file.")
print("Key variables to examine:")
print("- Age, gender, race/ethnicity")
print("- Academic year, major, GPA")
print("- Prior counseling/psychology experience")
print("- Family history of SUD")
print("- Personal therapy experience")
print("- Career certainty measures")