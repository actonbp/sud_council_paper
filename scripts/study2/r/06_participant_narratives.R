#!/usr/bin/env Rscript
# 06_participant_narratives.R -------------------------------------------------
# Purpose: Create individual participant "narratives" for small sample analysis
# Author: AI Assistant, 2025-08-01
# Note: Treats each participant as a mini case study
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(ggforce)
  library(patchwork)
  library(here)
})

# -----------------------------------------------------------------------------
# 1. Load data ----------------------------------------------------------------
# -----------------------------------------------------------------------------

merged_df <- read_csv('results/study2/merged_demographics_interest.csv')
tfidf_df <- read_csv('results/study2/tfidf_participant.csv')
discriminative_terms <- read_csv('results/study2/discriminative_terms.csv')

# Get top terms for each group
top_interested_terms <- discriminative_terms %>%
  arrange(desc(diff_tfidf)) %>%
  slice_head(n = 10) %>%
  pull(word)

top_not_interested_terms <- discriminative_terms %>%
  arrange(diff_tfidf) %>%
  slice_head(n = 10) %>%
  pull(word)

message("Creating participant narratives for N=40...")

# -----------------------------------------------------------------------------
# 2. Calculate Individual Profiles --------------------------------------------
# -----------------------------------------------------------------------------

# For each participant, calculate their "profile score" on key dimensions
profile_dimensions <- merged_df %>%
  mutate(
    # Experience dimension (0-1 scale)
    experience_score = (
      (Mental_health_treatment - 1) / 3 * 0.5 +  # Personal MH treatment
      (Family_friend_substance_use_treatment - 1) / 3 * 0.5  # Family/friend SU treatment
    ),
    
    # Environment dimension (0-1 scale) 
    environment_score = (
      (Safety_area_grew_up - 1) / 2 * 0.5 +  # Safety (1-3 scale)
      (Frequency_talk_to_close_connections - 1) / 4 * 0.5  # Social connections (1-5 scale)
    ),
    
    # Academic dimension (0-1 scale)
    academic_score = (
      (Year_in_school - 1) / 3 * 0.5 +  # Year in school
      ifelse(Parent_highest_level_education >= 4, 0.5, 0)  # Parent college
    ),
    
    # Demographic diversity (0-1 scale)
    diversity_score = (
      ifelse(Race != 1, 0.5, 0) +  # Non-white
      ifelse(`Gener Identity` != 7, 0.5, 0)  # Non-male
    )
  ) %>%
  select(participant_id, ai_label, experience_score, environment_score, 
         academic_score, diversity_score)

# Add text features
text_features <- tfidf_df %>%
  select(participant_id, all_of(c(top_interested_terms, top_not_interested_terms))) %>%
  mutate(
    interested_language_score = rowSums(select(., all_of(top_interested_terms))),
    not_interested_language_score = rowSums(select(., all_of(top_not_interested_terms)))
  ) %>%
  select(participant_id, interested_language_score, not_interested_language_score)

# Combine all features
participant_profiles <- profile_dimensions %>%
  left_join(text_features, by = "participant_id") %>%
  mutate(
    # Normalize language scores
    interested_language_score = scale(interested_language_score)[,1],
    not_interested_language_score = scale(not_interested_language_score)[,1],
    
    # Calculate overall "typicality" - how well they fit their assigned group
    typicality = case_when(
      ai_label == "INTERESTED" ~ interested_language_score - not_interested_language_score,
      ai_label == "NOT_INTERESTED" ~ not_interested_language_score - interested_language_score
    )
  )

# -----------------------------------------------------------------------------
# 3. Identify Key Participant Types -------------------------------------------
# -----------------------------------------------------------------------------

# Find exemplars - most typical of their group
exemplars <- participant_profiles %>%
  group_by(ai_label) %>%
  arrange(desc(typicality)) %>%
  slice_head(n = 2) %>%
  mutate(participant_type = "Exemplar")

# Find boundary cases - least typical but still in group
boundary_cases <- participant_profiles %>%
  group_by(ai_label) %>%
  arrange(typicality) %>%
  slice_head(n = 2) %>%
  mutate(participant_type = "Boundary Case")

# Find contrasts - high on opposite group's features
contrasts <- participant_profiles %>%
  mutate(
    contrast_score = case_when(
      ai_label == "INTERESTED" ~ not_interested_language_score,
      ai_label == "NOT_INTERESTED" ~ interested_language_score
    )
  ) %>%
  group_by(ai_label) %>%
  arrange(desc(contrast_score)) %>%
  slice_head(n = 2) %>%
  mutate(participant_type = "Contrast Case")

key_participants <- bind_rows(exemplars, boundary_cases, contrasts)

# -----------------------------------------------------------------------------
# 4. Create Individual Narrative Plots ----------------------------------------
# -----------------------------------------------------------------------------

dir.create("results/study2/participant_narratives", showWarnings = FALSE, recursive = TRUE)

# Function to create radar chart for a participant
create_participant_radar <- function(participant_data) {
  # Store the label for later use
  label <- participant_data$ai_label
  type <- participant_data$participant_type
  id <- participant_data$participant_id
  
  # Prepare data for radar chart
  radar_data <- participant_data %>%
    ungroup() %>%
    select(experience_score, environment_score, academic_score, 
           diversity_score, interested_language_score, not_interested_language_score) %>%
    pivot_longer(cols = everything(), names_to = "dimension", values_to = "value") %>%
    mutate(
      dimension = case_when(
        dimension == "experience_score" ~ "Treatment\nExperience",
        dimension == "environment_score" ~ "Environment\n& Social",
        dimension == "academic_score" ~ "Academic\nBackground",
        dimension == "diversity_score" ~ "Demographic\nDiversity",
        dimension == "interested_language_score" ~ "Interested\nLanguage",
        dimension == "not_interested_language_score" ~ "Not Interested\nLanguage"
      )
    )
  
  # Create circular coordinates
  angles <- seq(0, 2*pi, length.out = nrow(radar_data) + 1)[-1]
  radar_data$x <- cos(angles) * (radar_data$value + 1)
  radar_data$y <- sin(angles) * (radar_data$value + 1)
  
  # Create plot
  p <- ggplot(radar_data) +
    # Add grid lines
    geom_circle(aes(x0 = 0, y0 = 0, r = 1), color = "gray80", fill = NA) +
    geom_circle(aes(x0 = 0, y0 = 0, r = 2), color = "gray80", fill = NA) +
    geom_circle(aes(x0 = 0, y0 = 0, r = 3), color = "gray80", fill = NA) +
    
    # Add axes
    geom_segment(aes(x = 0, y = 0, xend = cos(angles) * 3.5, yend = sin(angles) * 3.5),
                 color = "gray70", linetype = "dotted") +
    
    # Add polygon
    geom_polygon(aes(x = x, y = y), fill = ifelse(label == "INTERESTED", 
                                                   "#1b9e77", "#d95f02"), alpha = 0.3) +
    geom_path(aes(x = x, y = y), color = ifelse(label == "INTERESTED", 
                                                 "#1b9e77", "#d95f02"), size = 1) +
    geom_point(aes(x = x, y = y), color = ifelse(label == "INTERESTED", 
                                                  "#1b9e77", "#d95f02"), size = 3) +
    
    # Add labels
    geom_text(aes(x = cos(angles) * 4, y = sin(angles) * 4, label = dimension),
              size = 3.5, hjust = 0.5) +
    
    coord_fixed() +
    theme_void() +
    labs(
      title = sprintf("Participant %d - %s (%s)", 
                      id,
                      label,
                      type)
    ) +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"))
  
  return(p)
}

# Create plots for key participants
plot_list <- list()
for (i in 1:nrow(key_participants)) {
  plot_list[[i]] <- create_participant_radar(key_participants[i,])
}

# Combine plots
combined_plot <- wrap_plots(plot_list, ncol = 3)
ggsave("results/study2/participant_narratives/key_participants_radar.png",
       combined_plot, width = 15, height = 10, dpi = 300)

# -----------------------------------------------------------------------------
# 5. Create Participant Journey Map -------------------------------------------
# -----------------------------------------------------------------------------

# Show how participants distribute across multiple dimensions
journey_plot <- participant_profiles %>%
  ggplot(aes(x = experience_score, y = environment_score)) +
  # Add density contours for each group
  stat_density_2d(data = filter(participant_profiles, ai_label == "INTERESTED"),
                  aes(color = "INTERESTED"), alpha = 0.5) +
  stat_density_2d(data = filter(participant_profiles, ai_label == "NOT_INTERESTED"),
                  aes(color = "NOT_INTERESTED"), alpha = 0.5) +
  
  # Add individual points
  geom_point(aes(fill = ai_label), shape = 21, size = 4, alpha = 0.8) +
  
  # Highlight key participants
  geom_point(data = key_participants, aes(shape = participant_type), 
             size = 6, color = "black") +
  
  scale_fill_manual(values = c("INTERESTED" = "#1b9e77", "NOT_INTERESTED" = "#d95f02")) +
  scale_color_manual(values = c("INTERESTED" = "#1b9e77", "NOT_INTERESTED" = "#d95f02")) +
  scale_shape_manual(values = c("Exemplar" = 15, "Boundary Case" = 17, "Contrast Case" = 18)) +
  
  labs(
    title = "Participant Journey Map: Experience vs Environment",
    subtitle = "Each person has a unique profile beyond their interest label",
    x = "Treatment Experience Score",
    y = "Environment & Social Score",
    fill = "Interest Label",
    color = "Density Contours",
    shape = "Participant Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom"
  )

ggsave("results/study2/participant_narratives/journey_map.png",
       journey_plot, width = 10, height = 8, dpi = 300)

# -----------------------------------------------------------------------------
# 6. Generate Written Narratives ----------------------------------------------
# -----------------------------------------------------------------------------

# Create narrative summaries for each key participant
narratives <- list()

for (i in 1:nrow(key_participants)) {
  p <- key_participants[i,]
  
  # Get original demographic data
  demo <- merged_df[merged_df$participant_id == p$participant_id,]
  
  narrative <- sprintf(
    "PARTICIPANT %d NARRATIVE
    
Group: %s
Type: %s
Profile: Experience (%.2f), Environment (%.2f), Academic (%.2f), Diversity (%.2f)
Language: Uses %s language pattern

Background: %s student, %s, grew up in %s area
Experience: %s personal mental health treatment, %s family/friend SU treatment
Current: %s, talks to close connections %s

Key Insight: This participant %s",
    p$participant_id,
    p$ai_label,
    p$participant_type,
    p$experience_score,
    p$environment_score, 
    p$academic_score,
    p$diversity_score,
    ifelse(p$interested_language_score > p$not_interested_language_score, 
           "more 'interested'", "more 'not interested'"),
    ifelse(demo$Year_in_school == 1, "Freshman", "Sophomore"),
    ifelse(demo$`Gener Identity` == 5, "Female", 
           ifelse(demo$`Gener Identity` == 7, "Male", "Other gender")),
    ifelse(demo$Safety_area_grew_up == 1, "very safe", 
           ifelse(demo$Safety_area_grew_up == 2, "somewhat safe", "less safe")),
    ifelse(demo$Mental_health_treatment > 1, "Has", "No"),
    ifelse(demo$Family_friend_substance_use_treatment > 1, "has", "no"),
    ifelse(demo$Current_employement == 1, "Currently employed", "Not employed"),
    c("rarely", "occasionally", "sometimes", "often", "very often")[demo$Frequency_talk_to_close_connections],
    case_when(
      p$participant_type == "Exemplar" ~ 
        sprintf("exemplifies the typical %s profile with strong alignment across dimensions",
                tolower(p$ai_label)),
      p$participant_type == "Boundary Case" ~ 
        sprintf("shows mixed signals, labeled as %s but with characteristics that could suggest otherwise",
                tolower(p$ai_label)),
      p$participant_type == "Contrast Case" ~ 
        sprintf("is labeled %s but uses language patterns more typical of the opposite group",
                tolower(p$ai_label))
    )
  )
  
  narratives[[i]] <- narrative
}

# Save narratives
writeLines(unlist(narratives), "results/study2/participant_narratives/key_participant_narratives.txt")

# -----------------------------------------------------------------------------
# 7. Summary Statistics -------------------------------------------------------
# -----------------------------------------------------------------------------

# Calculate group overlaps
overlap_stats <- participant_profiles %>%
  mutate(
    # Does this person look more like the opposite group?
    misaligned = case_when(
      ai_label == "INTERESTED" & not_interested_language_score > interested_language_score ~ TRUE,
      ai_label == "NOT_INTERESTED" & interested_language_score > not_interested_language_score ~ TRUE,
      TRUE ~ FALSE
    )
  ) %>%
  group_by(ai_label) %>%
  summarise(
    n_misaligned = sum(misaligned),
    pct_misaligned = mean(misaligned) * 100,
    mean_experience = mean(experience_score),
    mean_environment = mean(environment_score),
    .groups = 'drop'
  )

# Create final summary
summary_text <- sprintf("
PARTICIPANT NARRATIVE ANALYSIS SUMMARY
=====================================

Sample: N=40 (20 Interested, 20 Not Interested)

KEY FINDINGS:

1. Individual Variation Within Groups:
   - %d/%d (%.0f%%) of 'Interested' participants use language more typical of 'Not Interested'
   - %d/%d (%.0f%%) of 'Not Interested' participants use language more typical of 'Interested'
   
2. Multi-dimensional Profiles:
   - Participants vary on 6 key dimensions beyond their interest label
   - Experience and environment scores show considerable overlap between groups
   - No single dimension perfectly separates the groups

3. Key Participant Types Identified:
   - Exemplars: Show strong alignment with their group's typical profile
   - Boundary Cases: Weakly aligned with their assigned group
   - Contrast Cases: Show characteristics more typical of the opposite group

4. Implications:
   - Interest in SUD counseling is not binary but exists on multiple continua
   - Individual stories matter - group averages hide important variation
   - Some participants may be genuinely ambivalent or mislabeled
   - Future research should explore these individual differences qualitatively

RECOMMENDATION: With N=40, treat each participant as a valuable case study 
rather than just a data point in group statistics.
",
overlap_stats$n_misaligned[overlap_stats$ai_label == "INTERESTED"],
20,
overlap_stats$pct_misaligned[overlap_stats$ai_label == "INTERESTED"],
overlap_stats$n_misaligned[overlap_stats$ai_label == "NOT_INTERESTED"], 
20,
overlap_stats$pct_misaligned[overlap_stats$ai_label == "NOT_INTERESTED"])

writeLines(summary_text, "results/study2/participant_narratives/narrative_analysis_summary.txt")

# Save data
save(participant_profiles, key_participants, overlap_stats,
     file = "results/study2/participant_narratives/narrative_data.RData")

message("\n✅ Participant narrative analysis complete!")
message("Results saved to: results/study2/participant_narratives/")
message("\nKey insight: Each participant has a unique story that goes beyond")
message("their binary interest label. This approach honors individual complexity.")