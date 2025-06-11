# =============================================================================
# Study 2: LDA Topic Modeling Visualizations
# SUD Counseling Career Research Project
# =============================================================================
# Creates publication-ready visualizations for LDA topic modeling results
# Following June 10, 2025 Plan - Academic manuscript figures
# =============================================================================

# Load required libraries
library(tidyverse)
library(here)
library(ggplot2)
library(scales)
library(patchwork)
library(viridis)
library(ggrepel)

# Set up directories
results_dir <- here("results", "r", "study2_lda_modeling")
figures_dir <- here("results", "figures")
if (!dir.exists(figures_dir)) {
  dir.create(figures_dir, recursive = TRUE)
}

cat("=== Study 2: LDA Visualizations ===\n")
cat("Loading results from:", results_dir, "\n")
cat("Saving figures to:", figures_dir, "\n\n")

# =============================================================================
# LOAD DATA
# =============================================================================

# Load LDA results (updated file names)
topic_terms <- read_csv(file.path(results_dir, "topic_term_probabilities.csv"), show_col_types = FALSE)
document_topics <- read_csv(file.path(results_dir, "document_topic_probabilities.csv"), show_col_types = FALSE)
top_terms_per_topic <- read_csv(file.path(results_dir, "top_terms_per_topic.csv"), show_col_types = FALSE)
model_summary <- read_csv(file.path(results_dir, "model_summary.csv"), show_col_types = FALSE)
model_comparison <- read_csv(file.path(results_dir, "model_comparison.csv"), show_col_types = FALSE)

cat("Data loaded successfully\n")
cat("Number of topics:", model_summary$optimal_k, "\n")
cat("Detection rate:", model_summary$sud_detection_rate, "%\n\n")

# =============================================================================
# FIGURE 1: Topic Model Selection Metrics
# =============================================================================

cat("Creating Figure 1: Topic model selection metrics...\n")

# Create simple perplexity plot for model selection
fig1 <- model_comparison %>%
  ggplot(aes(x = k, y = perplexity)) +
  geom_line(color = "#2166AC", size = 1.2) +
  geom_point(color = "#2166AC", size = 3) +
  geom_point(data = filter(model_comparison, k == model_summary$optimal_k),
             color = "#D73027", size = 4) +
  labs(
    title = "LDA Topic Model Selection: Perplexity",
    subtitle = paste("Optimal k =", model_summary$optimal_k, "(highlighted in red)"),
    x = "Number of Topics (k)",
    y = "Perplexity (lower is better)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 11),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(figures_dir, "study2_lda_topic_selection.png"), 
       fig1, width = 12, height = 4, dpi = 300)

# =============================================================================
# FIGURE 2: Top Terms by Topic
# =============================================================================

cat("Creating Figure 2: Top terms by topic...\n")

# Prepare top terms data
top_terms_plot <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 8) %>%
  ungroup() %>%
  mutate(
    term = fct_reorder(term, beta),
    topic_label = paste("Topic", topic)
  )

# Create terms plot
fig2 <- top_terms_plot %>%
  ggplot(aes(x = beta, y = term, fill = topic_label)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~topic_label, scales = "free_y", ncol = 2) +
  scale_fill_viridis_d(option = "plasma", end = 0.8) +
  scale_x_continuous(labels = percent_format(accuracy = 0.1)) +
  labs(
    title = "Top Terms by Topic",
    subtitle = paste("LDA with k =", model_metadata$n_topics, "topics"),
    x = "Term Probability (β)",
    y = "Terms"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    strip.text = element_text(size = 11, face = "bold"),
    axis.title = element_text(size = 11),
    axis.text.y = element_text(size = 9),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank()
  )

ggsave(file.path(figures_dir, "study2_lda_top_terms.png"), 
       fig2, width = 10, height = 8, dpi = 300)

# =============================================================================
# FIGURE 3: Document-Topic Distribution
# =============================================================================

cat("Creating Figure 3: Document-topic distribution...\n")

# Calculate document-topic assignment (dominant topic per document)
document_assignments <- document_topics %>%
  group_by(response_id) %>%
  slice_max(gamma, n = 1, with_ties = FALSE) %>%
  ungroup()

# Create topic distribution plot
fig3a <- document_assignments %>%
  count(topic) %>%
  mutate(
    topic_label = paste("Topic", topic),
    percentage = n / sum(n) * 100
  ) %>%
  ggplot(aes(x = topic_label, y = percentage, fill = topic_label)) +
  geom_col(show.legend = FALSE) +
  scale_fill_viridis_d(option = "plasma", end = 0.8) +
  scale_y_continuous(labels = percent_format(scale = 1)) +
  labs(
    title = "Document Distribution Across Topics",
    subtitle = "Percentage of documents assigned to each topic",
    x = "Topic",
    y = "Percentage of Documents"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 11),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

# Create gamma distribution plot
fig3b <- document_topics %>%
  mutate(topic_label = paste("Topic", topic)) %>%
  ggplot(aes(x = gamma, fill = topic_label)) +
  geom_histogram(bins = 20, alpha = 0.7, position = "identity") +
  facet_wrap(~topic_label, ncol = 2) +
  scale_fill_viridis_d(option = "plasma", end = 0.8) +
  scale_x_continuous(labels = percent_format()) +
  labs(
    title = "Distribution of Topic Probabilities (γ)",
    subtitle = "Probability distribution of topics within documents",
    x = "Topic Probability (γ)",
    y = "Number of Documents"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    strip.text = element_text(size = 11, face = "bold"),
    axis.title = element_text(size = 11),
    legend.position = "none",
    panel.grid.minor = element_blank()
  )

# Combine document topic plots
fig3 <- fig3a / fig3b + plot_layout(heights = c(1, 1.2))

ggsave(file.path(figures_dir, "study2_lda_document_topics.png"), 
       fig3, width = 10, height = 10, dpi = 300)

# =============================================================================
# FIGURE 4: Topic Summary for Manuscript
# =============================================================================

cat("Creating Figure 4: Topic summary table...\n")

# Create a clean summary table
manuscript_summary <- topic_summaries %>%
  select(topic, top_terms, preliminary_theme) %>%
  mutate(
    topic_label = paste("Topic", topic),
    word_count = str_count(top_terms, ",") + 1,
    top_terms_clean = str_replace_all(top_terms, ", ", " • ")
  ) %>%
  arrange(topic)

# Create table plot
fig4 <- manuscript_summary %>%
  ggplot() +
  geom_text(aes(x = 0, y = topic, label = topic_label), 
            hjust = 0, size = 4, fontface = "bold") +
  geom_text(aes(x = 1, y = topic, label = preliminary_theme), 
            hjust = 0, size = 3.5) +
  geom_text(aes(x = 3, y = topic, label = top_terms_clean), 
            hjust = 0, size = 3) +
  scale_y_reverse() +
  xlim(0, 8) +
  labs(
    title = "LDA Topic Modeling Results Summary",
    subtitle = paste("Based on", model_metadata$sud_utterances, 
                     "SUD-related utterances (", 
                     model_metadata$detection_rate_percent, "% detection rate)")
  ) +
  theme_void() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 12, hjust = 0),
    plot.margin = margin(20, 20, 20, 20)
  ) +
  annotate("text", x = 0, y = 0, label = "Topic", fontface = "bold", hjust = 0) +
  annotate("text", x = 1, y = 0, label = "Preliminary Theme", fontface = "bold", hjust = 0) +
  annotate("text", x = 3, y = 0, label = "Top Terms", fontface = "bold", hjust = 0)

ggsave(file.path(figures_dir, "study2_lda_summary_table.png"), 
       fig4, width = 12, height = 6, dpi = 300)

# =============================================================================
# FIGURE 5: Method Comparison (LDA vs Previous Approach)
# =============================================================================

cat("Creating Figure 5: Method comparison...\n")

# Create comparison data
comparison_data <- tibble(
  method = c("Previous\n(Hierarchical Clustering)", "Current\n(LDA Topic Modeling)"),
  detection_rate = c(19.7, model_metadata$detection_rate_percent),
  approach = c("Conservative", "Less Conservative"),
  n_utterances = c(61, model_metadata$sud_utterances),
  n_themes = c(4, model_metadata$n_topics)
)

# Detection rate comparison
fig5a <- comparison_data %>%
  ggplot(aes(x = method, y = detection_rate, fill = approach)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(detection_rate, "%")), 
            vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("Conservative" = "#762A83", "Less Conservative" = "#1B7837")) +
  labs(
    title = "Detection Rate Comparison",
    y = "Detection Rate (%)",
    x = "",
    fill = "Approach"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

# Sample size comparison
fig5b <- comparison_data %>%
  ggplot(aes(x = method, y = n_utterances, fill = approach)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = n_utterances), 
            vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("Conservative" = "#762A83", "Less Conservative" = "#1B7837")) +
  labs(
    title = "Sample Size Comparison",
    y = "Number of Utterances",
    x = "",
    fill = "Approach"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

# Combine comparison plots
fig5 <- fig5a + fig5b + plot_layout(guides = "collect") & 
  theme(legend.position = "bottom")

ggsave(file.path(figures_dir, "study2_method_comparison.png"), 
       fig5, width = 10, height = 6, dpi = 300)

# =============================================================================
# SAVE MANUSCRIPT-READY OUTPUTS
# =============================================================================

cat("\n=== Creating Manuscript Outputs ===\n")

# Create manuscript table
manuscript_table <- manuscript_summary %>%
  select(
    `Topic` = topic_label,
    `Preliminary Theme` = preliminary_theme,
    `Representative Terms` = top_terms
  )

write_csv(manuscript_table, file.path(figures_dir, "study2_lda_manuscript_table.csv"))

# Create analysis summary for results section
analysis_summary <- tibble(
  metric = c(
    "Total focus group utterances",
    "SUD-related utterances included",
    "Detection rate",
    "Optimal number of topics (k)",
    "Vocabulary size",
    "Method",
    "Algorithm"
  ),
  value = c(
    format(model_metadata$total_utterances, big.mark = ","),
    format(model_metadata$sud_utterances, big.mark = ","),
    paste0(model_metadata$detection_rate_percent, "%"),
    as.character(model_metadata$n_topics),
    format(model_metadata$n_terms, big.mark = ","),
    "Less conservative filtering",
    "Latent Dirichlet Allocation (LDA)"
  )
)

write_csv(analysis_summary, file.path(figures_dir, "study2_lda_analysis_summary.csv"))

cat("\n=== Visualization Summary ===\n")
cat("Figures created:\n")
cat("• study2_lda_topic_selection.png - Topic model selection metrics\n")
cat("• study2_lda_top_terms.png - Top terms by topic\n") 
cat("• study2_lda_document_topics.png - Document-topic distributions\n")
cat("• study2_lda_summary_table.png - Summary table for manuscript\n")
cat("• study2_lda_method_comparison.png - Method comparison\n\n")

cat("Manuscript outputs:\n")
cat("• study2_lda_manuscript_table.csv - Table for manuscript\n")
cat("• study2_lda_analysis_summary.csv - Analysis summary for results section\n\n")

cat("All visualizations saved to:", figures_dir, "\n")
cat("Visualization script completed successfully!\n") 