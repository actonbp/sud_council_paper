# =============================================================================
# Study 2: Tidymodels Results Visualization
# SUD Counseling Career Research Project  
# =============================================================================
# Creates publication-ready figures from tidymodels analysis results
# Date: June 12, 2025
# =============================================================================

# Load required libraries
library(tidyverse)
library(here)
library(glue)
library(patchwork)
library(ggrepel)
library(scales)
library(RColorBrewer)

# Set up paths
results_dir <- here("results", "r", "study2_tidymodels")
figures_dir <- here("results", "figures")

if (!dir.exists(figures_dir)) {
  dir.create(figures_dir, recursive = TRUE)
}

cat("=== TIDYMODELS VISUALIZATION SUITE ===\n")
cat("Creating publication-ready figures from tidymodels results\n\n")

# =============================================================================
# 1. Load results
# =============================================================================

cat("📥 Loading tidymodels results...\n")

# Check if results exist
required_files <- c(
  "topic_term_probabilities.csv",
  "document_topic_probabilities.csv", 
  "topic_summaries.csv",
  "model_metadata.csv",
  "tuning_metrics.csv",
  "best_parameters.csv"
)

missing_files <- required_files[!file.exists(file.path(results_dir, required_files))]
if (length(missing_files) > 0) {
  stop("❌ Missing required files: ", paste(missing_files, collapse = ", "),
       "\nRun study2_tidymodels_analysis.R first.")
}

# Load all results
topic_terms <- read_csv(file.path(results_dir, "topic_term_probabilities.csv"), 
                       show_col_types = FALSE)
doc_topics <- read_csv(file.path(results_dir, "document_topic_probabilities.csv"),
                      show_col_types = FALSE)  
topic_summaries <- read_csv(file.path(results_dir, "topic_summaries.csv"),
                           show_col_types = FALSE)
model_metadata <- read_csv(file.path(results_dir, "model_metadata.csv"),
                          show_col_types = FALSE)
tuning_metrics <- read_csv(file.path(results_dir, "tuning_metrics.csv"),
                          show_col_types = FALSE)
best_params <- read_csv(file.path(results_dir, "best_parameters.csv"),
                       show_col_types = FALSE)

cat(glue("✅ Loaded results for {model_metadata$optimal_k} topics\n"))

# =============================================================================
# 2. Publication Figure 1: Top Terms by Topic
# =============================================================================

cat("\n📊 Creating Figure 1: Top Terms by Topic...\n")

# Enhanced top terms visualization
fig1 <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 8) %>%
  ungroup() %>%
  mutate(
    term = reorder_within(term, beta, topic),
    topic_label = glue("Topic {topic}")
  ) %>%
  ggplot(aes(x = beta, y = term, fill = factor(topic))) +
  geom_col(show.legend = FALSE, alpha = 0.8) +
  facet_wrap(~ topic_label, scales = "free_y", ncol = 2) +
  scale_y_reordered() +
  scale_fill_brewer(type = "qual", palette = "Set2") +
  scale_x_continuous(labels = percent_format(accuracy = 0.1)) +
  labs(
    title = "Topic Modeling Results: Most Characteristic Terms",
    subtitle = glue("Tidymodels LDA with k = {model_metadata$optimal_k} topics, {model_metadata$optimal_max_tokens} vocabulary terms"),
    x = "Term probability within topic (β)",
    y = "Terms (stemmed)",
    caption = "Higher probability indicates term is more characteristic of the topic"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    strip.text = element_text(size = 11, face = "bold"),
    axis.text.y = element_text(size = 9),
    axis.text.x = element_text(size = 9),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 8, color = "gray50")
  )

ggsave(file.path(figures_dir, "study2_tidymodels_topic_terms.png"), fig1,
       width = 12, height = 8, dpi = 300, bg = "white")

cat("✅ Figure 1 saved: study2_tidymodels_topic_terms.png\n")

# =============================================================================
# 3. Publication Figure 2: Model Selection Process
# =============================================================================

cat("\n📊 Creating Figure 2: Model Selection Process...\n")

# Model selection visualization
fig2 <- tuning_metrics %>%
  filter(.metric == "perplexity") %>%
  mutate(
    is_best = (k == best_params$k & 
               max_tokens == best_params$max_tokens & 
               min_freq == best_params$min_freq)
  ) %>%
  ggplot(aes(x = k, y = .estimate)) +
  geom_line(aes(group = interaction(max_tokens, min_freq), 
                color = factor(max_tokens)), 
            alpha = 0.6, size = 0.8) +
  geom_point(aes(color = factor(max_tokens), 
                 size = is_best, alpha = is_best)) +
  scale_size_manual(values = c(2, 4), guide = "none") +
  scale_alpha_manual(values = c(0.7, 1), guide = "none") +
  scale_color_brewer(type = "qual", palette = "Set1", name = "Max tokens") +
  facet_wrap(~ min_freq, labeller = label_both) +
  labs(
    title = "Hyperparameter Tuning Results",
    subtitle = "Cross-validated model selection (lower perplexity = better fit)",
    x = "Number of topics (k)",
    y = "Perplexity",
    caption = glue("Best model: k={best_params$k}, max_tokens={best_params$max_tokens}, min_freq={best_params$min_freq} (large points)")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    strip.text = element_text(size = 10, face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 8, color = "gray50")
  )

ggsave(file.path(figures_dir, "study2_tidymodels_model_selection.png"), fig2,
       width = 12, height = 8, dpi = 300, bg = "white")

cat("✅ Figure 2 saved: study2_tidymodels_model_selection.png\n")

# =============================================================================
# 4. Publication Figure 3: Topic Prevalence and Overlap
# =============================================================================

cat("\n📊 Creating Figure 3: Topic Prevalence Analysis...\n")

# Calculate topic prevalence
topic_prevalence <- doc_topics %>%
  group_by(topic) %>%
  summarize(
    avg_gamma = mean(gamma),
    documents_dominant = sum(gamma > 0.5),
    total_docs = n(),
    prevalence_pct = documents_dominant / total_docs * 100,
    .groups = "drop"
  ) %>%
  arrange(desc(avg_gamma))

# Topic prevalence plot
fig3a <- topic_prevalence %>%
  mutate(topic_label = glue("Topic {topic}")) %>%
  ggplot(aes(x = reorder(topic_label, avg_gamma), y = avg_gamma)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  geom_text(aes(label = percent(avg_gamma, accuracy = 0.1)), 
            hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_y_continuous(labels = percent_format(), 
                     expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Topic Prevalence in Corpus",
    x = "Topics",
    y = "Average document-topic probability",
    caption = "Higher values indicate topics that appear more frequently across documents"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 12, face = "bold"),
    panel.grid.minor = element_blank()
  )

# Document-topic distribution heatmap
fig3b <- doc_topics %>%
  mutate(document = as.numeric(document)) %>%
  filter(document <= 50) %>%  # Show first 50 documents for readability
  ggplot(aes(x = factor(topic), y = factor(document), fill = gamma)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "darkblue", 
                      name = "Topic\nprobability", 
                      labels = percent_format()) +
  labs(
    title = "Document-Topic Distribution (First 50 Documents)",
    x = "Topics",
    y = "Documents"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 12, face = "bold"),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank()
  )

# Combine prevalence plots
fig3 <- fig3a / fig3b + plot_layout(heights = c(1, 2))

ggsave(file.path(figures_dir, "study2_tidymodels_topic_prevalence.png"), fig3,
       width = 10, height = 10, dpi = 300, bg = "white")

cat("✅ Figure 3 saved: study2_tidymodels_topic_prevalence.png\n")

# =============================================================================
# 5. Manuscript Tables
# =============================================================================

cat("\n📋 Creating manuscript tables...\n")

# Table 1: Model specifications and performance
model_specs_table <- tibble(
  Specification = c(
    "Optimal number of topics (k)",
    "Vocabulary size (max_tokens)",
    "Minimum term frequency", 
    "Cross-validation folds",
    "Cross-validation repeats",
    "Total documents analyzed",
    "Preprocessing approach"
  ),
  Value = c(
    as.character(model_metadata$optimal_k),
    as.character(model_metadata$optimal_max_tokens),
    as.character(model_metadata$optimal_min_freq),
    "3",
    "3", 
    as.character(model_metadata$total_documents),
    "SUD terms removed, stemming applied"
  )
)

write_csv(model_specs_table, 
          file.path(results_dir, "manuscript_table_model_specs.csv"))

# Table 2: Topic characteristics
topic_characteristics_table <- topic_summaries %>%
  left_join(topic_prevalence, by = "topic") %>%
  select(
    Topic = topic,
    `Top Terms` = top_terms,
    `Average Probability` = avg_probability,
    `Prevalence (%)` = prevalence_pct,
    `Preliminary Theme` = preliminary_theme
  ) %>%
  mutate(
    `Average Probability` = round(`Average Probability`, 3),
    `Prevalence (%)` = round(`Prevalence (%)`, 1)
  )

write_csv(topic_characteristics_table,
          file.path(results_dir, "manuscript_table_topic_characteristics.csv"))

cat("✅ Manuscript tables saved:\n")
cat("   - manuscript_table_model_specs.csv\n")
cat("   - manuscript_table_topic_characteristics.csv\n")

# =============================================================================
# 6. Summary statistics for text
# =============================================================================

cat("\n📊 Generating summary statistics...\n")

# Analysis summary
analysis_summary <- tibble(
  metric = c(
    "Total substantive utterances",
    "Topics identified",
    "Vocabulary size",
    "Minimum term frequency",
    "Cross-validation approach",
    "Model selection metric",
    "Preprocessing approach"
  ),
  value = c(
    as.character(model_metadata$total_documents),
    as.character(model_metadata$optimal_k),
    as.character(model_metadata$optimal_max_tokens),
    glue("{model_metadata$optimal_min_freq}+ occurrences"),
    "3-fold × 3 repeats",
    "Perplexity (lower = better)",
    "Tidymodels + textrecipes"
  )
)

write_csv(analysis_summary, 
          file.path(results_dir, "analysis_summary_for_manuscript.csv"))

# Topic quality metrics
quality_metrics <- topic_prevalence %>%
  summarize(
    avg_topic_prevalence = mean(avg_gamma),
    topic_balance = sd(avg_gamma),
    max_topic_dominance = max(prevalence_pct),
    min_topic_presence = min(prevalence_pct)
  ) %>%
  mutate(across(where(is.numeric), round, 3))

write_csv(quality_metrics, 
          file.path(results_dir, "topic_quality_metrics.csv"))

cat("✅ Summary statistics saved:\n")
cat("   - analysis_summary_for_manuscript.csv\n") 
cat("   - topic_quality_metrics.csv\n")

# =============================================================================
# 7. Final summary report
# =============================================================================

cat("\n📋 VISUALIZATION SUMMARY\n")
cat("=======================\n")
cat("Publication figures created:\n")
cat("✅ study2_tidymodels_topic_terms.png - Main topic visualization\n")
cat("✅ study2_tidymodels_model_selection.png - Hyperparameter tuning\n")
cat("✅ study2_tidymodels_topic_prevalence.png - Topic distribution analysis\n")

cat("\nManuscript tables created:\n")
cat("✅ manuscript_table_model_specs.csv - Model specifications\n")
cat("✅ manuscript_table_topic_characteristics.csv - Topic details\n")

cat("\nSupporting files:\n")
cat("✅ analysis_summary_for_manuscript.csv - Key statistics\n")
cat("✅ topic_quality_metrics.csv - Topic quality assessment\n")

cat(glue("\n🎯 KEY FINDINGS:\n"))
cat(glue("• {model_metadata$optimal_k} topics identified through cross-validation\n"))
cat(glue("• {model_metadata$optimal_max_tokens} vocabulary terms retained\n"))
cat(glue("• {model_metadata$total_documents} utterances analyzed\n"))
cat(glue("• Topics show prevalence range: {round(min(topic_prevalence$prevalence_pct), 1)}% - {round(max(topic_prevalence$prevalence_pct), 1)}%\n"))

cat("\n🔄 NEXT STEPS:\n")
cat("1. Review topic characteristics table for theme interpretation\n")
cat("2. Research team should assign meaningful names to topics\n")
cat("3. Update manuscript methodology section with tidymodels approach\n")
cat("4. Include figures in Results section\n")
cat("5. Discuss topic prevalence and interpretation in Discussion\n")

cat("\n🏁 TIDYMODELS VISUALIZATION COMPLETE!\n")
cat("All figures ready for publication and manuscript integration\n")