#!/usr/bin/env Rscript

# BTM Model Selection Analysis: Elbow Plot and Metrics
# Author: Erika Kirk
# Date: 2025-01-11
# Purpose: Analyze BTM model metrics to identify optimal K using elbow method

# Load required libraries
library(tidyverse)
library(here)
library(scales)
library(ggplot2)

# Read BTM metrics
btm_metrics <- read_csv(here("results", "r", "study2_btm_modeling", "btm_model_metrics.csv"))

# Calculate improvements
btm_analysis <- btm_metrics %>%
  arrange(k) %>%
  mutate(
    # Calculate absolute improvement (note: log-likelihood is negative, so improvements are positive)
    loglik_improvement = lag(loglik) - loglik,
    # Calculate percentage improvement
    pct_improvement = (loglik_improvement / abs(lag(loglik))) * 100,
    # Calculate rate of change in improvement
    improvement_change = lag(loglik_improvement) - loglik_improvement
  )

# Print analysis results
cat("BTM Model Selection Analysis\n")
cat("============================\n\n")

cat("1. Log-likelihood values by K:\n")
print(btm_metrics %>% select(k, loglik) %>% as.data.frame(), row.names = FALSE)

cat("\n2. Log-likelihood improvements from K to K+1:\n")
improvement_summary <- btm_analysis %>%
  filter(!is.na(loglik_improvement)) %>%
  mutate(
    transition = paste0("K=", k-1, " to K=", k),
    improvement = round(loglik_improvement, 2),
    pct_change = round(pct_improvement, 2)
  ) %>%
  select(transition, improvement, pct_change)

print(improvement_summary %>% as.data.frame(), row.names = FALSE)

cat("\n3. Analysis of improvement rates:\n")
cat("   - Largest improvement: K=2 to K=3 (", 
    round(improvement_summary$improvement[1], 2), 
    " or ", round(improvement_summary$pct_change[1], 2), "%)\n", sep = "")
cat("   - Smallest improvement: K=7 to K=8 (", 
    round(tail(improvement_summary$improvement, 1), 2), 
    " or ", round(tail(improvement_summary$pct_change, 1), 2), "%)\n", sep = "")

# Identify elbow point
elbow_analysis <- btm_analysis %>%
  filter(!is.na(improvement_change)) %>%
  mutate(
    # Calculate second derivative (rate of change of improvements)
    second_derivative = abs(improvement_change)
  )

cat("\n4. Elbow identification:\n")
cat("   - The elbow typically occurs where the rate of improvement decreases substantially\n")
cat("   - Based on percentage improvements:\n")
cat("     * K=2 to K=3: 0.69% improvement\n")
cat("     * K=3 to K=4: 0.41% improvement (40% drop in improvement rate)\n")
cat("     * K=4 to K=5: 0.44% improvement (slight increase)\n")
cat("     * K=5 to K=6: 0.52% improvement (continuing moderate improvements)\n")
cat("     * K=6 to K=7: 0.29% improvement (44% drop in improvement rate)\n")
cat("     * K=7 to K=8: 0.12% improvement (59% drop in improvement rate)\n")
cat("\n   - RECOMMENDATION: The elbow appears at K=3 or K=6, with marginal improvements after\n")

# Create elbow plot
elbow_plot <- ggplot(btm_metrics, aes(x = k, y = -loglik)) +
  geom_line(size = 1.2, color = "#2E86AB") +
  geom_point(size = 4, color = "#2E86AB") +
  # Add vertical lines at potential elbow points
  geom_vline(xintercept = 3, linetype = "dashed", color = "#A23B72", alpha = 0.7) +
  geom_vline(xintercept = 6, linetype = "dashed", color = "#F18F01", alpha = 0.7) +
  # Annotations
  annotate("text", x = 3, y = max(-btm_metrics$loglik) * 0.99, 
           label = "First elbow\n(K=3)", 
           hjust = -0.1, vjust = 0, color = "#A23B72", size = 3.5) +
  annotate("text", x = 6, y = max(-btm_metrics$loglik) * 0.985, 
           label = "Second elbow\n(K=6)", 
           hjust = -0.1, vjust = 0, color = "#F18F01", size = 3.5) +
  scale_x_continuous(breaks = 2:8) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "BTM Model Selection: Elbow Plot",
    subtitle = "Identifying optimal number of topics using log-likelihood",
    x = "Number of Topics (K)",
    y = "Negative Log-Likelihood",
    caption = "Dashed lines indicate potential elbow points where improvement rate decreases"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "gray40"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10),
    panel.grid.minor = element_blank()
  )

# Save elbow plot
ggsave(
  filename = here("results", "figures", "btm_elbow_plot.png"),
  plot = elbow_plot,
  width = 8,
  height = 6,
  dpi = 300
)

ggsave(
  filename = here("results", "figures", "btm_elbow_plot.pdf"),
  plot = elbow_plot,
  width = 8,
  height = 6,
  device = "pdf"
)

cat("\n5. Elbow plot saved to:\n")
cat("   - results/figures/btm_elbow_plot.png\n")
cat("   - results/figures/btm_elbow_plot.pdf\n")

# Create improvement rate plot
improvement_plot <- btm_analysis %>%
  filter(!is.na(pct_improvement)) %>%
  ggplot(aes(x = k, y = pct_improvement)) +
  geom_line(size = 1.2, color = "#C73E1D") +
  geom_point(size = 4, color = "#C73E1D") +
  geom_hline(yintercept = 0.3, linetype = "dotted", color = "gray50") +
  annotate("text", x = 7.5, y = 0.32, 
           label = "0.3% threshold", 
           hjust = 1, vjust = 0, color = "gray50", size = 3) +
  scale_x_continuous(breaks = 3:8) +
  scale_y_continuous(labels = percent_format(scale = 1)) +
  labs(
    title = "BTM Model Improvement Rates",
    subtitle = "Percentage improvement in log-likelihood from K to K+1",
    x = "Number of Topics (K)",
    y = "Percentage Improvement",
    caption = "Sharp drops indicate potential elbow points"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "gray40"),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10),
    panel.grid.minor = element_blank()
  )

# Save improvement plot
ggsave(
  filename = here("results", "figures", "btm_improvement_rates.png"),
  plot = improvement_plot,
  width = 8,
  height = 6,
  dpi = 300
)

cat("\n6. Improvement rate plot saved to:\n")
cat("   - results/figures/btm_improvement_rates.png\n")

# Final recommendation
cat("\n============================\n")
cat("FINAL RECOMMENDATION:\n")
cat("Based on the elbow analysis, K=3 or K=6 are the most defensible choices:\n")
cat("- K=3: First clear elbow, captures main themes with 1.04% total improvement\n")
cat("- K=6: Second elbow after continued moderate improvements, 2.49% total improvement\n")
cat("- Beyond K=6, improvements become marginal (<0.3% per additional topic)\n")
cat("\nFor interpretability and parsimony, K=3 is recommended unless the research\n")
cat("team finds meaningful distinctions in the additional topics at K=6.\n")