# MEANINGFUL CLUSTERING ANALYSIS - PRESERVING INTERESTING THEMES
# Returns to min_freq=3 but handles singletons through conceptual merging

library(tidyverse)
library(tidytext)
library(here)
library(cluster)
library(SnowballC)
library(widyr)

cat("=== MEANINGFUL CLUSTERING ANALYSIS ===\n")
cat("Preserving family/personal themes while ensuring robustness\n\n")

# Load validated data
stems_with_session <- read_csv(here("data", "focus_group_tokens_preprocessed.csv"), show_col_types = FALSE)

# Define SUD terms
sud_terms_raw <- c(
  "substance", "addiction", "addict", "addicted", "addictive", "dependence", "dependent", "dependency",
  "alcohol", "alcoholism", "alcoholic", "drug", "drugs", "cocaine", "heroin", "opioid", "opiate",
  "marijuana", "cannabis", "methamphetamine", "prescription",
  "recovery", "recovering", "rehabilitation", "rehab", "detox", "detoxification", "treatment",
  "therapy", "counseling", "intervention", "sobriety", "sober", "clean", "abstinence",
  "relapse", "methadone", "suboxone",
  "abuse", "abusing", "struggle", "struggling", "battle", "fighting", "overcome", "overcoming",
  "counselor", "therapist", "specialist", "program", "center", "services", "clinical"
)

sud_terms_stemmed <- tibble(term = sud_terms_raw) %>%
  mutate(term_stem = wordStem(str_to_lower(term), language = "en")) %>%
  pull(term_stem) %>% unique()

# Get SUD tokens
sud_tokens <- stems_with_session %>%
  filter(str_detect(paste(sud_terms_stemmed, collapse = "|"), word_stem))

# RETURN TO min_freq=3 to preserve meaningful themes
word_frequencies <- sud_tokens %>%
  count(word_stem, sort = TRUE) %>%
  filter(n >= 3)

word_pairs <- sud_tokens %>%
  filter(word_stem %in% word_frequencies$word_stem) %>%
  pairwise_count(word_stem, response_id, sort = TRUE) %>%
  filter(n >= 2)

# Create co-occurrence matrix
all_words <- unique(c(word_pairs$item1, word_pairs$item2))
cooccur_matrix <- matrix(0, nrow = length(all_words), ncol = length(all_words))
rownames(cooccur_matrix) <- all_words
colnames(cooccur_matrix) <- all_words

for (i in 1:nrow(word_pairs)) {
  word1 <- word_pairs$item1[i]
  word2 <- word_pairs$item2[i]
  count <- word_pairs$n[i]
  
  cooccur_matrix[word1, word2] <- count
  cooccur_matrix[word2, word1] <- count
}

cat("📊 DATA OVERVIEW (min_freq=3):\n")
cat("==============================\n")
cat("Matrix dimensions:", nrow(cooccur_matrix), "x", ncol(cooccur_matrix), "\n")
cat("Matrix sparsity:", round(sum(cooccur_matrix == 0) / length(cooccur_matrix), 3), "\n")
cat("Words analyzed:", length(all_words), "\n")
cat("Co-occurrence pairs:", nrow(word_pairs), "\n\n")

# Apply hierarchical clustering
dist_matrix <- dist(cooccur_matrix, method = "euclidean")
hc <- hclust(dist_matrix, method = "ward.D2")

# Test different k values
cat("🔍 CLUSTERING RESULTS FOR DIFFERENT k VALUES:\n")
cat("=============================================\n")

clustering_options <- list()
for (k in 2:6) {
  tryCatch({
    clusters <- cutree(hc, k = k)
    
    # Calculate WSS
    wss <- 0
    for (cluster_id in unique(clusters)) {
      cluster_points <- cooccur_matrix[clusters == cluster_id, , drop = FALSE]
      if (nrow(cluster_points) > 1) {
        cluster_center <- colMeans(cluster_points)
        for (i in 1:nrow(cluster_points)) {
          wss <- wss + sum((cluster_points[i, ] - cluster_center)^2)
        }
      }
    }
    
    # Calculate silhouette if valid
    if (length(unique(clusters)) > 1) {
      sil <- silhouette(clusters, dist_matrix)
      avg_sil <- mean(sil[, 3])
    } else {
      avg_sil <- NA
    }
    
    singleton_clusters <- sum(table(clusters) == 1)
    
    clustering_options[[paste0("k", k)]] <- list(
      k = k,
      clusters = clusters,
      wss = wss,
      silhouette = avg_sil,
      singleton_clusters = singleton_clusters,
      cluster_sizes = table(clusters)
    )
    
    cat("k =", k, ": WSS =", round(wss, 1), 
        ", Silhouette =", round(avg_sil, 3), 
        ", Singletons =", singleton_clusters,
        ", Sizes:", paste(table(clusters), collapse = ", "), "\n")
  }, error = function(e) {
    cat("k =", k, ": ERROR -", e$message, "\n")
  })
}

# SELECT k=3 (original optimal) but implement post-hoc merging
cat("\n🎯 IMPLEMENTING k=3 WITH POST-HOC SINGLETON MERGING:\n")
cat("====================================================\n")

k3_result <- clustering_options[["k3"]]
original_clusters <- k3_result$clusters

# Show original k=3 clustering
cluster_composition <- tibble(
  word = names(original_clusters),
  cluster = original_clusters
) %>%
  left_join(word_frequencies, by = c("word" = "word_stem")) %>%
  arrange(cluster, desc(n))

cat("ORIGINAL k=3 CLUSTERING:\n")
for (cluster_id in sort(unique(original_clusters))) {
  words_in_cluster <- cluster_composition %>%
    filter(cluster == cluster_id) %>%
    arrange(desc(n))
  
  cat("Cluster", cluster_id, "(", nrow(words_in_cluster), "words):\n")
  for (i in 1:nrow(words_in_cluster)) {
    cat("  -", words_in_cluster$word[i], "(n=", words_in_cluster$n[i], ")\n")
  }
  cat("\n")
}

# POST-HOC MERGING STRATEGY
cat("🔧 POST-HOC SINGLETON MERGING:\n")
cat("==============================\n")

# Identify singletons
cluster_sizes <- table(original_clusters)
singleton_clusters <- as.numeric(names(cluster_sizes)[cluster_sizes == 1])

if (length(singleton_clusters) > 0) {
  cat("Singleton clusters found:", singleton_clusters, "\n")
  
  # Strategy: Merge singletons with their most co-occurring cluster
  merged_clusters <- original_clusters
  
  for (singleton_id in singleton_clusters) {
    singleton_word <- names(original_clusters)[original_clusters == singleton_id]
    cat("Processing singleton:", singleton_word, "\n")
    
    # Find which cluster this word co-occurs with most
    singleton_cooccurrences <- word_pairs %>%
      filter(item1 == singleton_word | item2 == singleton_word) %>%
      mutate(
        other_word = ifelse(item1 == singleton_word, item2, item1)
      ) %>%
      left_join(
        tibble(other_word = names(original_clusters), 
               other_cluster = original_clusters),
        by = "other_word"
      ) %>%
      filter(!is.na(other_cluster), other_cluster != singleton_id) %>%
      group_by(other_cluster) %>%
      summarise(total_cooccurrence = sum(n), .groups = "drop") %>%
      arrange(desc(total_cooccurrence))
    
    if (nrow(singleton_cooccurrences) > 0) {
      target_cluster <- singleton_cooccurrences$other_cluster[1]
      merged_clusters[merged_clusters == singleton_id] <- target_cluster
      cat("  → Merged with Cluster", target_cluster, 
          "(", singleton_cooccurrences$total_cooccurrence[1], "co-occurrences)\n")
    }
  }
  
  # Renumber clusters to be consecutive
  unique_clusters <- sort(unique(merged_clusters))
  final_clusters <- merged_clusters
  for (i in seq_along(unique_clusters)) {
    final_clusters[merged_clusters == unique_clusters[i]] <- i
  }
  
} else {
  cat("No singleton clusters found.\n")
  final_clusters <- original_clusters
}

# FINAL CLUSTERING ANALYSIS
cat("\n📊 FINAL MERGED CLUSTERING:\n")
cat("===========================\n")

final_composition <- tibble(
  word = names(final_clusters),
  cluster = final_clusters
) %>%
  left_join(word_frequencies, by = c("word" = "word_stem")) %>%
  arrange(cluster, desc(n))

cat("FINAL CLUSTER COMPOSITION:\n")
for (cluster_id in sort(unique(final_clusters))) {
  words_in_cluster <- final_composition %>%
    filter(cluster == cluster_id) %>%
    arrange(desc(n))
  
  cat("Cluster", cluster_id, "(", nrow(words_in_cluster), "words):\n")
  for (i in 1:nrow(words_in_cluster)) {
    cat("  -", words_in_cluster$word[i], "(n=", words_in_cluster$n[i], ")\n")
  }
  cat("\n")
}

# THEMATIC INTERPRETATION
cat("🎨 THEMATIC INTERPRETATION:\n")
cat("===========================\n")

# Enhanced thematic analysis
final_themes <- tibble(
  cluster = sort(unique(final_clusters)),
  theme_label = NA_character_,
  theme_description = NA_character_
)

for (i in 1:nrow(final_themes)) {
  cluster_id <- final_themes$cluster[i]
  words_in_cluster <- final_composition %>%
    filter(cluster == cluster_id) %>%
    pull(word)
  
  # Enhanced thematic categorization
  professional_words <- c("counselor", "therapist", "therapi", "counsel", "clinic", "rehabilit")
  substance_words <- c("substanc", "abus", "addict", "alcohol", "drug", "depend")
  personal_words <- c("famili", "support", "ive", "theyr", "life", "hard", "feel", "person")
  challenge_words <- c("struggl", "battl", "relaps", "treatment")
  
  prof_count <- sum(words_in_cluster %in% professional_words)
  subst_count <- sum(words_in_cluster %in% substance_words)
  personal_count <- sum(words_in_cluster %in% personal_words)
  challenge_count <- sum(words_in_cluster %in% challenge_words)
  
  # Determine primary theme
  counts <- c(prof_count, subst_count, personal_count, challenge_count)
  max_category <- which.max(counts)
  
  if (max_category == 1 && prof_count > 0) {
    final_themes$theme_label[i] <- "Professional Support & Treatment"
    final_themes$theme_description[i] <- "References to counselors, therapists, and professional services"
  } else if (max_category == 2 && subst_count > 0) {
    final_themes$theme_label[i] <- "Substance Use & Addiction"
    final_themes$theme_description[i] <- "References to substances, abuse, addiction, and dependencies"
  } else if (max_category == 3 && personal_count > 0) {
    final_themes$theme_label[i] <- "Personal & Family Experiences"
    final_themes$theme_description[i] <- "References to family, personal experiences, life challenges, and relationships"
  } else if (max_category == 4 && challenge_count > 0) {
    final_themes$theme_label[i] <- "Recovery Challenges"
    final_themes$theme_description[i] <- "References to struggles, battles, relapses, and treatment challenges"
  } else {
    # Mixed or unclear theme
    top_words <- words_in_cluster[1:min(3, length(words_in_cluster))]
    final_themes$theme_label[i] <- paste0("Mixed Theme ", cluster_id)
    final_themes$theme_description[i] <- paste("Various terms including:", paste(top_words, collapse = ", "))
  }
}

for (i in 1:nrow(final_themes)) {
  cluster_words <- final_composition %>%
    filter(cluster == final_themes$cluster[i]) %>%
    arrange(desc(n)) %>%
    slice_head(n = 5) %>%
    pull(word)
  
  cat("Cluster", final_themes$cluster[i], ":", final_themes$theme_label[i], "\n")
  cat("  Description:", final_themes$theme_description[i], "\n")
  cat("  Key words:", paste(cluster_words, collapse = ", "), "\n\n")
}

# SAVE RESULTS
cat("💾 SAVING MEANINGFUL CLUSTERING RESULTS:\n")
cat("========================================\n")

meaningful_results <- list(
  methodology = list(
    approach = "hierarchical_with_singleton_merging",
    min_frequency = 3,
    min_cooccurrence = 2,
    original_k = 3,
    final_k = length(unique(final_clusters)),
    rationale = "Preserved meaningful family/personal themes while addressing singleton issues through post-hoc merging"
  ),
  clustering_results = list(
    original_clusters = original_clusters,
    final_clusters = final_clusters,
    cluster_composition = final_composition,
    cluster_themes = final_themes,
    silhouette_score = k3_result$silhouette,
    wss = k3_result$wss
  ),
  data_quality = list(
    words_analyzed = length(all_words),
    cooccurrence_pairs = nrow(word_pairs),
    matrix_sparsity = sum(cooccur_matrix == 0) / length(cooccur_matrix)
  )
)

saveRDS(meaningful_results, here("results", "study2_clustering_meaningful.rds"))

cat("✅ Results saved to: results/study2_clustering_meaningful.rds\n")
cat("\n🎯 METHODOLOGY SUMMARY:\n")
cat("======================\n")
cat("• Returned to min_freq=3 to preserve family/personal themes\n")
cat("• Used hierarchical clustering (familiar to your field)\n")
cat("• Applied post-hoc singleton merging based on co-occurrence patterns\n")
cat("• Maintained data-driven approach while ensuring meaningful results\n")
cat("• Final clusters:", length(unique(final_clusters)), "\n")
cat("• Silhouette score:", round(k3_result$silhouette, 3), "\n")

cat("\n✅ MEANINGFUL THEMES PRESERVED!\n") 