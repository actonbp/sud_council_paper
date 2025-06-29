# K-MEANS VS HIERARCHICAL CLUSTERING COMPARISON
# Testing alternative clustering approaches for co-occurrence data

library(tidyverse)
library(tidytext)
library(here)
library(cluster)
library(SnowballC)
library(widyr)

cat("=== K-MEANS VS HIERARCHICAL CLUSTERING COMPARISON ===\n")
cat("Testing data-driven alternatives to hierarchical clustering\n\n")

# Load validated data
stems_with_session <- read_csv(here("data", "focus_group_tokens_preprocessed.csv"), show_col_types = FALSE)

# Define SUD terms (consistent with validation)
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

# Use validated parameters
word_frequencies <- sud_tokens %>%
  count(word_stem, sort = TRUE) %>%
  filter(n >= 5)

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

cat("📊 DATA OVERVIEW:\n")
cat("=================\n")
cat("Matrix dimensions:", nrow(cooccur_matrix), "x", ncol(cooccur_matrix), "\n")
cat("Matrix sparsity:", round(sum(cooccur_matrix == 0) / length(cooccur_matrix), 3), "\n")
cat("Words analyzed:", length(all_words), "\n\n")

# Store results for comparison
comparison_results <- list()

# === METHOD 1: HIERARCHICAL CLUSTERING (BASELINE) ===
cat("🌳 METHOD 1: HIERARCHICAL CLUSTERING (BASELINE)\n")
cat("===============================================\n")

hierarchical_results <- list()
for (k in 2:5) {
  tryCatch({
    dist_matrix <- dist(cooccur_matrix, method = "euclidean")
    hc <- hclust(dist_matrix, method = "ward.D2")
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
    
    # Calculate silhouette if more than 1 cluster and valid
    if (k > 1 && length(unique(clusters)) > 1) {
      sil <- silhouette(clusters, dist_matrix)
      avg_sil <- mean(sil[, 3])
    } else {
      avg_sil <- NA
    }
    
    hierarchical_results[[paste0("k", k)]] <- list(
      k = k,
      clusters = clusters,
      wss = wss,
      silhouette = avg_sil,
      singleton_clusters = sum(table(clusters) == 1),
      cluster_sizes = table(clusters)
    )
    
    cat("k =", k, ": WSS =", round(wss, 1), 
        ", Silhouette =", round(avg_sil, 3), 
        ", Singletons =", sum(table(clusters) == 1),
        ", Sizes:", paste(table(clusters), collapse = ", "), "\n")
  }, error = function(e) {
    cat("k =", k, ": ERROR -", e$message, "\n")
  })
}

best_hierarchical <- hierarchical_results[[which.max(sapply(hierarchical_results, function(x) 
  ifelse(is.na(x$silhouette), -1, x$silhouette)))]]
comparison_results[["hierarchical"]] <- best_hierarchical

cat("Best hierarchical: k =", best_hierarchical$k, 
    "with silhouette =", round(best_hierarchical$silhouette, 3), "\n\n")

# === METHOD 2: K-MEANS CLUSTERING ===
cat("🎯 METHOD 2: K-MEANS CLUSTERING\n")
cat("===============================\n")

kmeans_results <- list()
for (k in 2:5) {
  tryCatch({
    set.seed(123)
    km <- kmeans(cooccur_matrix, centers = k, nstart = 25, iter.max = 100)
    
    # Calculate silhouette
    if (k > 1 && length(unique(km$cluster)) > 1) {
      dist_matrix <- dist(cooccur_matrix)
      sil <- silhouette(km$cluster, dist_matrix)
      avg_sil <- mean(sil[, 3])
    } else {
      avg_sil <- NA
    }
    
    kmeans_results[[paste0("k", k)]] <- list(
      k = k,
      clusters = km$cluster,
      wss = km$tot.withinss,
      silhouette = avg_sil,
      singleton_clusters = sum(table(km$cluster) == 1),
      cluster_sizes = table(km$cluster)
    )
    
    cat("k =", k, ": WSS =", round(km$tot.withinss, 1), 
        ", Silhouette =", round(avg_sil, 3), 
        ", Singletons =", sum(table(km$cluster) == 1),
        ", Sizes:", paste(table(km$cluster), collapse = ", "), "\n")
  }, error = function(e) {
    cat("k =", k, ": ERROR -", e$message, "\n")
  })
}

best_kmeans <- kmeans_results[[which.max(sapply(kmeans_results, function(x) 
  ifelse(is.na(x$silhouette), -1, x$silhouette)))]]
comparison_results[["kmeans"]] <- best_kmeans

cat("Best k-means: k =", best_kmeans$k, 
    "with silhouette =", round(best_kmeans$silhouette, 3), "\n\n")

# === METHOD 3: PAM (PARTITIONING AROUND MEDOIDS) ===
cat("🔄 METHOD 3: PAM CLUSTERING\n")
cat("===========================\n")

pam_results <- list()
for (k in 2:5) {
  tryCatch({
    dist_matrix <- dist(cooccur_matrix, method = "euclidean")
    pam_result <- pam(dist_matrix, k = k)
    
    pam_results[[paste0("k", k)]] <- list(
      k = k,
      clusters = pam_result$clustering,
      silhouette = pam_result$silinfo$avg.width,
      singleton_clusters = sum(table(pam_result$clustering) == 1),
      cluster_sizes = table(pam_result$clustering)
    )
    
    cat("k =", k, ": Silhouette =", round(pam_result$silinfo$avg.width, 3), 
        ", Singletons =", sum(table(pam_result$clustering) == 1),
        ", Sizes:", paste(table(pam_result$clustering), collapse = ", "), "\n")
  }, error = function(e) {
    cat("k =", k, ": ERROR -", e$message, "\n")
  })
}

if (length(pam_results) > 0) {
  best_pam <- pam_results[[which.max(sapply(pam_results, function(x) x$silhouette))]]
  comparison_results[["pam"]] <- best_pam
  cat("Best PAM: k =", best_pam$k, 
      "with silhouette =", round(best_pam$silhouette, 3), "\n\n")
}

# === METHOD 4: FUZZY C-MEANS ===
cat("🌊 METHOD 4: FUZZY C-MEANS\n")
cat("==========================\n")

# Check if e1071 package is available
if (requireNamespace("e1071", quietly = TRUE)) {
  library(e1071)
  
  fuzzy_results <- list()
  for (k in 2:4) {
    tryCatch({
      fuzzy_result <- cmeans(cooccur_matrix, centers = k, iter.max = 100, m = 2)
      
      # Get hard clustering assignments
      hard_clusters <- apply(fuzzy_result$membership, 1, which.max)
      
      # Calculate silhouette for hard assignments
      if (length(unique(hard_clusters)) > 1) {
        dist_matrix <- dist(cooccur_matrix)
        sil <- silhouette(hard_clusters, dist_matrix)
        avg_sil <- mean(sil[, 3])
      } else {
        avg_sil <- NA
      }
      
      fuzzy_results[[paste0("k", k)]] <- list(
        k = k,
        clusters = hard_clusters,
        silhouette = avg_sil,
        singleton_clusters = sum(table(hard_clusters) == 1),
        cluster_sizes = table(hard_clusters),
        membership_entropy = -sum(fuzzy_result$membership * log(fuzzy_result$membership + 1e-10))
      )
      
      cat("k =", k, ": Silhouette =", round(avg_sil, 3), 
          ", Singletons =", sum(table(hard_clusters) == 1),
          ", Sizes:", paste(table(hard_clusters), collapse = ", "), "\n")
    }, error = function(e) {
      cat("k =", k, ": ERROR -", e$message, "\n")
    })
  }
  
  if (length(fuzzy_results) > 0) {
    best_fuzzy <- fuzzy_results[[which.max(sapply(fuzzy_results, function(x) 
      ifelse(is.na(x$silhouette), -1, x$silhouette)))]]
    comparison_results[["fuzzy"]] <- best_fuzzy
    cat("Best Fuzzy C-means: k =", best_fuzzy$k, 
        "with silhouette =", round(best_fuzzy$silhouette, 3), "\n\n")
  }
} else {
  cat("e1071 package not available - skipping fuzzy c-means\n\n")
}

# === COMPREHENSIVE COMPARISON ===
cat("🏆 COMPREHENSIVE METHOD COMPARISON\n")
cat("==================================\n")

if (length(comparison_results) > 0) {
  comparison_table <- tibble(
    method = names(comparison_results),
    k = sapply(comparison_results, function(x) x$k),
    silhouette = sapply(comparison_results, function(x) 
      ifelse(is.null(x$silhouette) || is.na(x$silhouette), 0, x$silhouette)),
    singleton_clusters = sapply(comparison_results, function(x) x$singleton_clusters),
    max_cluster_size = sapply(comparison_results, function(x) max(x$cluster_sizes)),
    min_cluster_size = sapply(comparison_results, function(x) min(x$cluster_sizes)),
    balance_ratio = sapply(comparison_results, function(x) 
      min(x$cluster_sizes) / max(x$cluster_sizes))
  ) %>%
    arrange(desc(silhouette), singleton_clusters, desc(balance_ratio))
  
  cat("COMPARISON TABLE (ranked by silhouette score):\n")
  print(comparison_table)
  
  # Select best method
  best_method <- comparison_table$method[1]
  best_result <- comparison_results[[best_method]]
  
  cat("\n🥇 RECOMMENDED METHOD:", toupper(best_method), "\n")
  cat("==================", paste(rep("=", nchar(best_method) + 5), collapse = ""), "\n")
  cat("Optimal k:", best_result$k, "\n")
  cat("Silhouette score:", round(best_result$silhouette, 3), "\n")
  cat("Singleton clusters:", best_result$singleton_clusters, "\n")
  cat("Cluster balance ratio:", round(comparison_table$balance_ratio[1], 3), "\n")
  cat("Cluster sizes:", paste(best_result$cluster_sizes, collapse = ", "), "\n\n")
  
  # Show detailed cluster composition
  cluster_composition <- tibble(
    word = all_words,
    cluster = best_result$clusters
  ) %>%
    left_join(word_frequencies, by = c("word" = "word_stem")) %>%
    arrange(cluster, desc(n))
  
  cat("DETAILED CLUSTER COMPOSITION:\n")
  for (cluster_id in sort(unique(best_result$clusters))) {
    words_in_cluster <- cluster_composition %>%
      filter(cluster == cluster_id) %>%
      arrange(desc(n))
    
    cat("Cluster", cluster_id, "(", nrow(words_in_cluster), "words):\n")
    for (i in 1:nrow(words_in_cluster)) {
      cat("  -", words_in_cluster$word[i], "(frequency:", words_in_cluster$n[i], ")\n")
    }
    cat("\n")
  }
  
  # Thematic interpretation
  cat("🎨 THEMATIC INTERPRETATION:\n")
  cat("===========================\n")
  
  # Analyze each cluster thematically
  for (cluster_id in sort(unique(best_result$clusters))) {
    words_in_cluster <- cluster_composition %>%
      filter(cluster == cluster_id) %>%
      pull(word)
    
    # Determine theme based on words
    if (any(c("counselor", "therapist", "therapi", "counsel", "clinic") %in% words_in_cluster)) {
      theme <- "Professional Support & Treatment"
      description <- "Terms related to counselors, therapists, and professional treatment services"
    } else if (any(c("substanc", "abus", "addict", "alcohol", "drug") %in% words_in_cluster)) {
      theme <- "Substance Use & Problems"
      description <- "Terms related to substances, abuse, addiction, and related problems"
    } else if (any(c("struggl", "battl", "depend", "pre") %in% words_in_cluster)) {
      theme <- "Challenges & Dependencies"
      description <- "Terms related to personal struggles, battles, and dependencies"
    } else {
      theme <- paste0("Mixed Theme ", cluster_id)
      description <- paste("Various terms including:", paste(head(words_in_cluster, 3), collapse = ", "))
    }
    
    cat("Cluster", cluster_id, ":", theme, "\n")
    cat("  Description:", description, "\n")
    cat("  Representative words:", paste(head(words_in_cluster, 5), collapse = ", "), "\n\n")
  }
  
  # Save comprehensive results
  final_comparison_results <- list(
    comparison_table = comparison_table,
    best_method = best_method,
    best_result = best_result,
    cluster_composition = cluster_composition,
    all_method_results = comparison_results,
    methodology_notes = list(
      matrix_sparsity = sum(cooccur_matrix == 0) / length(cooccur_matrix),
      words_analyzed = length(all_words),
      cooccurrence_pairs = nrow(word_pairs),
      recommendation = paste("Use", best_method, "clustering with k =", best_result$k),
      advantages = case_when(
        best_method == "kmeans" ~ "Better handling of spherical clusters, less sensitive to outliers",
        best_method == "hierarchical" ~ "Deterministic results, good for nested cluster structure",
        best_method == "pam" ~ "Robust to outliers, works well with distance matrices",
        best_method == "fuzzy" ~ "Allows soft cluster assignments, handles overlapping themes",
        TRUE ~ "Data-driven optimization"
      )
    )
  )
  
  saveRDS(final_comparison_results, here("results", "study2_clustering_comparison.rds"))
  
  cat("💾 RESULTS SAVED:\n")
  cat("=================\n")
  cat("File: results/study2_clustering_comparison.rds\n")
  cat("Recommended approach:", best_method, "clustering with k =", best_result$k, "\n")
  cat("Primary advantage:", final_comparison_results$methodology_notes$advantages, "\n")
  
  cat("\n✅ CLUSTERING METHOD COMPARISON COMPLETE!\n")
  cat("Your analysis now has a data-driven, scientifically defensible clustering approach.\n")
  
} else {
  cat("❌ No successful clustering methods found.\n")
} 