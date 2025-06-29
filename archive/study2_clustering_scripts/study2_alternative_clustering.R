# ALTERNATIVE CLUSTERING METHODS FOR STUDY 2
# Testing k-means, spectral, density-based, and other approaches

library(tidyverse)
library(tidytext)
library(here)
library(cluster)
library(SnowballC)
library(widyr)
library(mixtools)  # For Gaussian mixture models
library(dbscan)    # For DBSCAN
library(kernlab)   # For spectral clustering
library(NMF)       # For non-negative matrix factorization

cat("=== ALTERNATIVE CLUSTERING METHODS ANALYSIS ===\n")
cat("Testing multiple algorithms for optimal results\n\n")

# Load the validated data
stems_with_session <- read_csv(here("data", "focus_group_tokens_preprocessed.csv"), show_col_types = FALSE)

# Define SUD terms (consistent approach)
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

# Use validated parameters (min_freq=5)
word_frequencies <- sud_tokens %>%
  count(word_stem, sort = TRUE) %>%
  filter(n >= 5)

# Create co-occurrence matrix (same as validated approach)
word_pairs <- sud_tokens %>%
  filter(word_stem %in% word_frequencies$word_stem) %>%
  pairwise_count(word_stem, response_id, sort = TRUE) %>%
  filter(n >= 2)

# Create symmetric co-occurrence matrix
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
cat("Number of words:", length(all_words), "\n\n")

# Store clustering results
clustering_results <- list()

# === METHOD 1: K-MEANS CLUSTERING ===
cat("🎯 METHOD 1: K-MEANS CLUSTERING\n")
cat("===============================\n")

kmeans_results <- list()
for (k in 2:5) {
  tryCatch({
    set.seed(123)
    km <- kmeans(cooccur_matrix, centers = k, nstart = 25, iter.max = 100)
    
    # Calculate silhouette score if possible
    if (k <= nrow(cooccur_matrix) && k > 1) {
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
      cluster_sizes = table(km$cluster)
    )
    
    cat("k =", k, ": WSS =", round(km$tot.withinss, 1), 
        ", Silhouette =", round(avg_sil, 3), 
        ", Sizes:", paste(table(km$cluster), collapse = ", "), "\n")
  }, error = function(e) {
    cat("k =", k, ": ERROR -", e$message, "\n")
  })
}

# Select best k-means result
if (length(kmeans_results) > 0) {
  best_kmeans <- kmeans_results[[which.max(sapply(kmeans_results, function(x) x$silhouette))]]
  clustering_results[["kmeans"]] <- best_kmeans
  cat("Best k-means: k =", best_kmeans$k, "with silhouette =", round(best_kmeans$silhouette, 3), "\n\n")
}

# === METHOD 2: SPECTRAL CLUSTERING ===
cat("🌈 METHOD 2: SPECTRAL CLUSTERING\n")
cat("================================\n")

spectral_results <- list()
for (k in 2:4) {
  tryCatch({
    # Convert to similarity matrix (higher values = more similar)
    sim_matrix <- cooccur_matrix / max(cooccur_matrix)
    
    spec <- specc(sim_matrix, centers = k)
    
    # Calculate silhouette
    dist_matrix <- dist(cooccur_matrix)
    sil <- silhouette(spec@.Data, dist_matrix)
    avg_sil <- mean(sil[, 3])
    
    spectral_results[[paste0("k", k)]] <- list(
      k = k,
      clusters = as.numeric(spec@.Data),
      silhouette = avg_sil,
      cluster_sizes = table(spec@.Data)
    )
    
    cat("k =", k, ": Silhouette =", round(avg_sil, 3), 
        ", Sizes:", paste(table(spec@.Data), collapse = ", "), "\n")
  }, error = function(e) {
    cat("k =", k, ": ERROR -", e$message, "\n")
  })
}

if (length(spectral_results) > 0) {
  best_spectral <- spectral_results[[which.max(sapply(spectral_results, function(x) x$silhouette))]]
  clustering_results[["spectral"]] <- best_spectral
  cat("Best spectral: k =", best_spectral$k, "with silhouette =", round(best_spectral$silhouette, 3), "\n\n")
}

# === METHOD 3: DBSCAN (DENSITY-BASED) ===
cat("🔍 METHOD 3: DENSITY-BASED CLUSTERING (DBSCAN)\n")
cat("===============================================\n")

tryCatch({
  # Test different epsilon values
  dist_matrix <- dist(cooccur_matrix, method = "euclidean")
  eps_values <- c(2, 3, 5, 8, 10)
  
  dbscan_results <- list()
  for (eps in eps_values) {
    db <- dbscan(cooccur_matrix, eps = eps, minPts = 2)
    n_clusters <- length(unique(db$cluster[db$cluster != 0]))
    n_noise <- sum(db$cluster == 0)
    
    if (n_clusters >= 2 && n_noise < nrow(cooccur_matrix) * 0.5) {
      # Calculate silhouette for non-noise points
      non_noise_idx <- db$cluster != 0
      if (sum(non_noise_idx) > 1 && n_clusters > 1) {
        sil <- silhouette(db$cluster[non_noise_idx], 
                         as.dist(as.matrix(dist_matrix)[non_noise_idx, non_noise_idx]))
        avg_sil <- mean(sil[, 3])
      } else {
        avg_sil <- NA
      }
      
      dbscan_results[[paste0("eps", eps)]] <- list(
        eps = eps,
        clusters = db$cluster,
        n_clusters = n_clusters,
        n_noise = n_noise,
        silhouette = avg_sil
      )
      
      cat("eps =", eps, ": Clusters =", n_clusters, ", Noise =", n_noise, 
          ", Silhouette =", round(avg_sil, 3), "\n")
    } else {
      cat("eps =", eps, ": Too few clusters (", n_clusters, ") or too much noise (", n_noise, ")\n")
    }
  }
  
  if (length(dbscan_results) > 0) {
    best_dbscan <- dbscan_results[[which.max(sapply(dbscan_results, function(x) x$silhouette))]]
    clustering_results[["dbscan"]] <- best_dbscan
    cat("Best DBSCAN: eps =", best_dbscan$eps, "with", best_dbscan$n_clusters, "clusters\n\n")
  }
}, error = function(e) {
  cat("DBSCAN ERROR:", e$message, "\n\n")
})

# === METHOD 4: GAUSSIAN MIXTURE MODEL ===
cat("📊 METHOD 4: GAUSSIAN MIXTURE MODEL\n")
cat("===================================\n")

gmm_results <- list()
for (k in 2:4) {
  tryCatch({
    # Flatten matrix for GMM (needs vector input)
    data_for_gmm <- as.vector(cooccur_matrix)
    
    # Fit GMM
    gmm <- normalmixEM(data_for_gmm, k = k, maxit = 100, epsilon = 1e-08)
    
    # Assign clusters based on posterior probabilities
    # This is a simplified approach - normally you'd use the full matrix
    clusters <- apply(gmm$posterior, 1, which.max)
    
    # Reshape clusters back to word assignments (simplified)
    if (length(clusters) == length(cooccur_matrix)) {
      word_clusters <- rep(1:nrow(cooccur_matrix), each = ncol(cooccur_matrix))
      word_clusters <- word_clusters[1:nrow(cooccur_matrix)]  # Take diagonal
      
      gmm_results[[paste0("k", k)]] <- list(
        k = k,
        clusters = word_clusters,
        loglik = gmm$loglik,
        cluster_sizes = table(word_clusters)
      )
      
      cat("k =", k, ": Log-likelihood =", round(tail(gmm$loglik, 1), 2), 
          ", Sizes:", paste(table(word_clusters), collapse = ", "), "\n")
    }
  }, error = function(e) {
    cat("k =", k, ": ERROR -", e$message, "\n")
  })
}

if (length(gmm_results) > 0) {
  best_gmm <- gmm_results[[which.max(sapply(gmm_results, function(x) tail(x$loglik, 1)))]]
  clustering_results[["gmm"]] <- best_gmm
  cat("Best GMM: k =", best_gmm$k, "\n\n")
}

# === METHOD 5: NON-NEGATIVE MATRIX FACTORIZATION ===
cat("🔢 METHOD 5: NON-NEGATIVE MATRIX FACTORIZATION\n")
cat("==============================================\n")

nmf_results <- list()
for (k in 2:4) {
  tryCatch({
    # NMF requires non-negative values
    nmf_matrix <- pmax(cooccur_matrix, 0)
    
    # Fit NMF
    nmf_result <- nmf(nmf_matrix, rank = k, nrun = 5, seed = 123)
    
    # Extract clusters from basis matrix
    basis_matrix <- basis(nmf_result)
    clusters <- apply(basis_matrix, 1, which.max)
    
    # Calculate reconstruction error
    recon_error <- rss(nmf_result)
    
    nmf_results[[paste0("k", k)]] <- list(
      k = k,
      clusters = clusters,
      rss = recon_error,
      cluster_sizes = table(clusters)
    )
    
    cat("k =", k, ": RSS =", round(recon_error, 2), 
        ", Sizes:", paste(table(clusters), collapse = ", "), "\n")
  }, error = function(e) {
    cat("k =", k, ": ERROR -", e$message, "\n")
  })
}

if (length(nmf_results) > 0) {
  best_nmf <- nmf_results[[which.min(sapply(nmf_results, function(x) x$rss))]]
  clustering_results[["nmf"]] <- best_nmf
  cat("Best NMF: k =", best_nmf$k, "with RSS =", round(best_nmf$rss, 2), "\n\n")
}

# === COMPARISON AND SELECTION ===
cat("🏆 CLUSTERING METHOD COMPARISON\n")
cat("===============================\n")

if (length(clustering_results) > 0) {
  comparison_table <- tibble(
    method = names(clustering_results),
    k = sapply(clustering_results, function(x) x$k),
    silhouette = sapply(clustering_results, function(x) 
      ifelse(is.null(x$silhouette) || is.na(x$silhouette), 0, x$silhouette)),
    singleton_clusters = sapply(clustering_results, function(x) 
      sum(table(x$clusters) == 1)),
    balanced_clusters = sapply(clustering_results, function(x) {
      sizes <- table(x$clusters)
      min(sizes) / max(sizes)  # Balance ratio
    })
  ) %>%
    arrange(desc(silhouette), singleton_clusters, desc(balanced_clusters))
  
  cat("COMPARISON TABLE:\n")
  print(comparison_table)
  
  # Select best method
  best_method <- comparison_table$method[1]
  best_result <- clustering_results[[best_method]]
  
  cat("\n🥇 RECOMMENDED METHOD:", toupper(best_method), "\n")
  cat("=====================", paste(rep("=", nchar(best_method)), collapse = ""), "\n")
  cat("Optimal k:", best_result$k, "\n")
  cat("Silhouette score:", round(best_result$silhouette, 3), "\n")
  cat("Singleton clusters:", sum(table(best_result$clusters) == 1), "\n")
  cat("Cluster sizes:", paste(table(best_result$clusters), collapse = ", "), "\n\n")
  
  # Show cluster composition
  cluster_composition <- tibble(
    word = all_words,
    cluster = best_result$clusters
  ) %>%
    left_join(word_frequencies, by = c("word" = "word_stem")) %>%
    arrange(cluster, desc(n))
  
  cat("FINAL CLUSTER COMPOSITION:\n")
  for (cluster_id in sort(unique(best_result$clusters))) {
    words_in_cluster <- cluster_composition %>%
      filter(cluster == cluster_id) %>%
      arrange(desc(n))
    
    cat("Cluster", cluster_id, "(", nrow(words_in_cluster), "words):\n")
    for (i in 1:nrow(words_in_cluster)) {
      cat("  -", words_in_cluster$word[i], "(n=", words_in_cluster$n[i], ")\n")
    }
    cat("\n")
  }
  
  # Save results
  alternative_clustering_results <- list(
    comparison_table = comparison_table,
    best_method = best_method,
    best_result = best_result,
    cluster_composition = cluster_composition,
    all_results = clustering_results,
    data_info = list(
      matrix_dimensions = dim(cooccur_matrix),
      sparsity = sum(cooccur_matrix == 0) / length(cooccur_matrix),
      words_analyzed = length(all_words)
    )
  )
  
  saveRDS(alternative_clustering_results, here("results", "study2_alternative_clustering.rds"))
  
  cat("✅ Alternative clustering analysis complete!\n")
  cat("Results saved to: results/study2_alternative_clustering.rds\n")
  cat("\nRecommendation: Use", best_method, "clustering with k =", best_result$k, "\n")
  
} else {
  cat("❌ No successful clustering methods found.\n")
  cat("Consider adjusting parameters or data preprocessing.\n")
} 