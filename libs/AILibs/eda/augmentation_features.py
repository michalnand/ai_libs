import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import umap
import hdbscan

def calculate_stats(x):
    return (
        np.mean(x),
        np.std(x),
        np.percentile(x, 5),
        np.percentile(x, 95),
        len(x)
    )

def analyze_augmentation_features(z, result_path):
    """
    Analyzes multi-augmentation feature embeddings.

    Parameters:
    - z: numpy float array of shape (num_samples, num_similar, num_features)
    - result_path: str, output directory path for plots and markdown report
    """

    # 1. Metadata Extraction
    num_samples, num_similar, num_features = z.shape
    total_vectors = num_samples * num_similar

    # Flatten tensor for pairwise calculations and projection: (N*S, D)
    z_flat = z.reshape(total_vectors, num_features)
    gt_labels = np.repeat(np.arange(num_samples), num_similar)

    # 2. Pairwise Cosine and Euclidean Calculations
    # Normalize features for Cosine Similarity
    norms = np.linalg.norm(z_flat, axis=1, keepdims=True) + 1e-10
    z_norm = z_flat / norms

    # Pairwise matrices
    cos_matrix = np.dot(z_norm, z_norm.T)
    euc_matrix = cdist(z_flat, z_flat, metric='euclidean')

    # Masks for Positive (same original sample) and Negative (different samples)
    same_sample_mask = (gt_labels[:, None] == gt_labels[None, :]) & ~np.eye(total_vectors, dtype=bool)
    diff_sample_mask = (gt_labels[:, None] != gt_labels[None, :])

    # Subsample pair arrays if large to manage memory while preserving statistics
    max_pairs = 100_000
    pos_indices = np.argwhere(same_sample_mask)
    neg_indices = np.argwhere(diff_sample_mask)

    if len(pos_indices) > max_pairs:
        pos_indices = pos_indices[np.random.choice(len(pos_indices), max_pairs, replace=False)]
    if len(neg_indices) > max_pairs:
        neg_indices = neg_indices[np.random.choice(len(neg_indices), max_pairs, replace=False)]

    cos_pos = cos_matrix[pos_indices[:, 0], pos_indices[:, 1]]
    cos_neg = cos_matrix[neg_indices[:, 0], neg_indices[:, 1]]
    euc_pos = euc_matrix[pos_indices[:, 0], pos_indices[:, 1]]
    euc_neg = euc_matrix[neg_indices[:, 0], neg_indices[:, 1]]

    # --- Plotting Histograms ---
    # Cosine Histogram
    plt.figure(figsize=(10, 6))
    bins_cos = np.linspace(-1, 1, 100)
    plt.hist(cos_pos, bins=bins_cos, alpha=0.6, color='lightcoral', label='Positive (Augmentations)', density=True)
    plt.hist(cos_neg, bins=bins_cos, alpha=0.6, color='royalblue', label='Negative (Other Images)', density=True)
    plt.title("Cosine Similarity Distribution Across Augmentations")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.xlim(-1.0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(result_path + "augmentations_features_histogram_cosine.png", dpi=300)
    plt.close()

    # Euclidean Distance Histogram
    plt.figure(figsize=(10, 6))
    max_dist = max(np.max(euc_pos), np.max(euc_neg))
    bins_euc = np.linspace(0, max_dist, 100)
    plt.hist(euc_pos, bins=bins_euc, alpha=0.6, color='lightcoral', label='Positive (Augmentations)', density=True)
    plt.hist(euc_neg, bins=bins_euc, alpha=0.6, color='royalblue', label='Negative (Other Images)', density=True)
    plt.title("Euclidean Distance Distribution Across Augmentations")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(result_path + "augmentations_features_histogram_distances.png", dpi=300)
    plt.close()

    # 3. UMAP Projection & Known Cluster Analysis
    reducer = umap.UMAP(n_components=2, random_state=42)
    z_2d = reducer.fit_transform(z_flat)

    # Guided K-Means (K = num_samples, as expected)
    kmeans = KMeans(n_clusters=num_samples, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(z_flat)    

    # Unsupervised HDBSCAN for Density & Noise Detection
    hdb = hdbscan.HDBSCAN(min_cluster_size=max(5, num_similar // 4))
    hdb_labels = hdb.fit_predict(z_2d)
    n_noise = np.sum(hdb_labels == -1)
    noise_ratio = (n_noise / total_vectors) * 100

    # Optimal Cluster Matching via Hungarian Algorithm to map K-Means to Ground Truth
    cost_matrix = np.zeros((num_samples, num_samples), dtype=int)
    for gt, pr in zip(gt_labels, pred_labels):
        cost_matrix[gt, pr] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    correct_matches = cost_matrix[row_ind, col_ind].sum()
    accuracy = (correct_matches / total_vectors) * 100
    missed_count = total_vectors - correct_matches

    # Clustering Validation Metrics
    ari_score = adjusted_rand_score(gt_labels, pred_labels)
    nmi_score = normalized_mutual_info_score(gt_labels, pred_labels)

    # Inter-centroid vs Intra-cluster Distance Calculations
    sample_centroids = np.array([z_flat[gt_labels == i].mean(axis=0) for i in range(num_samples)])
    intra_spreads = [np.mean(cdist(z_flat[gt_labels == i], [sample_centroids[i]])) for i in range(num_samples)]
    avg_intra_spread = float(np.mean(intra_spreads))
    
    inter_centroid_dists = cdist(sample_centroids, sample_centroids)
    np.fill_diagonal(inter_centroid_dists, np.nan)
    avg_inter_centroid_dist = float(np.nanmean(inter_centroid_dists))

    # --- Plotting UMAP Projections ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Ground Truth UMAP
    scatter1 = ax1.scatter(z_2d[:, 0], z_2d[:, 1], c=gt_labels, cmap='tab10', s=12, alpha=0.8)
    ax1.set_title(f"Ground Truth Clusters (K={num_samples})")
    ax1.grid(alpha=0.2)

    # Predicted KMeans UMAP
    scatter2 = ax2.scatter(z_2d[:, 0], z_2d[:, 1], c=pred_labels, cmap='tab10', s=12, alpha=0.8)
    ax2.set_title(f"K-Means Predicted Clusters (Acc: {accuracy:.1f}%)")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(result_path + "augmentations_features_umap_projection.png", dpi=300)
    plt.close()

    # 4. Markdown Report Construction
    cp_mean, cp_std, cp_p5, cp_p95, cp_cnt = calculate_stats(cos_pos)
    cn_mean, cn_std, cn_p5, cn_p95, cn_cnt = calculate_stats(cos_neg)
    ep_mean, ep_std, ep_p5, ep_p95, ep_cnt = calculate_stats(euc_pos)
    en_mean, en_std, en_p5, en_p95, en_cnt = calculate_stats(euc_neg)

    md_lines = [
        "# Augmentation Feature Embedding & Cluster Report",
        "",
        "## 1. Batch & Feature Metadata",
        f"- **Num Independent Samples (Classes):** `{num_samples}`",
        f"- **Num Augmentations per Sample:** `{num_similar}`",
        f"- **Num Feature Dimensions:** `{num_features}`",
        f"- **Total Feature Vectors:** `{total_vectors}`",
        "",
        "## 2. Augmentation Similarity & Distance Distributions",
        "| Metric | Pair Type | Mean | Std Dev | 5th Percentile | 95th Percentile | Sample Count |",
        "|---|---|---|---|---|---|---|",
        f"| **Cosine Similarity** | Positive (Augmentations) | {cp_mean:.4f} | {cp_std:.4f} | {cp_p5:.4f} | {cp_p95:.4f} | {cp_cnt} |",
        f"| **Cosine Similarity** | Negative (Other Images) | {cn_mean:.4f} | {cn_std:.4f} | {cn_p5:.4f} | {cn_p95:.4f} | {cn_cnt} |",
        f"| **Euclidean Distance** | Positive (Augmentations) | {ep_mean:.4f} | {ep_std:.4f} | {ep_p5:.4f} | {ep_p95:.4f} | {ep_cnt} |",
        f"| **Euclidean Distance** | Negative (Other Images) | {en_mean:.4f} | {en_std:.4f} | {en_p5:.4f} | {en_p95:.4f} | {en_cnt} |",
        "",
        "## 3. Cluster Recovery & Manifold Analysis",
        f"- **Expected Clusters (K):** `{num_samples}`",
        f"- **Adjusted Rand Index (ARI):** `{ari_score:.4f}`",
        f"- **Normalized Mutual Information (NMI):** `{nmi_score:.4f}`",
        f"- **Cluster Assignment Accuracy:** `{accuracy:.2f}%`",
        f"- **Misclassified / Missed Augmentations:** `{missed_count}` / `{total_vectors}`",
        f"- **HDBSCAN Unstructured Noise Points:** `{n_noise}` ({noise_ratio:.2f}%)",
        "",
        "### Geometric Metrics",
        f"- **Mean Intra-Cluster Augmentation Spread:** `{avg_intra_spread:.4f}`",
        f"- **Mean Inter-Centroid Distance:** `{avg_inter_centroid_dist:.4f}`",
        f"- **Separation Ratio (Inter / Intra):** `{avg_inter_centroid_dist / (avg_intra_spread + 1e-10):.2f}x`"
    ]

    markdown_report = "\n".join(md_lines)

    with open(result_path + "augmentation_report.md", "w") as f:
        f.write(markdown_report)

    return markdown_report