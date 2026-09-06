import os
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
    
def stats(x):
    return (
        np.mean(x), 
        np.std(x), 
        np.percentile(x, 5), 
        np.percentile(x, 95)
    )

def features_similarity(za, zb, result_path):
    # Ensure result_path ends with a slash for file saving
    if not result_path.endswith('/') and not result_path.endswith('\\'):
        result_path += '/'
        
    num_samples, num_features = za.shape

    # 1. Negative Sampling
    random_shuffled_indices = np.random.permutation(num_samples)
    zb_neg = zb[random_shuffled_indices]

    # 2. Finalize Similarity / Distance Measuring (Numpy Syntax)
    # Cosine Similarity: (A dot B) / (||A|| * ||B||)
    norm_za = np.linalg.norm(za, axis=-1) + 1e-10
    norm_zb = np.linalg.norm(zb, axis=-1) + 1e-10
    norm_zb_neg = np.linalg.norm(zb_neg, axis=-1) + 1e-10

    cos_pos = (za * zb).sum(axis=-1) / (norm_za * norm_zb)
    cos_neg = (za * zb_neg).sum(axis=-1) / (norm_za * norm_zb_neg)

    # Euclidean Distance (L2 norm)
    # Note: Using standard L2 distance here. If you prefer Squared L2, remove np.sqrt()
    euclidean_pos = np.sqrt(((za - zb)**2).sum(axis=-1))
    euclidean_neg = np.sqrt(((za - zb_neg)**2).sum(axis=-1))

    # ==========================================
    # 3. Plotting
    # ==========================================
    
    # --- Cosine Similarity Histogram ---
    plt.figure(figsize=(10, 6)) 
    bins_cos = np.linspace(-1, 1, 100)
    plt.hist(cos_pos, bins=bins_cos, alpha=0.8, color='lightcoral', label='Positive (Match)', density=True)
    plt.hist(cos_neg, bins=bins_cos, alpha=0.8, color='royalblue', label='Negative (Random)', density=True)
    plt.title("Cosine Similarity Distributions of Feature Pairs")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")   
    plt.xlim(-1.0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(result_path + "features_histogram_cosine.png", dpi=300)
    plt.close()

    # --- Euclidean Distance Histogram ---
    plt.figure(figsize=(10, 6)) 
    # Dynamically scale bins for distances (starts at 0)
    max_dist = max(np.max(euclidean_pos), np.max(euclidean_neg))
    bins_euc = np.linspace(0, max_dist, 100)
    plt.hist(euclidean_pos, bins=bins_euc, alpha=0.8, color='lightcoral', label='Positive (Match)', density=True)
    plt.hist(euclidean_neg, bins=bins_euc, alpha=0.8, color='royalblue', label='Negative (Random)', density=True)
    plt.title("Euclidean Distances Distributions of Feature Pairs")
    plt.xlabel("Euclidean Distance (L2)")
    plt.ylabel("Density")
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(result_path + "features_histogram_distances.png", dpi=300)
    plt.close()

    # --- UMAP & HDBSCAN Clustering ---
    print("Computing UMAP and HDBSCAN...")
    # Reduce to 2D for visualization and clustering speed
    reducer = umap.UMAP(n_components=2, random_state=42)
    za_2d = reducer.fit_transform(za)

    # Cluster the 2D representations
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean')
    cluster_labels = clusterer.fit_predict(za_2d)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    noise_ratio = (n_noise / num_samples) * 100

    plt.figure(figsize=(10, 8))
    # 'viridis' or 'Spectral' cmap works well; noise (-1) will be distinctly colored
    scatter = plt.scatter(za_2d[:, 0], za_2d[:, 1], c=cluster_labels, cmap='Spectral', s=5, alpha=0.6)
    plt.title(f"UMAP Projection (HDBSCAN: {n_clusters} clusters, {noise_ratio:.1f}% noise)")
    plt.colorbar(scatter, label="Cluster ID (-1 is Noise)")
    plt.tight_layout()
    plt.savefig(result_path + "features_umap_projection.png", dpi=300)
    plt.close()

    # ==========================================
    # 4. Markdown Report Generation
    # ==========================================
    
    # Feature Magnitudes
    z_mag = (za**2).mean(axis=1)
    mag_mean, mag_std, mag_p5, mag_p95 = stats(z_mag)

    # Compile the Markdown lines correctly
    md_lines = [
        "# Features Similarity & Manifold Report",
        "",
        "## 1. Dataset Dimensions",
        f"- **Num Samples:** `{num_samples}`",
        f"- **Num Dims:** `{num_features}`",
        "",
        "## 2. Feature Magnitude (za)",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean | {mag_mean:.4f} |",
        f"| Std Dev | {mag_std:.4f} |",
        f"| 5th Percentile | {mag_p5:.4f} |",
        f"| 95th Percentile | {mag_p95:.4f} |", 
        "",
        "## 3. Manifold Shape (UMAP + HDBSCAN)",
        f"- **Identified Clusters:** `{n_clusters}`",
        f"- **Noise Points (Outliers):** `{n_noise}` ({noise_ratio:.2f}% of total samples)",
        "*(Note: Points labeled as -1 by HDBSCAN are considered unstructured noise)*",
        "",
        "## 4. Distance & Similarity Distributions",
        "| Metric | Pair Type | Mean | Std Dev | 5th Percentile | 95th Percentile |",
        "|---|---|---|---|---|---|"
    ]

    # Unpack stats for the table
    c_pos_stats = stats(cos_pos)
    c_neg_stats = stats(cos_neg)
    e_pos_stats = stats(euclidean_pos)
    e_neg_stats = stats(euclidean_neg)

    # Add Cosine Table rows
    md_lines.append(f"| **Cosine** | Positive | {c_pos_stats[0]:.4f} | {c_pos_stats[1]:.4f} | {c_pos_stats[2]:.4f} | {c_pos_stats[3]:.4f} |")
    md_lines.append(f"| **Cosine** | Negative | {c_neg_stats[0]:.4f} | {c_neg_stats[1]:.4f} | {c_neg_stats[2]:.4f} | {c_neg_stats[3]:.4f} |")
    
    # Add Euclidean Table rows
    md_lines.append(f"| **Euclidean** | Positive | {e_pos_stats[0]:.4f} | {e_pos_stats[1]:.4f} | {e_pos_stats[2]:.4f} | {e_pos_stats[3]:.4f} |")
    md_lines.append(f"| **Euclidean** | Negative | {e_neg_stats[0]:.4f} | {e_neg_stats[1]:.4f} | {e_neg_stats[2]:.4f} | {e_neg_stats[3]:.4f} |")

    # Combine into final string
    markdown_report = "\n".join(md_lines)

    # Save Markdown File
    with open(result_path + "features_report.md", "w") as text_file:
        text_file.write(markdown_report)

    return markdown_report