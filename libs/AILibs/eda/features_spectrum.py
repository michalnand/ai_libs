import numpy
import matplotlib.pyplot as plt


def features_svd_spectrum(z, n_samples, result_path):

    # 1. random subsample, if matrix is huge
    if n_samples is not None and z.shape[0] > n_samples:
        indices = numpy.random.choice(z.shape[0], n_samples, replace=False)
        z_sel = z[indices]
    else:   
        z_sel = z

    # 2. Compute SVD
    # We only need the singular values (S), so compute_uv=False saves massive overhead
    S = numpy.linalg.svd(z_sel, full_matrices=False, compute_uv=False)

    # 3. Calculate Variance Metrics
    # The variance explained by each component is proportional to the square of its singular value
    eigenvalues = S ** 2
    total_variance = numpy.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = numpy.cumsum(explained_variance_ratio)

    # Calculate effective rank (how many dimensions explain 90/95/99% of the variance)
    idx_90 = numpy.searchsorted(cumulative_variance, 0.90) + 1
    idx_95 = numpy.searchsorted(cumulative_variance, 0.95) + 1
    idx_99 = numpy.searchsorted(cumulative_variance, 0.99) + 1
    total_dims = z_sel.shape[1]

    plt.clf()
    plt.cla()

    # 4. Plot the Spectrums
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot A: Log-Scale Singular Values
    ax1.plot(S, marker='o', markersize=3, linestyle='-', color='indigo')
    ax1.set_yscale('log')
    ax1.set_title("Singular Value Spectrum")
    ax1.set_xlabel("Component Index")
    ax1.set_ylabel("Singular Value (Log Scale)")
    ax1.grid(alpha=0.3)

    # Plot B: Cumulative Explained Variance
    ax2.plot(cumulative_variance, marker='o', markersize=3, linestyle='-', color='teal')
    ax2.axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='90% Variance')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95% Variance')
    ax2.set_title("Cumulative Explained Variance")
    ax2.set_xlabel("Component Index")
    ax2.set_ylabel("Cumulative Ratio")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(result_path + "features_svd_spectrum.png", dpi=300)

    # 5. Generate Markdown Report for LLM Summarization
    md_lines = [
        "## Feature Dimensionality & SVD Spectrum Report",
        f"**Input Data Shape:** `{z_sel.shape}` (Samples: {z_sel.shape[0]}, Features: {total_dims})",
        "",
        "### Explained Variance Thresholds",
        "The number of principal components required to capture the data's variance. Lower numbers indicate feature collapse into a lower-dimensional subspace:",
        f"- **90% Variance:** {idx_90} dimensions ({idx_90/total_dims*100:.1f}%)",
        f"- **95% Variance:** {idx_95} dimensions ({idx_95/total_dims*100:.1f}%)",
        f"- **99% Variance:** {idx_99} dimensions ({idx_99/total_dims*100:.1f}%)",
        "",
        "### Top 20 Dominant Components",
        "| Rank | Singular Value | Variance Explained | Cumulative Variance |",
        "|---|---|---|---|"
    ]

    for i in range(min(20, len(S))):
        md_lines.append(f"| {i+1} | {S[i]:.3f} | {explained_variance_ratio[i]*100:.2f}% | {cumulative_variance[i]*100:.2f}% |")

    
    markdown_report = "\n".join(md_lines)

    text_file = open(result_path + "features_svd_spectrum.md", "w")
    text_file.write(markdown_report)
    text_file.close()

    return markdown_report
    

