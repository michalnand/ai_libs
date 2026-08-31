import matplotlib.pyplot as plt
import numpy
import AILibs
import os


def forest_save_detector_prediction_distribution(result_path, y_gt, y_pred, threshold = 0.5):
    tn_mask  = y_gt < threshold
    tp_mask   = y_gt >= threshold 

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y_pred[tn_mask], bins=80, alpha=0.6, label="True Negative", color="steelblue", density=True)
    ax.hist(y_pred[tp_mask],  bins=80, alpha=0.6, label="True Positive", color="tomato", density=True)
    ax.axvline(threshold, color="grey", linestyle="--", label="Threshold = " + str(threshold))
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "prediction_distribution.png"), dpi=300)
    plt.close(fig)



def forest_save_detector_cm(result_path, metrics,):
    cm = numpy.array([[metrics["tn"], metrics["fp"]],
                        [metrics["fn"], metrics["tp"]]])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Fraud"])
    ax.set_yticklabels(["Normal", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"), dpi=300)
    plt.close(fig)



def forest_save_detector_metrics(result_path, config, metrics):

    # --- Plot 3: Detection Metrics ---
    metric_names  = ["accuracy", "precision", "recall", "f1_score", "mcc", "balanced_accuracy"]
    metric_values = [metrics.get(m, 0.0) for m in metric_names]

    


    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(metric_names, metric_values, color="steelblue", edgecolor="white")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Value")
    ax.set_title("Metrics")

    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "detection_metrics.png"), dpi=300)
    plt.close(fig)

    # 4. Generate and save the Markdown report
    md_file = os.path.join(result_path, "results.md")
    
    # Safely extract dataset shapes 

    x, _ = config.dataset_train
    x_test, _ = config.dataset_test

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Model Evaluation Results\n\n")
        
        f.write("## Dataset Information\n")
        f.write(f"- **Training Samples:** {len(x)}\n")
        f.write(f"- **Testing Samples:** {len(x_test)}\n")
        f.write(f"- **Feature Shape:** {str(x[0].shape)}\n\n")

        f.write("## Hyperparameters\n")
        f.write("| Parameter | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| `batch_size` | {config.batch_size} |\n")
        f.write(f"| `num_trees` | {config.num_trees} |\n")
        f.write(f"| `learning_rate` | {config.learning_rate} |\n")
        f.write(f"| `max_depth` | {config.max_depth} |\n")
        f.write(f"| `min_leaf_size` | {config.min_leaf_size} |\n")
        f.write(f"| `feature_subsample_ratio` | {config.feature_subsample_ratio} |\n")
        f.write(f"| `threshold` | {config.threshold} |\n\n")

        f.write("## Metrics\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"| **{key}** | {value:.5f} |\n")
            else:
                f.write(f"| **{key}** | {value} |\n")
        f.write("\n")

        f.write("## Visualizations\n\n")
        f.write("### Prediction Distribution\n")
        f.write("![Prediction Distribution](prediction_distribution.png)\n\n")
        
        f.write("### Confusion Matrix\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        
        f.write("### Detection Metrics\n")
        f.write("![Detection Metrics](detection_metrics.png)\n")





def forest_save_anomaly_detection_metrics(result_path, config, metrics):

    # --- Plot 3: Detection Metrics ---
    metric_names  = ["auc_roc", "auc_pr", "precision", "recall", "f1_score", "mcc", "balanced_accuracy"]
    metric_values = [metrics[m] for m in metric_names]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(metric_names, metric_values, color="steelblue", edgecolor="white")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Value")
    ax.set_title("Anomaly Detection Metrics")

    # Value labels
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    plt.tight_layout()

    plt.savefig(os.path.join(result_path, "detection_metrics.png"), dpi=300)
    plt.close(fig)

    # 4. Generate and save the Markdown report
    md_file = os.path.join(result_path, "results.md")
    
    # Safely extract dataset shapes 

    x         = config.dataset_train
    x_test, _ = config.dataset_test

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Model Evaluation Results\n\n")
        
        f.write("## Dataset Information\n")
        f.write(f"- **Training Samples:** {len(x)}\n")
        f.write(f"- **Testing Samples:** {len(x_test)}\n")
        f.write(f"- **Feature Shape:** {str(x[0].shape)}\n\n")

        f.write("## Hyperparameters\n")
        f.write("| Parameter | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| `batch_size` | {config.batch_size} |\n")
        f.write(f"| `num_trees` | {config.num_trees} |\n")
        f.write(f"| `min_leaf_size` | {config.min_leaf_size} |\n")
        f.write(f"| `feature_subsample_ratio` | {config.feature_subsample_ratio} |\n")
        f.write(f"| `projection_dim` | {config.projection_dim} |\n")
        f.write(f"| `threshold` | {config.threshold} |\n\n")


        

        f.write("## Metrics\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"| **{key}** | {value:.5f} |\n")
            else:
                f.write(f"| **{key}** | {value} |\n")
        f.write("\n")

        f.write("## Visualizations\n\n")
        f.write("### Prediction Distribution\n")
        f.write("![Prediction Distribution](prediction_distribution.png)\n\n")
        
        f.write("### Confusion Matrix\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        
        f.write("### Detection Metrics\n")
        f.write("![Detection Metrics](detection_metrics.png)\n")

            