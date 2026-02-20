
def format_metrics(metrics):
    """Format a flat metrics dict as a readable multi-line string."""
    lines = [f"{key}: {value}" for key, value in metrics.items()]
    return "\n".join(lines)
