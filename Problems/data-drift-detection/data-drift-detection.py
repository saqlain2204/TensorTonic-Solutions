def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    categories = set(reference_counts) | set(production_counts)

    ref_total = sum(reference_counts)
    prod_total = sum(production_counts)

    if ref_total == 0 or prod_total == 0:
        raise ValueError("Input distributions must have non-zero total counts.")

    tvd = 0.5 * sum(
        abs(r / ref_total - p / prod_total)
        for r, p in zip(reference_counts, production_counts)
    )

    return {
        "score": tvd,
        "drift_detected": tvd > threshold
    }