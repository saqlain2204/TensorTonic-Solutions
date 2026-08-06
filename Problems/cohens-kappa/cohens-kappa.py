import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)

    if len(rater1) != len(rater2):
        raise ValueError("Both raters must have the same number of observations.")

    p_observed = np.mean(rater1 == rater2)

    labels = np.union1d(rater1, rater2)
    p_expected = 0.0

    for label in labels:
        p_rater1 = np.mean(rater1 == label)
        p_rater2 = np.mean(rater2 == label)
        p_expected += p_rater1 * p_rater2

    if p_expected == 1:
        return 1.0 if p_observed == 1 else 0.0

    kappa = (p_observed - p_expected) / (1 - p_expected)

    return kappa
