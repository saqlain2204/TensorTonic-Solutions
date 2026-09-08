def jaccard_similarity(set_a: list, set_b: list) -> float:
    """
    Returns the Jaccard similarity of the two item collections.
    """
    # Write code here
    set_a = set(set_a)
    set_b = set(set_b)

    try:
        return len(set_a & set_b) / len(set_a | set_b)
    except ZeroDivisionError:
        return 0.0
