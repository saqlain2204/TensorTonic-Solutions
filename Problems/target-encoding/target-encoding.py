def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Write code here
    sums = {}
    counts = {}

    for category, target in zip(categories, targets):
        sums[category] = sums.get(category, 0) + target
        counts[category] = counts.get(category, 0) + 1

    means = {category: sums[category] / counts[category] for category in sums}

    return [means[category] for category in categories]
