def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    # Write code here
    items =  set()

    for items_list in recommendations:
        for item in items_list:
            items.add(item)

    return len(items)/n_items
