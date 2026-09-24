def differencing(series: list, order: int) -> list:
    """
    Returns the series after the requested differencing order.
    """
    # Write code here
    for _ in range(order):
        series = [series[i] - series[i - 1] for i in range(1, len(series))]
    
    return series
