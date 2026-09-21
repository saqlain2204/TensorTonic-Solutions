def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """
    # Write code here
    ema = []
    ema = [values[0]]

    for i in range(1, len(values)):
        val = alpha*values[i] + (1-alpha)*ema[i-1]
        ema.append(val)

    return ema
