def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    # Write code here
    ans = []

    for i in range(1, len(series)):
        try:
            ans.append((series[i] - series[i-1])/series[i-1])
        except ZeroDivisionError:
            ans.append(0.0)

    return ans
