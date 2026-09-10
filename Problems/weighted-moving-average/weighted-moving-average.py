def weighted_moving_average(values: list, weights: list) -> list:
    """
    Returns the weighted average of every complete window.
    """
    # Write code here

    i = 0
    j = 0

    N = len(values)
    window_length = len(weights)
    temp = 0
    ans = []
    window_ptr = 0
    
    while j < N:
        while j - i + 1 > window_length:
            i += 1

        if j - i + 1 == window_length:
            temp = 0
            for k in range(window_length):
                temp += values[i + k] * weights[k]
            ans.append(temp/sum(weights))

        j += 1

    return ans
