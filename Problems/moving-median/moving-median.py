def moving_median(values: list, window_size: int) -> list:
    """
    Returns the median of every complete sliding window.
    """
    # Write code here
    i = 0
    j = 0
    N = len(values)
    ans = []

    temp = []

    while j < N:
        temp.append(values[j])

        while j - i + 1 > window_size:
            temp.pop(0)
            i += 1

        if j - i + 1 == window_size:
            new_temp = sorted(temp)

            middle = window_size // 2

            if window_size % 2:
                ans.append(float(new_temp[middle]))
            else:
                mean = (new_temp[middle] + new_temp[middle - 1]) / 2
                ans.append(float(mean))

        j += 1

    return ans
