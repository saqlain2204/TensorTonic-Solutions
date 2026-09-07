def simple_moving_average(values: list, window_size: int) -> list:
    """
    Returns the mean of every complete sliding window.
    """
    # Write code here
    i = 0
    j = 0

    N = len(values)
    temp = 0

    ans = []

    while j < N:
        temp += values[j]

        while j - i + 1 > window_size:
            temp -= values[i]
            i += 1

        if j - i + 1 == window_size:
            ans.append(temp/window_size)

        j += 1

    return ans
