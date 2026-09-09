import collections
def rank_transform(values: list) -> list:
    """
    Returns the one-based average rank of every value.
    """
    # Write code here
    mapper = collections.defaultdict(list)

    for i in range(len(values)):
        mapper[values[i]].append(i)

    values.sort()
    ans = [0] * len(values)

    pos = 1
    i = 0

    while i < len(values):
        num = values[i]

        j = i
        while j < len(values) and values[j] == num:
            j += 1

        avg_rank = (pos + j) / 2
        for idx in mapper[num]:
            ans[idx] = avg_rank

        pos = j + 1
        i = j

    return ans
