import collections
def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    words_to_count = collections.defaultdict(int)

    for arr in sentences:
        for x in arr:
            words_to_count[x] += 1

    return words_to_count