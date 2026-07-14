def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    stopwords = set(stopwords)
    ans = []

    for x in tokens:
        if x not in stopwords:
            ans.append(x)

    return ans