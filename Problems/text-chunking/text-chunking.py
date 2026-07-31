def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here

    if not tokens:
        return []

    if len(tokens) <= chunk_size:
        return [tokens]

    ans = []
    step = chunk_size - overlap

    for i in range(0, len(tokens), step):
        ans.append(tokens[i:i + chunk_size])

        if i + chunk_size >= len(tokens):
            break

    return ans