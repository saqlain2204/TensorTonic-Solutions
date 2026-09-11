import collections
def image_histogram(image: list) -> list:
    """
    Returns a list of intensity and count pairs.
    """
    # Write code here
    c = collections.Counter()

    for i in range(len(image)):
        for j in range(len(image[i])):
            c[image[i][j]] += 1

    ans = []
    for k, v in c.items():
        ans.append([k, v])

    ans.sort(key=lambda x:x[0])
    return ans
