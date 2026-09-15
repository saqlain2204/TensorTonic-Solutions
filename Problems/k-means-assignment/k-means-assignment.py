def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    # Write code here
    ans = []

    for point in points:
        dist = float('inf')
        temp_ans = -1

        for i, centroid in enumerate(centroids):
            temp = sum((x - y) ** 2 for x, y in zip(point, centroid))

            if temp < dist:
                dist = temp
                temp_ans = i

        ans.append(temp_ans)

    return ans
