def iou(box_a: list, box_b: list) -> float:
    """
    Returns IoU as a float.
    """
    # Write code here
    x1, y1, x2, y2 = box_a
    X1, Y1, X2, Y2 = box_b

    ix1 = max(x1, X1)
    iy1 = max(y1, Y1)
    ix2 = min(x2, X2)
    iy2 = min(y2, Y2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = (x2 - x1) * (y2 - y1)
    area_b = (X2 - X1) * (Y2 - Y1)

    union = area_a + area_b - intersection

    return intersection / union if union else 0.0
