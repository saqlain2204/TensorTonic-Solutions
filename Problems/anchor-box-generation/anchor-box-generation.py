def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride = image_size/feature_size
    anchors = []

    for x in range(feature_size):
        for y in range(feature_size):
            cx = (y + 0.5)*stride
            cy = (x + 0.5)*stride

            for s in scales:
                for r in aspect_ratios:
                    w = s*(r**0.5)
                    h = s/(r**0.5)

                    anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
            
    return anchors