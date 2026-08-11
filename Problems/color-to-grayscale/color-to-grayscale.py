def color_to_grayscale(images):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    ans = []

    for image in images:
        grayscale_image = []

        for pixel in image:
            R, G, B = pixel
            grayscale_image.append(0.299 * R + 0.587 * G + 0.114 * B)

        ans.append(grayscale_image)

    return ans
