def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    result = []

    for i in range(len(X)):
        row = []
        for j in range(len(W[0])):
            total = 0
            for k in range(len(W)):
                total += X[i][k] * W[k][j]
            row.append(total + b[j])
        result.append(row)

    return result
