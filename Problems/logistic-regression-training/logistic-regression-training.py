import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    w = np.zeros(X.shape[1])
    b = 0

    for epoch in range(steps):
        z = X @ w + b
        y_pred = _sigmoid(z)

        loss = y_pred - y
        dw = (X.T @ loss)//X.shape[0]
        db = np.mean(loss)

        w -= lr*dw
        b -= lr*db

    return w, b
        