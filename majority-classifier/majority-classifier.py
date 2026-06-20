import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    majority_label = np.bincount(y_train).argmax()
    return np.array([majority_label for _ in range(len(X_test))])