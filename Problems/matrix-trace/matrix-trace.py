import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    ans = 0
    rows = len(A)


    for i in range(rows):
        for  j in range(rows):
            if i == j:
                ans += A[i][j]

    return ans
