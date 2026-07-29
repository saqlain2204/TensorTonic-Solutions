def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    # Write code here
    M = len(s1)
    N = len(s2)

    dp = [[-1]*(N+1) for _ in range(M+1)]
    
    def go(i, j):
        if i == len(s1):
            return len(s2) - j
        
        if j == len(s2):
            return len(s1) - i

        if dp[i][j] != -1:
            return dp[i][j]

        nothing, something = float('inf'), float('inf')
        if s1[i] == s2[j]:
            nothing = go(i+1, j+1)

        else:
            change_left = 1 + go(i+1, j)
            change_right = 1 + go(i, j+1)
            delete = 1 + go(i+1, j+1)
            something = min(change_left, change_right, delete)

        dp[i][j] = min(something, nothing)
        return dp[i][j]

    return go(0, 0)