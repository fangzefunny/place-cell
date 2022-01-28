import numpy as np 

def pearson_corr( X, Y):
    '''Pearson correlation 
    R = cov( X, Y) / (std(X)*std(Y))
    '''
    X, Y = X.reshape([-1]), Y.reshape([-1])
    return np.cov( X, Y) / ( np.std( X) * np.std( Y))

def icc(Y, icc_type="icc(3,1)"):
    """
    Reference:
    https://stackoverflow.com/questions/40965579/intraclass-correlation-in-python-module
    
    Args:
        Y: The data Y are entered as a 'table' ie. subjects are in rows and repeated
            measures in columns
        icc_type: type of ICC to calculate. (ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k)) 
    Returns:
        ICC: (np.array) intraclass correlation coefficient
    """

    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k - 1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    # MSC = SSC / dfc / n
    MSC = SSC / dfc

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == "icc(2,1)" or icc_type == 'icc(2,k)':
        if icc_type=='icc(2,k)':
            k=1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
    elif icc_type == "icc(3,1)" or icc_type == 'icc(3,k)':
        if icc_type=='icc(3,k)':
            k=1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)

    return ICC