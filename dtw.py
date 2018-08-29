###########################################################
from numpy import array, zeros, argmin, inf, ndim
#from scipy.spatial.distance import cdist

def dtw(x, y, dist, constraint = None, band = inf, coeff = inf):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param str constraint: type of constraint, use 'SC' for Sakoe-Chiba,
    'Itakura' for parallelogram constraint, default 'None' for no constraints
    :param int band: size of Sakoe-Chiba band (default=inf)
    :param float coeff: angular coefficient of parallelogram (default=inf)
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
        
    if constraint == "SC":
        return dtw_band(x, y, dist, band = band)
    elif constraint == "Itakura":
        return dtw_itakura(x, y, dist, coeff)
    elif constraint == None:
        return dtw_basic(x, y, dist)
    else:
        print( "Error: available constraints (None, 'SC', 'I')" ) 
        return

    
    
def dtw_basic(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    
    # D0 = D1 = matrix of point-to-point costs
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view (hide first column and first row)
    
    # Fill the point-to-point costs matrix
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
            
    # C = matrix of optimal paths costs
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    
    # Infer the path from matrix C
    if len(x)==1:
        path = zeros(len(y)), range(len(y))  # special case 1
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))  # special case 2
    else:
        path = _traceback(D0)  # general case
        
    return D1[-1, -1], C, D1, path

# Function for inferring the optimal path (general case)
# Starts from last cell and goes backward...
def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def dtw_band(x, y, dist, band=inf):
    """
    Computes Dynamic Time Warping (DTW) of two sequences with Sakoe-Chiba band.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int band: size of Sakoe-Chiba band (default=inf)

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    
    # D0 = D1 = matrix of point-to-point costs
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view (hide first column and first row)
    
    # Fill the point-to-point costs matrix
    # Effect of bands: cells farther than "band" from diagonal have "inf" cost
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j]) if abs(i-j)<band else inf
            
    # C = matrix of optimal paths costs
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    
    # Infer the path from matrix C
    if len(x)==1:
        path = zeros(len(y)), range(len(y))  # special case 1
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))  # special case 2
    else:
        path = _traceback(D0)  # general case
        
    return D1[-1, -1], C, D1, path


def dtw_itakura(x, y, dist, coeff=inf):
    """
    Computes Dynamic Time Warping (DTW) of two sequences with Itakura Parallelogram constraints

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param float coeff: angular coefficient of parallelogram (default=inf)

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j]) if abs(j-i) < (min(i,j,r-i,c-j)+1)*coeff else inf
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


###########################################################
# visualization

def nice_table(cost_matrix, title, first_timeseries, second_timeseries):
    import seaborn as sns
    import pandas as pd
    from numpy import zeros_like
    df = pd.DataFrame(cost_matrix.transpose().astype(int))[::-1]
    df.columns = first_timeseries
    df.index = second_timeseries[::-1]
    mask = zeros_like(df)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if(array(df)[i][j] == -1):
                mask[i][j] = True
    ##sns.set_context('notebook', font_scale=2.5)
    ax = sns.heatmap(df, annot=True, fmt="d", cbar=False, mask=mask)
    ax.set_title(title)


def matrix_and_best_path(mymatrix,path):
    import seaborn as sns
    sns.reset_orig()
    import matplotlib.pyplot as plt
    #sns.set_context('notebook', font_scale=2.5)
    plt.imshow(mymatrix.T, origin='lower', interpolation='nearest')
    plt.title('Optimal path')
    plt.plot(path[0], path[0], 'c-')
    plt.plot(path[0], path[1], 'y')
    plt.plot(path[0], path[1], 'ro')
    plt.xlim(-0.5,mymatrix.shape[0]-0.5)
    plt.ylim(-0.5,mymatrix.shape[1]-0.5)

def alignment_plot(path, ys1, ys2, yshift = 10):
    # :param  yshift: shifts y a bit to make the plot more readable
    for (i,j) in zip(path[0],path[1]):
        col = 'r-' if i == j else 'y-'
        plt.plot([ i, j ] , [ ys1[i], ys2[j]+yshift ], col)
    plt.xlim(0.5,max(len(ys1),len(ys2))+0.5)
    plt.plot(ys1)
    plt.plot(ys2+yshift)


    

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    plt.rcParams['interactive'] = True
    plt.rcParams['interactive']
    plt.rcParams['figure.figsize'] = 10,5

    x = array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0])
    x = array([1, 2, 4, 2, 1, 2, 0])
    y = array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0])

    plt.plot(x, 'b-')
    plt.plot(x, 'bo')
    plt.plot(y, 'r-')
    plt.plot(y, 'ro')

    def distance(x,y):
        return abs(x-y)

    (dist, cost, acc, path) = dtw(x, y, distance)

    plt.close()
    nice_table(cost, 'Point-to-point costs', x, y)
    plt.close()
    nice_table(acc, 'Cumulative costs', x, y)


    path


    for coordinates in zip(path[0],path[1]):
        print(coordinates)


    plt.close()
    matrix_and_best_path(acc,path)
    plt.tight_layout()

    matrix_and_best_path(cost,path)
