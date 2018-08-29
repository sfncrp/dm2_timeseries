# coding: utf-8

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
rcParams['figure.figsize'] = 16,4

# ## Dataset: IBM stock values from 1962 to 2018

series = pd.Series.from_csv('ibm_daily_1962-2018.csv', sep='\t',header=0)

(series)


df = pd.read_csv('ibm_daily_1962-2018.csv', sep='\t',header=0)


plt.rcParams['interactive'] = True

plt.rcParams['interactive']

plt.rcParams['backend']


plt.plot(series)

fig, ax = plt.subplots()
ax.plot(series)
ax.grid()

ax.plot(series["Open"])


fig.show()

# ## Split the time series into years (57 series)
# and plot normalized time series

# In[ ]:


from pandas import TimeGrouper
from pandas import DataFrame
groups = series.groupby(TimeGrouper('Y'))

pd.Grouper()

df.groupby(Grouper(key = ))

df.index = pd.to_datetime(df.index)
out = df.groupby(df.index.year)

out.get_group(2000)

(out)


out.max()


print(out )

type(df.index[0])

df.index.year


mm = DataFrame()
for name, group in groups:
    norm_values = (group.values - np.mean(group.values)) / np.std(group.values)
    # Padding performed to have series of same length -- important only for plotting the matrix
    pad_values = np.pad(norm_values,(0,365-len(norm_values)),'constant',constant_values=-3)
    mm["%s" % (name.year)] = pad_values
plt.matshow(mm.T, interpolation=None, aspect='auto')


# ## Split the time series into months (676 series)
# and plot normalized time series

# In[ ]:

series = pd.Series.from_csv('ibm_daily_1962-2018.csv', sep='\t',header=0)
from pandas import TimeGrouper
from pandas import DataFrame
groups = series.groupby(TimeGrouper('Y'))

mm = DataFrame()
for name, group in groups:
    norm_values = (group.values - mean(group.values)) / std(group.values)
    # Padding performed to have series of same length -- important only for plotting the matrix
    pad_values = np.pad(norm_values,(0,31-len(norm_values)),'constant',constant_values=-3)
    mm["%s-%s" % (name.year,name.month)] = pad_values
matshow(mm.T, interpolation=None, aspect='auto')

###########################################################

import math

prova = math.inf

def dtw_band(x, y, dist, band=inf):
    """
    Computes Dynamic Time Warping (DTW) of two sequences with Sakoe-Chiba band.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int band: size of Sakow-Chiba band (default=inf)

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

# Function for inferring the optima path (general case)
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


def dtw(x, y, dist):
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

# Function for inferring the optima path (general case)
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


assert()

 if 3==3:
    print("ok")

