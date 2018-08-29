

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['interactive'] = True

df = pd.read_csv('ibm_daily_1962-2018.csv', sep='\t',header=0, 
                 index_col = 0)


def f1(x):
    return( '{:.2f}'.format(x) )


fs = [f1 for i in range(df.shape[1])]

print(df[-10:].to_latex(formatters = fs))


# fig, ax = plt.subplots()
# ax.plot(df["Open"])
# ax.grid()
