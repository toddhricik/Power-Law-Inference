from plfit_python3_lean_pandas import *
import time
import pandas as pd
import numpy as np
import sys
import networkx as nx



numrows = int(sys.argv[1])
print(numrows)
path_to_data = sys.argv[2]

#print('Read In Edgelist to DataFrame')
start=time.perf_counter()
edges = pd.read_csv(path_to_data, usecols=[0, 1], names=['source', 'target'], header=None, sep=" ", dtype=np.int32, nrows=numrows)
end=time.perf_counter()
print(end-start)
print()

print("Create Graph From Edgelist")
start=time.perf_counter()
G = nx.from_pandas_edgelist(edges)
end=time.perf_counter()
print(end-start)
print()

print('Get Degree of Each Node')
start=time.perf_counter()
degrees = pd.Series([val for (node,val) in G.degree()],dtype=np.int32)
end=time.perf_counter()
print(end-start)
print()

print('Estimate Alpha and Xmin')
start=time.perf_counter()
a = plfit(degrees)
end = time.perf_counter()
print(a)
print(end-start)
print()
