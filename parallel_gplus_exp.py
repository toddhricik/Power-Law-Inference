from gpu_plfit import *
import time
import sys
import cudf as cd
import cugraph


numrows = int(sys.argv[1])
print(numrows)
path_to_data = sys.argv[2]


print('Read In Edgelist to DataFrame')
start=time.perf_counter()
edges = cd.read_csv(path_to_data, usecols=[0, 1], names=['source', 'destination'], header=None, sep=' ', dtype=['int32','int32'], nrows=numrows)
end=time.perf_counter()
print(end-start)
print()

print("Create Graph From Edgelist")
start=time.perf_counter()
G = cugraph.Graph()
G = cugraph.from_cudf_edgelist(edges,source='source',destination='destination',edge_attr=None)
end=time.perf_counter()
print(end-start)
print()

print('Get Degree of Each Node')
start=time.perf_counter()
degrees = G.degrees()
end=time.perf_counter()
print(end-start)
print()


print('Estimate Alpha and Xmin')
start=time.perf_counter()
a = plfit(cd.Series(degrees.in_degree.values))
end = time.perf_counter()
print(a)
print(end-start)
print()

