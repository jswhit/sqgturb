import numpy as np
import sys
# generate time mean statistics from standard output of sqg_enkf.py
file = sys.argv[1]
data = np.loadtxt(file)
nskip = int(sys.argv[2])
if len(sys.argv) > 3:
    nend = int(sys.argv[3])
else:
    nend = -1
if nend == -1:
    data2 = data[nskip:,:]
else:
    data2 = data[nskip:nend,:]
data_mean = data2.mean(axis=0)
data_mean[0] = data2.shape[0]
print_list = ''.join(['%g ' % x for x in data_mean])
print(print_list)

