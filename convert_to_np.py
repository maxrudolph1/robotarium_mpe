import numpy as np
import joblib
import sys



loaded_params = joblib.load(sys.argv[1])

out_file = np.array([0] * len(loaded_params),dtype=object)

for i,param in enumerate(loaded_params):
    out_file[i] = (param)


with open(sys.argv[1] + '1.npy', 'wb') as f:
    np.save(f, out_file[:int(len(out_file)/2)])

with open(sys.argv[1] + '2.npy', 'wb') as f:
    np.save(f, out_file[int(len(out_file)/2):])