import numpy as np
f='models/embeddings.npz'
try:
    d = np.load(f, allow_pickle=True)
    print('keys:', list(d.keys()))
    for k in d:
        print(k, d[k].shape)
except Exception as e:
    print('load failed', e)
