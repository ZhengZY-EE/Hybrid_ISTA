import spams
import numpy as np
import time

model = np.load('model.npy', allow_pickle=True).item()

# Online Dic Learning Parameters
param={'return_model': True,
        'model': model,
        'D': None,
        'numThreads': -1,
        'batchsize': 512,
        'K': 512,
        'lambda1': 0.2,
        'lambda2': 10e-10,
        'iter': 100000,
        't0': 1e-5,
        'rho': 1.0,
        'modeParam': 0}


# Input signals: (signal_size, number of signals)
X = np.load('image.npy')
X = X.reshape((-1, X.shape[2])).transpose()

print('The input dimesion is:', X.shape)

tic = time.time()
(D, model) = spams.trainDL(X, **param)
tac = time.time()
t = tac - tic
print('time of computation for Dictionary Learning: %f' %t)

np.save('D.npy', D)
np.save('model.npy', model)

print('Mission Completed.')
