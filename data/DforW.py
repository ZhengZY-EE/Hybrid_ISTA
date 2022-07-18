import numpy as np
import scipy.io as scio

mm_mat = np.load('./data/0_50_mm.npy')
D = np.load('./data/D_online.npy')

D_new = np.dot(mm_mat, D)
print(D_new.shape)
scio.savemat('D_50.mat', {'D':D_new})

# D_pse = np.dot(np.linalg.inv(np.dot(D_new.T, D_new)), D_new.T)

# scio.savemat('D_pse.mat', {'D':D_pse})
