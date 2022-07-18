import numpy as np 
import scipy.io as scio

# data = np.load('prob.npz')
# dataNew = './D_k50.mat'

# scio.savemat(dataNew, {'D':data['A']})

#########################################

data = scio.loadmat('./W_k30.mat')['W']

np.save('W_k30.npy', data)