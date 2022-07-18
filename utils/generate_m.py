import numpy as np 

rate = 0.04
n = 256
m = np.ceil(n * rate)
mm_init = np.random.normal(loc=0.0, scale=np.sqrt(1/m), size=[int(m), int(n)])

col_num = mm_init.shape[1]
A = np.zeros([int(m), int(n)])
for i in range(col_num):
    A[:,i] = mm_init[:,i] / np.linalg.norm(mm_init[:,i])
    print(np.linalg.norm(A[:,i]))

np.save('0_'+str(rate*100)+'.npy', A)
