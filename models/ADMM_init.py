'''
Description: 
Version: 1.0
Autor: Ziyang Zheng
Date: 2022-02-07 23:53:53
LastEditors: Ziyang Zheng
LastEditTime: 2022-02-07 23:53:53
'''
import time
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky

# Python version of https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
def lasso_admm(A,b,lam,rho=1.,alpha=1.,record=False,QUIET=False,\
                MAX_ITER=800,ABSTOL=1e-4,RELTOL= 1e-2):
    """
     Solve lasso problem via ADMM
     [z, history] = lasso_admm(A,b,lam,rho,alpha)
     Solves the following problem via ADMM:
    
       minimize 1/2*|| Ax - b ||_2^2 + \lam || x ||_1
    OUTPUT:
     The solution is returned in the vector z.
     The solution in each iteration is recorded in z_iters.
    
     history is a dictionary containing the objective value, the primal and
     dual residual norms, and the tolerances for the primal and dual residual
     norms at each iteration.

    INPUT:
     rho is the augmented Lagrangian parameter.
    
     alpha is the over-relaxation parameter (typical values for alpha are
     between 1.0 and 1.8).
     
     record is true for recording the results in each iteration, false for saving time.

     More information can be found in the paper linked at:
     http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    """

    if not QUIET:
        tic = time.time()

    #Data preprocessing
    m,n = A.shape
    #save a matrix-vector multiply
    Atb = A.T.dot(b)

    #ADMM solver
    x = np.zeros((n,1))
    z = np.zeros((n,1))
    u = np.zeros((n,1))
    if record:
        z_iters = np.zeros((n,MAX_ITER))

    # cache the (Cholesky) factorization
    L,U = factor(A,rho)

    if not QUIET:
        print ('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' %('iter',
                                                      'r norm', 
                                                      'eps pri', 
                                                      's norm', 
                                                      'eps dual', 
                                                      'objective'))

    # Saving state
    h = {}
    h['objval']     = np.zeros(MAX_ITER)
    h['r_norm']     = np.zeros(MAX_ITER)
    h['s_norm']     = np.zeros(MAX_ITER)
    h['eps_pri']    = np.zeros(MAX_ITER)
    h['eps_dual']   = np.zeros(MAX_ITER)

    for k in range(MAX_ITER):

        # x-update 
        q = Atb+rho*(z-u) #(temporary value)
        if m>=n:  #if skinny
            x = spsolve(U,spsolve(L,q))[...,np.newaxis]
        else: # if fat
            ULAq = spsolve(U,spsolve(L,A.dot(q)))[...,np.newaxis]
            x = (q*1./rho)-((A.T.dot(ULAq))*1./(rho**2))

        # z-update with relaxation
        zold = np.copy(z)
        x_hat = alpha*x+(1.-alpha)*zold
        z = shrinkage(x_hat+u,lam*1./rho)

        # u-update
        u+=(x_hat-z)

        # diagnostics, reporting, termination checks
        h['objval'][k]   = objective(A,b,lam,x,z)
        h['r_norm'][k]   = norm(x-z)
        h['s_norm'][k]   = norm(-rho*(z-zold))
        h['eps_pri'][k]  = np.sqrt(n)*ABSTOL+\
                            RELTOL*np.maximum(norm(x),norm(-z))
        h['eps_dual'][k] = np.sqrt(n)*ABSTOL+\
                            RELTOL*norm(rho*u)
        if not QUIET:
            print ('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' %(k+1,\
                                                          h['r_norm'][k],\
                                                          h['eps_pri'][k],\
                                                          h['s_norm'][k],\
                                                          h['eps_dual'][k],\
                                                          h['objval'][k]))
        if record:
            z_iters[:,k] = z.ravel()
        else:
            if (h['r_norm'][k]<h['eps_pri'][k]) and (h['s_norm'][k]<h['eps_dual'][k]):
                break            

    if not QUIET:
        toc = time.time()-tic
        print ("\nElapsed time is %.2f seconds"%toc)
    if record:
        return z_iters,h
    else:
        return z.ravel(),h

def objective(A,b,lam,x,z):
    return .5*np.square(A.dot(x)-b).sum()+lam*norm(z,1)

def shrinkage(x,kappa):
    return np.maximum(0.,x-kappa)-np.maximum(0.,-x-kappa)

def factor(A,rho):
    m,n = A.shape
    if m>=n:
       L = cholesky(A.T.dot(A)+rho*sparse.eye(n))
    else:
       L = cholesky(sparse.eye(m)+1./rho*(A.dot(A.T)))
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L,U
    