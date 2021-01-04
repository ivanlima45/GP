import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class GP:

  def __init__(self, kernel, sigma):
    self.kernel = kernel
    self.sigma = sigma
    print(sigma)

  def addData(self, train, labels):
    self.xtrain = train
    self.ytrain = labels

  def posterior(self, xtest, ell=1.0, s =1.0):
    # p( f*|X*,X, f ) = N ( f*|mu*,Sigma*)
    # Sigma* = K** - K*^TK^-1K*
    # Sigma_s = K_2s - K_s^TK_invK_s
    
    K_2s = self.kernel(xtest, xtest, ell, s) + self.sigma**2 * np.eye(len(xtest))
    K_s = self.kernel(self.xtrain, xtest, ell, s)
    K = self.kernel(self.xtrain, self.xtrain, ell, s) + self.sigma**2 * np.eye(len(self.xtrain))
    ##K_inv = np.linalg.inv(K)

    ##mu_s = K_s.T.dot(K_inv).dot(self.ytrain)
    ##cov_s = K_2s - K_s.T.dot(K_inv).dot(K_s)
  
    ##cholesky decompositon to avoid matrix invertion
    L = np.linalg.cholesky(K)
    aux = np.linalg.solve(L,self.ytrain)
    alpha = np.linalg.solve(L.T,aux)
    mu_s = K_s.T.dot(alpha)
    v = np.linalg.solve(L,K_s)
    cov_s = K_2s - v.T.dot(v)
  
    # Must return mu and s2
    return mu_s, cov_s

  def logmarglikelihood(self,ell,s):
    K = self.kernel(self.xtrain, self.xtrain, ell, s) + self.sigma**2*np.eye(len(self.xtrain))
    return 0.5*(self.ytrain.T@np.linalg.inv(K)@self.ytrain + np.log(np.linalg.det(K)) + len(self.xtrain)*np.log(2*np.pi))
  
  def logmarglikelihood2(self,ell,s, sig):
    K = self.kernel(self.xtrain, self.xtrain, ell, s) + sig**2*np.eye(len(self.xtrain))
    return 0.5*(self.ytrain.T@np.linalg.inv(K)@self.ytrain + np.log(np.linalg.det(K)) + len(self.xtrain)*np.log(2*np.pi))
  
  ##parameters optimization
  def opt_hyper(self,ell,s):
    result = minimize(lambda hyper: self.logmarglikelihood(hyper[0],hyper[1]),[ell,s])
    return result.x[0],result.x[1] 

  def plot_opt(self,xtest):
    n_ell = 100
    n_s   = 100
    c_ell = np.logspace(-1,1.1,n_ell)
    ##c_ell = np.logspace(-1,1.1,n_ell)
    c_s   = np.logspace(-2,-0.5,n_s)
    ##c_s   = np.logspace(-2,-0.5,n_s)
    L,S   = np.meshgrid(c_ell,c_s)
    Z = np.zeros((n_ell,n_s))
    
    for i in range(n_ell):
        for j in range(n_s):
            Z[i,j] = - self.logmarglikelihood(L[i,j],S[i,j])
    plt.contour(L,S, Z, 500, vmin=-8)
    plt.ylabel('s vertical scale')
    plt.xlabel('ell lenghtscale')
    plt.title('hyperparameters')
    plt.savefig("../output/"+'hyperparameters'+".png")
    plt.show()

    
