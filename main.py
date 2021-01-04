# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from GP import GP

'''
  Methods for Plotting 1D and 2D Gaussian Processes
'''
def plot(mu_s, cov_s, xtest, title='graph', std_dev_mul=2.0, xtrain=None, ytrain=None, samples=[]):
    xtest = xtest.ravel()
    mu_s = mu_s.ravel()
    uncertainty = std_dev_mul * np.sqrt(np.diag(cov_s))
    
    plt.clf()

    plt.fill_between(xtest, mu_s + uncertainty, mu_s - uncertainty, alpha=0.1)
    plt.plot(xtest, mu_s, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(xtest, sample, lw=1, ls='--', label=f'Prior {i+1}')
    if xtrain is not None:
        plt.plot(xtrain, ytrain, 'rx')
    plt.legend()
    plt.ylabel('output f(x)')
    plt.xlabel('input x')
    #plt.title(title)
    plt.savefig("../output/"+title+".png", bbox_inches='tight')
    plt.clf()

def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    plt.clf()
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_xlabel('input x1')
    ax.set_ylabel('input x2')
    ax.set_zlabel('output y')
    #bbox = ax.figure.bbox_inches.from_bounds(1, 1, 8, 1)
    #ax.dist = 15
    #ax.set_title(title)
    ax.figure.savefig("../output/"+title+".png")#bbox_inches =bbox)

'''
  Defining Different Kernels to be used
'''
def kernel(x1, x2, ell=1.0, s=1.0):
  # First calculate the distance (x - x')^2 = x^2 + x'^2 -2 xx'
  dist = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
  return s**2 * np.exp(-0.5 * dist / ell**2)

def kernel_R(a, b, ell = 1.0 , s=1.0):
  """ rational quadratic kernel (alpha > 0) """
  alpha =0.5
  d2 = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
  return s**2*(1 + d2/(2*alpha*ell**2))**(-alpha)

def kernel_O(a, b, ell = 1.0 , s=1.0):
  """ Ornstein Uhlenbeck kernel """  
  d = np.abs(np.sum(a,1).reshape(-1,1) - np.sum(b,1))
  return s**2*np.exp(-.5*d/ell**2)

## gamma-exponential (0 < gamma =< 2
def kernel_G(a, b, ell = 1.0 , s=1.0):
  gamma=0.2
  d = np.abs(np.sum(a,1).reshape(-1,1) - np.sum(b,1))
  return s**2*np.exp((-gamma)*d/ell**2)

''' 
  Data generation Methods
'''
def generate_2d_testset(downLimit, upperLimit, size, seed=33):
  np.random.seed(42)
  dist = abs((upperLimit - downLimit)/size)
  x, y = np.arange(downLimit, upperLimit, dist), np.arange(downLimit, upperLimit, dist)
  mesh_x, mesh_y = np.meshgrid(x, y)
  
  return mesh_x, mesh_y, np.c_[mesh_x.ravel(), mesh_y.ravel()]

def generate_2d_dataset(downLimit, upperLimit, size, y_function, seed=42, noise = 0.0):
  np.random.seed(seed)
  data = np.random.uniform(downLimit, upperLimit, (size, 2))
  labels = y_function(0.5 * np.linalg.norm(data, axis=1)) + noise * np.random.randn(len(data))
  return data, labels

def generate_1d_dataset(downLimit, upperLimit, size, y_function, seed=42, noise=0.0):
  np.random.seed(seed)
  data = np.random.uniform(downLimit, upperLimit, size=size).reshape(-1, 1)
  labels = y_function(data)
  return data, labels

'''
  Models Execution Methods
'''
def execute_1d(title, kernelFn=kernel, ell=1.0, s=1.0, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=lambda x: np.sin(x).flatten()): 
  # Finite number of points
  X_train, Y_train = generate_1d_dataset(-4, 4, (train_size,1), function, seed=train_seed)
  
  X_test, Y_test = generate_1d_dataset(-5,5, (100,1), function, seed=test_seed, noise=noise)
  X_test.sort(axis=0)
  
  # Plot GP mean, confidence interval and samples 
  gp = GP(kernelFn,noise)
  
  gp.addData(X_train, Y_train)
  mu, cov = gp.posterior(X_test)

  # Draw three samples from the prior
  samples = np.random.multivariate_normal(mu.ravel(), cov, 4)

  plot(mu, cov, X_test, title=title, xtrain=X_train, ytrain=Y_train, samples=samples)

def execute_2d(title, kernelFn=kernel, ell=1.0, s=1.0, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=lambda x: np.sin(x).flatten()):

  gx, gy, X_2D = generate_2d_testset(-5,5, 100, test_seed)
  X_2D_train, Y_2D_train = generate_2d_dataset(-4,4, train_size, function, train_seed, noise=noise)
  
  gp = GP(kernelFn, noise)
  gp.addData(X_2D_train, Y_2D_train)

  mu_s, _ = gp.posterior(X_2D)
  plot_gp_2D(gx, gy, mu_s, X_2D_train, Y_2D_train, title, 1)
  
def execute_1d_optimized(title, kernelFn=kernel, ell_0 =1, s_0 =1, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=lambda x: np.sin(x).flatten()): 
  #This function stills need to be more object-oreinted, method plot_opt has to be more parameterized in order to be more abstract
  print ("\n -Experimenting hypermarameter optimizations in 1 Dimension")
  X_train, Y_train = generate_1d_dataset(-4, 4, (train_size,1), function, seed=train_seed)
  X_test, Y_test = generate_1d_dataset(-5,5, (100,1), function, seed=test_seed, noise=noise)
  X_test.sort(axis=0)
  gp = GP(kernelFn,noise)
  gp.addData(X_train, Y_train)

  ell_opt, s_opt = gp.opt_hyper(ell=ell_0,s=s_0)
  lgl = gp.logmarglikelihood(ell_opt,s_opt)
  print('ell_opt = ', ell_opt)
  print('s_opt = ', s_opt)
  print('ell_0 = ', ell_0)
  print('s_0 = ', s_0) 
  print('lik = ', lgl)
  file= open("../output/hyperparameters_opt.txt","a")
  file.write(" " + '\n')
  file.write(title)
  file.write(" " + '\n')
  file.write("number of points = %d\n"       %(train_size))
  file.write("noise variance = %.5f\n"       %(noise))       
  file.write('initial ell = %.2f\n'          %(ell_0))
  file.write('initial s = %.2f\n'            %(s_0))
  file.write('optimal ell = %.2f\n'          %(ell_opt))
  file.write('optimal s = %.2f\n'            %(s_opt))
  file.write('neg marg likelihood = %.2f\n'  %(lgl))
  file.write(" " + '\n')
  file.close()

def execute_1d_contour(title, kernelFn=kernel,ell_max=1.0,sigma_max=0.5,s=1.0, n_ell=100, n_sigma=100, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=lambda x: np.sin(x).flatten()):  
  X_train, Y_train = generate_1d_dataset(-4, 4, (train_size,1), function, seed=train_seed)
  X_test, Y_test = generate_1d_dataset(-5,5, (100,1), function, seed=test_seed, noise=noise)
  X_test.sort(axis=0)
  gp = GP(kernelFn,noise)
  gp.addData(X_train, Y_train)
  
  c_ell = np.logspace(-1,ell_max,n_ell)
  ##c_ell = np.logspace(-1,1.5,n_ell)
  c_sigma   = np.logspace(-2,sigma_max,n_sigma)
  ##c_s   = np.logspace(-2,3,n_s)
  L,S   = np.meshgrid(c_ell,c_sigma)
  Z = np.zeros((n_ell,n_sigma))
  
  for i in range(n_ell):
      for j in range(n_sigma):
          Z[i,j] = - gp.logmarglikelihood2(L[i,j],s,sig=S[i,j])
  plt.clf()
  plt.contour(np.log(L),np.log(S), Z, 500, vmin=-8)
  plt.ylabel('log noise std deviation')
  plt.xlabel('log ell horizontal scale')
  plt.title('contour marginal likelihood')
  plt.savefig("../output/"+title+"_log.png", bbox_inches='tight')
  #plt.show()
  ##  plt.savefig('SE_Kernel_xtrain10_plotn100_log_ell_vs_noise_Lik')  
  plt.clf()
  plt.contour(L,S, Z, 500, vmin=-8)
  plt.ylabel('noise std deviation')
  plt.xlabel('ell horizontal scale')
  plt.title('contour marginal likelihood')
  plt.savefig("../output/"+title+".png", bbox_inches='tight')
  
def execute_1d_contour2(title, kernelFn=kernel,ell_max=1.0,s_max=0.5,sigma=0.00005, n_ell=100, n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=lambda x: np.sin(x).flatten()):  
  
  X_train, Y_train = generate_1d_dataset(-4, 4, (train_size,1), function, seed=train_seed)
  X_test, Y_test = generate_1d_dataset(-5,5, (100,1), function, seed=test_seed, noise=noise)
  X_test.sort(axis=0)
  gp = GP(kernelFn,noise)
  gp.addData(X_train, Y_train)
  
  c_ell = np.logspace(-1,ell_max,n_ell)
  ##c_ell = np.logspace(-1,1.5,n_ell)
  c_s   = np.logspace(-2,s_max,n_s)
  ##c_s   = np.logspace(-2,3,n_s)
  L,S   = np.meshgrid(c_ell,c_s)
  Z = np.zeros((n_ell,n_s))
  
  for i in range(n_ell):
      for j in range(n_s):
          Z[i,j] = - gp.logmarglikelihood2(L[i,j],s=S[i,j],sig=sigma)
  plt.clf()
  plt.contour(np.log(L),np.log(S), Z, 500, vmin=-8)
  plt.ylabel('log s vertical scale')
  plt.xlabel('log ell horizontal scale')
  plt.title('contour marginal likelihood')
  plt.savefig("../output/"+title+"_log.png", bbox_inches='tight')
  #plt.show()
  ##  plt.savefig('SE_Kernel_xtrain10_plotn100_log_ell_vs_noise_Lik')  
  plt.clf()
  plt.contour(L,S, Z, 500, vmin=-8)
  plt.ylabel('s vertical scale')
  plt.xlabel('ell horizontal scale')
  #plt.title('contour marginal likelihood')
  plt.savefig("../output/"+title+".png", bbox_inches='tight')

def execute_1d_like_vs_ell(title,kernelFn=kernel,ell_max=1.0, s=1.0, sigma=0.00005,  n_ell=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=lambda x: np.sin(x).flatten()):
  X_train, Y_train = generate_1d_dataset(-4, 4, (train_size,1), function, seed=train_seed)
  X_test, Y_test = generate_1d_dataset(-5,5, (100,1), function, seed=test_seed, noise=noise)
  X_test.sort(axis=0)
  gp = GP(kernelFn,noise)
  gp.addData(X_train, Y_train)
  
  c_ell = np.linspace(0.01,ell_max,n_ell)
  Z = np.zeros(n_ell)
  
  for j in range(n_ell):
      Z[j] = - gp.logmarglikelihood2(c_ell[j],s,sig=sigma)
  plt.clf()
  plt.plot(c_ell, Z)
  plt.ylabel('marginal likelihood')
  plt.xlabel('ell horizontal scale')
  plt.savefig("../output/"+title+".png", bbox_inches='tight')

def execute_1d_like_vs_s(title,kernelFn=kernel,s_max=1.0, ell=1.0, sigma=0.00005,  n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=lambda x: np.sin(x).flatten()):
  X_train, Y_train = generate_1d_dataset(-4, 4, (train_size,1), function, seed=train_seed)
  X_test, Y_test = generate_1d_dataset(-5,5, (100,1), function, seed=test_seed, noise=noise)
  X_test.sort(axis=0)
  gp = GP(kernelFn,noise)
  gp.addData(X_train, Y_train)
  
  c_s = np.linspace(0.001,s_max,n_s)
  Z = np.zeros(n_s)
  
  for j in range(n_s):
      Z[j] = - gp.logmarglikelihood2(ell, c_s[j] ,sig=sigma)
  plt.clf()
  plt.plot(c_s, Z)
  plt.ylabel('marginal likelihood')
  plt.xlabel('s vertical scale')
  plt.savefig("../output/"+title+".png", bbox_inches='tight')
 
def execute_1d_like_vs_noise(title,kernelFn=kernel,sig_max=1.0, ell=1.0, s=1.0,  n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=lambda x: np.sin(x).flatten()):
  X_train, Y_train = generate_1d_dataset(-4, 4, (train_size,1), function, seed=train_seed)
  X_test, Y_test = generate_1d_dataset(-5,5, (100,1), function, seed=test_seed, noise=noise)
  X_test.sort(axis=0)
  gp = GP(kernelFn,noise)
  gp.addData(X_train, Y_train)
  
  c_s = np.linspace(0.001,sig_max,n_s)
  Z = np.zeros(n_s)
  
  for j in range(n_s):
      Z[j] = - gp.logmarglikelihood2(ell, s ,sig=c_s[j])
  plt.clf()
  plt.plot(c_s, Z)
  plt.ylabel('marginal likelihood')
  plt.xlabel('noise std deviation')
  plt.savefig("../output/"+title+".png",bbox_inches='tight')

'''
  Execution Experiments
  Possible functions we are going to use to calibrate our GP
  
  f = lambda x: (0.1*(x**2)).flatten()
  f = lambda x: (0.2*(x**3)).flatten()
  f = lambda x:  np.sin(0.5*x**2).flatten()
  f = lambda x:  (0.02*x**3).flatten()*np.sin(1.0*x**2).flatten()
  f = lambda x:  (np.exp(x)*0.1).flatten()*np.sin(0.02*x**2).flatten()
'''
#############################################################################################################
def experiments_1D():
  print ("Experimenting in 1 Dimension - Changing Kernel, inputsize and function")
  print ("Default Kernel - Noiseless Sin Function")
  f = lambda x: np.sin(x).flatten()
  execute_1d("1D_5points_SE_Kernel_sin_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_SE_Kernel_sin_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_SE_Kernel_sin_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Default Kernel - With noise Sin Function")
  execute_1d("1D_5points_SE_Kernel_sin_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_SE_Kernel_sin_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_SE_Kernel_sin_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
  print ("Rational Kernel - Noiseless Sin Function")
  execute_1d("1D_5points_Rational_Kernel_sin_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_Rational_Kernel_sin_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_Rational_Kernel_sin_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Rational Kernel - With noise Sin Function")
  execute_1d("1D_5points_Rational_Kernel_sin_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_Rational_Kernel_sin_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_Rational_Kernel_sin_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
 
  print ("Ornstein Uhlenbeck Kernel - Noiseless Sin Function")
  execute_1d("1D_5points_Ornstein_Kernel_sin_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_Ornstein_Kernel_sin_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_Ornstein_Kernel_sin_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Ornstein Uhlenbeck Kernel - With noise Sin Function")
  execute_1d("1D_5points_Ornstein_Kernel_sin_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_Ornstein_Kernel_sin_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_Ornstein_Kernel_sin_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
 
  print ("Gamma Kernel - Noiseless Sin Function")
  execute_1d("1D_5points_Gamma_Kernel_sin_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_Gamma_Kernel_sin_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_Gamma_Kernel_sin_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Gamma Kernel - With noise Sin Function")
  execute_1d("1D_5points_Gamma_Kernel_sin_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_Gamma_Kernel_sin_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_Gamma_Kernel_sin_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)

  print ("Experimenting in 1 Dimension - Changing Kernel, inputsize and function")
  print ("Default Kernel - Noiseless Square Function")
  f = lambda x: (0.1*(x**2)).flatten()
  execute_1d("1D_5points_SE_Kernel_square_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_SE_Kernel_square_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_SE_Kernel_square_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Default Kernel - With noise square Function")
  execute_1d("1D_5points_SE_Kernel_square_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_SE_Kernel_square_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_SE_Kernel_square_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
  print ("Rational Kernel - Noiseless square Function")
  execute_1d("1D_5points_Rational_Kernel_square_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_Rational_Kernel_square_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_Rational_Kernel_square_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Rational Kernel - With noise square Function")
  execute_1d("1D_5points_Rational_Kernel_square_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_Rational_Kernel_square_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_Rational_Kernel_square_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
 
  print ("Ornstein Uhlenbeck Kernel - Noiseless square Function")
  execute_1d("1D_5points_Ornstein_Kernel_square_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_Ornstein_Kernel_square_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_Ornstein_Kernel_square_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Ornstein Uhlenbeck Kernel - With noise square Function")
  execute_1d("1D_5points_Ornstein_Kernel_square_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_Ornstein_Kernel_square_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_Ornstein_Kernel_square_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
 
  print ("Gamma Kernel - Noiseless square Function")
  execute_1d("1D_5points_Gamma_Kernel_square_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_Gamma_Kernel_square_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_Gamma_Kernel_square_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Gamma Kernel - With noise square Function")
  execute_1d("1D_5points_Gamma_Kernel_square_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_Gamma_Kernel_square_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_Gamma_Kernel_square_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
#############################################################################################################

  print ("Experimenting in 1 Dimension - Changing Kernel, inputsize and function")
  print ("Default Kernel - Noiseless Cubic Function")
  f = lambda x: (0.2*(x**3)).flatten()
  execute_1d("1D_5points_SE_Kernel_cubic_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_SE_Kernel_cubic_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_SE_Kernel_cubic_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Default Kernel - With noise cubic Function")
  execute_1d("1D_5points_SE_Kernel_cubic_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_SE_Kernel_cubic_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_SE_Kernel_cubic_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
  print ("Rational Kernel - Noiseless cubic Function")
  execute_1d("1D_5points_Rational_Kernel_cubic_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_Rational_Kernel_cubic_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_Rational_Kernel_cubic_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Rational Kernel - With noise cubic Function")
  execute_1d("1D_5points_Rational_Kernel_cubic_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_Rational_Kernel_cubic_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_Rational_Kernel_cubic_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
 
  print ("Ornstein Uhlenbeck Kernel - Noiseless cubic Function")
  execute_1d("1D_5points_Ornstein_Kernel_cubic_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_Ornstein_Kernel_cubic_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_Ornstein_Kernel_cubic_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Ornstein Uhlenbeck Kernel - With noise cubic Function")
  execute_1d("1D_5points_Ornstein_Kernel_cubic_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_Ornstein_Kernel_cubic_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_Ornstein_Kernel_cubic_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
 
  print ("Gamma Kernel - Noiseless cubic Function")
  execute_1d("1D_5points_Gamma_Kernel_cubic_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_Gamma_Kernel_cubic_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_Gamma_Kernel_cubic_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)
  
  print ("Gamma Kernel - With noise cubic Function")
  execute_1d("1D_5points_Gamma_Kernel_cubic_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_Gamma_Kernel_cubic_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_Gamma_Kernel_cubic_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  #############################################################################################################
  print ("Experimenting in 1 Dimension - Changing S and ELL parameters")
  print ("Default Kernel - Noiseless Sin Function and different S and Ell")
  execute_1d("1D_5points_SE_Kernel_sin_noiseless_low_params", kernelFn=kernel, ell=0.01, s=0.01, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_SE_Kernel_sin_noiseless_low_params", kernelFn=kernel, ell=0.01, s=0.01, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_SE_Kernel_sin_noiseless_low_params", kernelFn=kernel, ell=0.01, s=0.01, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)

  execute_1d("1D_5points_SE_Kernel_sin_noiseless_medium_params", kernelFn=kernel, ell=0.5, s=0.5, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_SE_Kernel_sin_noiseless_medium_params", kernelFn=kernel, ell=0.5, s=0.5, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_SE_Kernel_sin_noiseless_medium_params", kernelFn=kernel, ell=0.5, s=0.5, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)

  execute_1d("1D_5points_SE_Kernel_sin_noiseless_high_params", kernelFn=kernel, ell=10.0, s=0.01, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_SE_Kernel_sin_noiseless_high_params", kernelFn=kernel, ell=10.0, s=0.01, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_SE_Kernel_sin_noiseless_high_params", kernelFn=kernel, ell=10.0, s=0.01, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
def experiments_opt_1D():
  f = lambda x: np.sin(x).flatten()
  execute_1d_optimized("1D_SE_Kernel_sin", kernelFn=kernel, ell_0 =1.0,s_0 =1.0,train_seed=42, test_seed=33, train_size=30, noise=0.0005, function=f)
  execute_1d_optimized("1D_SE_Kernel_sin", kernelFn=kernel, ell_0 =0.5,s_0 =1.0,train_seed=42, test_seed=33, train_size=30, noise=0.0005, function=f)
  execute_1d_optimized("1D_SE_Kernel_sin", kernelFn=kernel, ell_0 =0.5,s_0 =5.0,train_seed=42, test_seed=33, train_size=30, noise=0.0005, function=f)
  execute_1d_optimized("1D_SE_Kernel_sin", kernelFn=kernel, ell_0 =2.0,s_0 =2.0,train_seed=42, test_seed=33, train_size=30, noise=0.0005, function=f)
  execute_1d_optimized("1D_SE_Kernel_sin", kernelFn=kernel, ell_0 =2.0,s_0 =5.0,train_seed=42, test_seed=33, train_size=30, noise=0.0005, function=f)
  execute_1d_optimized("1D_SE_Kernel_sin", kernelFn=kernel, ell_0 =3.0,s_0 =4.0,train_seed=42, test_seed=33, train_size=30, noise=0.0005, function=f)
def experiments_contour_1D():
  f = lambda x: np.sin(x).flatten()
  execute_1d_contour("1D_SE_Kernel_sin_contour_noise_vs_ell",  kernelFn=kernel,ell_max=1.0, sigma_max=0.5, s=1.0,  n_ell=100, n_sigma=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_contour("1D_R_Kernel_sin_contour_noise_vs_ell", kernelFn=kernel_R,ell_max=1.0, sigma_max=0.5, s=1.0,  n_ell=100, n_sigma=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_contour("1D_O_Kernel_sin_contour_noise_vs_ell", kernelFn=kernel_O,ell_max=1.0, sigma_max=0.5, s=1.0,  n_ell=100, n_sigma=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_contour("1D_G_Kernel_sin_contour_noise_vs_ell", kernelFn=kernel_G,ell_max=1.0, sigma_max=0.5, s=1.0,  n_ell=100, n_sigma=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  
  execute_1d_contour2("1D_SE_Kernel_sin_contour_s_vs_ell",  kernelFn=kernel,ell_max=1.0, s_max=0.5, sigma=0.0005,  n_ell=100, n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_contour2("1D_R_Kernel_sin_contour_s_vs_ell", kernelFn=kernel_R,ell_max=1.0, s_max=0.5, sigma=0.0005,  n_ell=100, n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_contour2("1D_O_Kernel_sin_contour_s_vs_ell", kernelFn=kernel_O,ell_max=1.0, s_max=0.5, sigma=0.0005,  n_ell=100, n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_contour2("1D_G_Kernel_sin_contour_s_vs_ell", kernelFn=kernel_G,ell_max=1.0, s_max=0.5, sigma=0.0005,  n_ell=100, n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  

def experiments_like_curve_1D():
  f = lambda x: np.sin(x).flatten()  
  execute_1d_like_vs_ell("1D_SE_Kernel_sin_likelihood_vs_ell",  kernelFn=kernel,ell_max=2.0, s=1.0, sigma=0.00005,  n_ell=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_like_vs_ell("1D_R_Kernel_sin_likelihood_vs_ell", kernelFn=kernel_R,ell_max=2.0, s=1.0, sigma=0.00005,  n_ell=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_like_vs_ell("1D_O_Kernel_sin_likelihood_vs_ell", kernelFn=kernel_O,ell_max=2.0, s=1.0, sigma=0.00005,  n_ell=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_like_vs_ell("1D_G_Kernel_sin_likelihood_vs_ell", kernelFn=kernel_G,ell_max=2.0, s=1.0, sigma=0.00005,  n_ell=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  
  execute_1d_like_vs_s("1D_SE_Kernel_sin_likelihood_vs_s",  kernelFn=kernel,s_max=0.1, ell=1.0, sigma=0.00005,  n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_like_vs_s("1D_R_Kernel_sin_likelihood_vs_s", kernelFn=kernel_R,s_max=0.1, ell=1.0, sigma=0.00005,  n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_like_vs_s("1D_O_Kernel_sin_likelihood_vs_s", kernelFn=kernel_O,s_max=0.1, ell=1.0, sigma=0.00005,  n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_like_vs_s("1D_G_Kernel_sin_likelihood_vs_s", kernelFn=kernel_G,s_max=0.1, ell=1.0, sigma=0.00005,  n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  
  execute_1d_like_vs_noise("1D_SE_Kernel_sin_likelihood_vs_noise",  kernelFn=kernel,sig_max=100.0, ell=1.0, s=1.0,  n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_like_vs_noise("1D_R_Kernel_sin_likelihood_vs_noise", kernelFn=kernel_R,sig_max=100.0, ell=1.0, s=1.0,  n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_like_vs_noise("1D_O_Kernel_sin_likelihood_vs_noise", kernelFn=kernel_O,sig_max=100.0, ell=1.0, s=1.0,  n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
  execute_1d_like_vs_noise("1D_G_Kernel_sin_likelihood_vs_noise", kernelFn=kernel_G,sig_max=100.0, ell=1.0, s=1.0,  n_s=100, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)  
 #############################################################################################################
def experiments_1D_paran():
  f = lambda x: np.sin(x).flatten()
  print ("Experimenting in 1 Dimension - Changing S and ELL parameters")
  print ("Default Kernel - Noiseless Sin Function and different S and Ell")
  execute_1d("1D_5points_SE_Kernel_sin_noiseless_A_params", kernelFn=kernel, ell=1.0, s=1.0, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_1d("1D_10points_SE_Kernel_sin_noiseless_A_params", kernelFn=kernel, ell=1.0, s=1.0, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_1d("1D_50points_SE_Kernel_sin_noiseless_A_params", kernelFn=kernel, ell=1.0, s=1.0, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)

  execute_1d("1D_5points_SE_Kernel_sin_noiseless_B_params", kernelFn=kernel, ell=0.3, s=1.08, train_seed=42, test_seed=33, train_size=5, noise=0.00005, function=f)
  execute_1d("1D_10points_SE_Kernel_sin_noiseless_B_params", kernelFn=kernel, ell=0.3, s=1.08, train_seed=42, test_seed=33, train_size=10, noise=0.00005, function=f)
  execute_1d("1D_50points_SE_Kernel_sin_noiseless_B_params", kernelFn=kernel, ell=0.3, s=1.08, train_seed=42, test_seed=33, train_size=50, noise=0.00005, function=f)

  execute_1d("1D_5points_SE_Kernel_sin_noiseless_C_params", kernelFn=kernel, ell=3.0, s=1.16, train_seed=42, test_seed=33, train_size=5, noise=0.89, function=f)
  execute_1d("1D_10points_SE_Kernel_sin_noiseless_C_params", kernelFn=kernel, ell=3.0, s=1.16, train_seed=42, test_seed=33, train_size=10, noise=0.89, function=f)
  execute_1d("1D_50points_SE_Kernel_sin_noiseless_C_params", kernelFn=kernel, ell=3.0, s=1.16, train_seed=42, test_seed=33, train_size=50, noise=0.89, function=f)

def experiments_2D():
  print ("Experimenting in 2 Dimensions - Changing Kernel, inputsize and function")
  print ("Default Kernel - Noiseless Sin Function")
  f = lambda x: np.sin(x).flatten()
  execute_2d("2D_5points_SE_Kernel_sin_noiseless",  kernelFn=kernel, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=f)
  execute_2d("2D_10points_SE_Kernel_sin_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=10, noise=0.0, function=f)
  execute_2d("2D_50points_SE_Kernel_sin_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=50, noise=0.0, function=f)
  print ("Default Kernel - With noise Sin Function")
  execute_2d("2D_5points_SE_Kernel_sin_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_2d("2D_10points_SE_Kernel_sin_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_2d("2D_50points_SE_Kernel_sin_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
  print ("Rational Kernel - Noiseless Sin Function")
  execute_2d("2D_5points_Rational_Kernel_sin_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=f)
  execute_2d("2D_10points_Rational_Kernel_sin_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=10, noise=0.0, function=f)
  execute_2d("2D_50points_Rational_Kernel_sin_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=50, noise=0.0, function=f)
  print ("Ornstein Uhlenbeck Kernel - With noise Sin Function")
  execute_2d("2D_5points_Rational_Kernel_sin_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_2d("2D_10points_Rational_Kernel_sin_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_2d("2D_50points_Rational_Kernel_sin_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
  print ("Ornstein Uhlenbeck Kernel - Noiseless Sin Function")
  execute_2d("2D_5points_Ornstein_Kernel_sin_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=f)
  execute_2d("2D_10points_Ornstein_Kernel_sin_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=10, noise=0.0, function=f)
  execute_2d("2D_50points_Ornstein_Kernel_sin_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=50, noise=0.0, function=f)
  print ("Ornstein Uhlenbeck Kernel - With noise Sin Function")
  execute_2d("2D_5points_Ornstein_Kernel_sin_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_2d("2D_10points_Ornstein_Kernel_sin_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_2d("2D_50points_Ornstein_Kernel_sin_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
  print ("Gamma Kernel - Noiseless Sin Function")
  execute_2d("2D_5points_Gamma_Kernel_sin_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=f)
  execute_2d("2D_10points_Gamma_Kernel_sin_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=10, noise=0.0, function=f)
  execute_2d("2D_50points_Gamma_Kernel_sin_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=50, noise=0.0, function=f)
  print ("Gamma Kernel - With noise Sin Function")
  execute_2d("2D_5points_Gamma_Kernel_sin_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_2d("2D_10points_Gamma_Kernel_sin_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_2d("2D_50points_Gamma_Kernel_sin_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
  
  
  
  print ("Experimenting in 2 Dimensions - Changing Kernel, inputsize and function")
  print ("Default Kernel - Noiseless Cos Function")
  f = lambda x: np.cos(x).flatten()
  execute_2d("2D_5points_SE_Kernel_cos_noiseless",  kernelFn=kernel, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=f)
  execute_2d("2D_10points_SE_Kernel_cos_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=10, noise=0.0, function=f)
  execute_2d("2D_50points_SE_Kernel_cos_noiseless", kernelFn=kernel, train_seed=42, test_seed=33, train_size=50, noise=0.0, function=f)
  print ("Default Kernel - With noise Cos Function")
  execute_2d("2D_5points_SE_Kernel_cos_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_2d("2D_10points_SE_Kernel_cos_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_2d("2D_50points_SE_Kernel_cos_noise", kernelFn=kernel, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
  print ("Rational Kernel - Noiseless Cos Function")
  execute_2d("2D_5points_Rational_Kernel_cos_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=f)
  execute_2d("2D_10points_Rational_Kernel_cos_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=10, noise=0.0, function=f)
  execute_2d("2D_50points_Rational_Kernel_cos_noiseless", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=50, noise=0.0, function=f)
  print ("Ornstein Uhlenbeck Kernel - With noise Cos Function")
  execute_2d("2D_5points_Rational_Kernel_cos_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_2d("2D_10points_Rational_Kernel_cos_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_2d("2D_50points_Rational_Kernel_cos_noise", kernelFn=kernel_R, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
  print ("Ornstein Uhlenbeck Kernel - Noiseless Cos Function")
  execute_2d("2D_5points_Ornstein_Kernel_cos_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=f)
  execute_2d("2D_10points_Ornstein_Kernel_cos_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=10, noise=0.0, function=f)
  execute_2d("2D_50points_Ornstein_Kernel_cos_noiseless", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=50, noise=0.0, function=f)
  print ("Ornstein Uhlenbeck Kernel - With noise Cos Function")
  execute_2d("2D_5points_Ornstein_Kernel_cos_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_2d("2D_10points_Ornstein_Kernel_cos_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_2d("2D_50points_Ornstein_Kernel_cos_noise", kernelFn=kernel_O, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  
  print ("Gamma Kernel - Noiseless Cos Function")
  execute_2d("2D_5points_Gamma_Kernel_cos_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=5, noise=0.0, function=f)
  execute_2d("2D_10points_Gamma_Kernel_cos_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=10, noise=0.0, function=f)
  execute_2d("2D_50points_Gamma_Kernel_cos_noiseless", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=50, noise=0.0, function=f)
  print ("Gamma Kernel - With noise Cos Function")
  execute_2d("2D_5points_Gamma_Kernel_cos_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=5, noise=0.1, function=f)
  execute_2d("2D_10points_Gamma_Kernel_cos_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=10, noise=0.1, function=f)
  execute_2d("2D_50points_Gamma_Kernel_cos_noise", kernelFn=kernel_G, train_seed=42, test_seed=33, train_size=50, noise=0.1, function=f)
  

'''
  Main Method
'''
def main(): 
  print ("Gaussian Process - Bayesian Optimization")

  experiments_1D()
  experiments_opt_1D()
  experiments_contour_1D()
  experiments_like_curve_1D()
  experiments_1D_paran()
  experiments_2D()

  
  
if __name__ == '__main__':
  main()
