import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from scipy.linalg import cholesky, cho_solve, solve_triangular

X = np.atleast_2d([float(i) for i in range(10)]).T
y = np.sin(X)

scaler_y = StandardScaler().fit(y) 
y_train = scaler_y.transform(y)

kernel = np.exp(-0.012689) * RBF(length_scale=np.exp(0.076071), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=np.exp(-11.687692), noise_level_bounds=(1e-10, 1e+1))
# kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0)
gp.fit(X, scaler_y.transform(y))

print (np.exp(gp.kernel.theta))
print (gp.kernel.theta)

"""
gp.kernel_.theta = gp.kernel.theta 
print (gp.kernel_.theta)
print (gp.log_marginal_likelihood([0.000000, 0.000000, -11.512925],eval_gradient=True))
print (gp.log_marginal_likelihood([-0.012689, 0.076071, -11.687692],eval_gradient=True))
print (gp.log_marginal_likelihood([-0.033371, 0.146722, -11.989325],eval_gradient=True))

"""

"""
whitekernel = WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
k, grad = whitekernel.__call__(X, eval_gradient=True)
print (grad)
"""

# """
K = gp.kernel(X)
print (K)

L = cholesky(K, lower=True)
alpha = cho_solve((L, True), y_train)

# log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
# print ("accurate lml_0 val", -0.5 * np.einsum("ik,ik->k", y_train, alpha)) # here

lml_0 = -0.5*(y_train.T).dot(np.linalg.inv(K)).dot(y_train) # ok
lml_0 = lml_0.sum(-1).sum(-1)
print("lml_0", lml_0)

lml_1 = -np.log(np.diag(L)).sum()
print("lml_1", lml_1)

lml_2 = -K.shape[0] / 2 * np.log(2 * np.pi)
print("lml_2", lml_2) # ok

log_likelihood = lml_0 + lml_1 + lml_2  # sum over dimensions
print ("sum", log_likelihood)
# """