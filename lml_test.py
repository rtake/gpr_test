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
print("sum", y.sum(), y_train.sum())

# kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01).fit(X, scaler_y.transform(y))

K = gp.kernel(X)
K[np.diag_indices_from(K)] += 0.01

# print (K)

L = cholesky(K, lower=True)
alpha = cho_solve((L, True), y_train)

log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
print ("accurate lml_0 val", -0.5 * np.einsum("ik,ik->k", y_train, alpha)) # here

lml_0 = -0.5*(y_train.T).dot(np.linalg.inv(K)).dot(y_train) # ok
print("y_train.T", y_train.T)

print("(np.linalg.inv(K))*(y_train)", (np.linalg.inv(K)).dot(y_train))
#print("inv(K)", np.linalg.inv(K))
#print("y_train", y_train)
print("lml_0", lml_0) # ok

#log_likelihood_dims -= np.log(np.diag(L)).sum()
#print (-np.log(np.diag(L)).sum()) # ok

# log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
# print (-K.shape[0] / 2 * np.log(2 * np.pi)) # ok

# log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
#print ("sum", log_likelihood)
