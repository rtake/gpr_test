"""
=============================================================
Gaussian process regression (GPR) with noise-level estimation
=============================================================

This example illustrates that GPR with a sum-kernel including a WhiteKernel can
estimate the noise level of data. An illustration of the
log-marginal-likelihood (LML) landscape shows that there exist two local
maxima of LML. The first corresponds to a model with a high noise level and a
large length scale, which explains all variations in the data by noise. The
second one has a smaller noise level and shorter length scale, which explains
most of the variation by the noise-free functional relationship. The second
model has a higher likelihood; however, depending on the initial value for the
hyperparameters, the gradient-based optimization might also converge to the
high-noise solution. It is thus important to repeat the optimization several
times for different initializations.
"""
print(__doc__)

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


X = np.atleast_2d([float(i) for i in range(10)]).T
y = np.sin(X)

scaler_y = StandardScaler().fit(y)

plt.figure()
kernel = 1 * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e3))# + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
# kernel = np.exp(2) * RBF(length_scale=np.exp(2), length_scale_bounds=(1e-2, 1e3))# + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01).fit(X, scaler_y.transform(y))
X_ = np.linspace(0, 10, 100)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
# plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)), y_mean + np.sqrt(np.diag(y_cov)), alpha=0.5, color='k')
# plt.plot(X_, np.sin(X_), 'r', lw=3, zorder=9)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"  % (kernel, gp.kernel_,  gp.log_marginal_likelihood(gp.kernel_.theta)))
plt.tight_layout()




# Plot LML landscape
plt.figure()
plt.plot(gp.kernel.theta[0], gp.kernel.theta[1], marker='.', c='r') # initial point
plt.plot(gp.kernel_.theta[0], gp.kernel_.theta[1], marker='.', c='r') # optinum point

theta0 = np.linspace(-5, 5, 100)
theta1 = np.linspace(-5, 5, 100)
Theta0, Theta1 = np.meshgrid(theta0, theta1)


LML = [[gp.log_marginal_likelihood([Theta0[i, j], Theta1[i, j]]) for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
LML = np.array(LML).T

"""
print ("X")
print (X)
"""

print ("gp.kernel.diag")
print (gp.kernel.diag(X))

print ("gp.kernel.__call__")
print (gp.kernel.__call__(X))

print ("gp.kernel_.diag")
print (gp.kernel_.diag(X))

print ("Optinum hyperparameter(log-transformed)")
print (gp.kernel_.theta)

print ("gp.log_marginal_likelihood [0,0]")
print (gp.log_marginal_likelihood([gp.kernel.theta[0], gp.kernel.theta[1]], eval_gradient=True))


cont = plt.contour(Theta0, Theta1, LML, levels=50)
cont.clabel(fmt='%1.1f', fontsize=5)
plt.colorbar()

plt.title("Log-marginal-likelihood")
plt.tight_layout()

plt.show()
