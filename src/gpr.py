import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct
from sklearn.preprocessing import StandardScaler

X = np.atleast_2d([float(i) for i in range(10)]).T
y = np.sin(X)

y.reshape(-1,1)

print (X)
print (y)

# hyperparams = gpr.get_params(deep=False)

scaler_y = StandardScaler().fit(y) 

kernel = ConstantKernel() * RBF() + WhiteKernel() 
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0)

# print (gpr.log_marginal_likelihood())

# params = gpr.kernel.get_params()
# for key in sorted(params) : print("%s : %s" % (key, params[key]))


# print (gpr.kernel.hyperparameters)
# print (gpr.score(X,y))

gpr.fit(X, scaler_y.transform(y))
print (gpr.log_marginal_likelihood())

# params = gpr.kernel.get_params()
# for key in sorted(params) : print("%s : %s" % (key, params[key]))

print (gpr.kernel)
print (gpr.kernel.theta)

print (gpr.kernel_)
print (gpr.kernel_.theta)
