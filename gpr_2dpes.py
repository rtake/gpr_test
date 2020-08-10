import numpy as np
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct
from sklearn.preprocessing import StandardScaler

# ndata = 10

csv_file = open("./h2o_2d_1000.csv", "r")
reader = csv.reader(csv_file)
reader_np = np.array(list(reader))

y = np.atleast_2d(reader_np[1:,1]).T.astype(np.float32) # energy
X = reader_np[1:,2:].astype(np.float32) # distance
# print (X)
# print (y)

scaler_y = StandardScaler().fit(y) 

kernel = ConstantKernel() * RBF() + WhiteKernel() 
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0) 
gpr.fit(X, scaler_y.transform(y))
# print(gpr.kernel_)

gridnum = 100;
xlim = [0,5]
ylim = [0,5]
x = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/gridnum)
y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/gridnum)
plot_X, plot_Y = np.meshgrid(x,y)

plot_XY =  np.array([plot_X.ravel(), plot_Y.ravel()]).T
# print (plot_X)
# print (plot_Y)
# print (plot_XY)

pred_mu, pred_sigma = gpr.predict(plot_XY, return_std=True) # posterior average
pred_mu = scaler_y.inverse_transform(pred_mu)
pred_sigma = pred_sigma.reshape(-1, 1) * scaler_y.scale_
# print (pred_mu, pred_sigma)

# print (plot_XY[:,0])
# print (plot_XY[:,1])
# print (plot_XY)
# print (pred_mu)

# For plot
fig = plt.figure(figsize=(8, 6))
plt.xlabel('$x0$', fontsize=16)
plt.ylabel('$x1$', fontsize=16)
# plt.xlim(xlim)
# plt.ylim(ylim)
plt.tick_params(labelsize=16)

# For contour plot
# """
plt.plot(X[:,0], X[:,1], 'r.', markersize=3)
plt.contourf(plot_X, plot_Y, pred_mu.reshape(gridnum, -1))
plt.colorbar()
# """

# For surface plot
"""
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(plot_X, plot_Y, pred_mu.reshape(gridnum, -1), cmap='bwr', linewidth=0)
"""

plt.show()