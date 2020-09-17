import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib import ticker, cm, colors
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct
from sklearn.preprocessing import StandardScaler

def Muller_Brown(x,y):
  energy=0
  A = [-200,-100,-170,15]
  a = [-1,-1,-6.5,0.7]
  b = [0,0,11,0.6]
  c = [-10,-10,-6.5,0.7]

  x0 = [ [1,0],[0,0.5],[-0.5,1.5],[-1,1] ] 

  for i in range(0,4):
    val = 0
    val += a[i]*(x-x0[i][0])*(x-x0[i][0])
    val += b[i]*(x-x0[i][0])*(y-x0[i][1])
    val += c[i]*(y-x0[i][1])*(y-x0[i][1])
    energy += A[i]*np.exp(val)

  return energy

# Grid
# X = [[0,0],[0,0.5],[0,1],[0.5,0],[0.5,1],[0.5,0.5],[-1,1],[-0.5,0.5],[-0.5,1],[-0.5,1.5],[1,0]]

# 3 points
# X = [[-0.5,1.5],[0,0.75],[0.5,0]]

# 4 points
# X = [[-0.5,1.5],[-0.17,1],[0.17,0.5],[0.5,0]] 

# 5 points
# X = [[-0.5,1.5],[-0.25,1.125],[0,0.75],[0.25,0.375],[0.5,0]]

# 6 points
# X = [[-0.5,1.5],[-0.33,1.25],[-0.17,1],[0,0.75],[0.17,0.5],[0.33,0.25],[0.5,0]]

# 8 points
# X = [[-0.5,1.5],[-0.375,1.3125],[-0.25,1.125],[-0.125,0.9375],[0,0.75],[0.125,0.5625],[0.25,0.375],[0.375,0.1875],[0.5,0]]


# 10 points
X = [[-0.5,1.5],[-0.4,1.35],[-0.3,1.2],[-0.2,1.05],[-0.1,0.9],[0,0.75],[0.1,0.6],[0.2,0.45],[0.3,0.3],[0.4,0.15],[0.5,0]]


# AFIR path
# 4 points
# X = [[-0.5,1.5],[-1,1],[-0.5,0.5],[0.5,0]]

# 5 points
# X = [[-0.5,1.5],[-1,1],[-0.5,0.5],[0,0.5],[0.5,0]]


y = []
for x in X:
  y.append( [Muller_Brown(x[0],x[1])] )

print (X)
print (y)

scaler_y = StandardScaler().fit(y)
kernel = ConstantKernel() * RBF() + WhiteKernel() 
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0) 



gpr.fit(X, scaler_y.transform(y))

print (gpr.kernel)
print (gpr.kernel_)
#print (gpr.log_marginal_likelihood(theta=[-0.012811,-1.163772,-1.145052], eval_gradient=True))
# print (gpr.log_marginal_likelihood(theta=[-0.002813,-0.014599,-0.006062], eval_gradient=True))
# print (gpr.log_marginal_likelihood(theta=[-0.000003,-0.000014,-0.000006], eval_gradient=True))

#print (gpr.log_marginal_likelihood(theta=[0.084148,-1.314127,-7.755164], eval_gradient=True))

# print (gpr.log_marginal_likelihood(theta=[0,0,0], eval_gradient=True))

# print (gpr.log_marginal_likelihood())

K, K_gradient = kernel(gpr.X_train_, eval_gradient=True)
print (K)
print (K_gradient)

gridnum = 100
x0lim = [-1.5,1.5]
x1lim = [-0.5,2]
x0 = np.arange(x0lim[0], x0lim[1], (x0lim[1]-x0lim[0])/gridnum)
x1 = np.arange(x1lim[0], x1lim[1], (x1lim[1]-x1lim[0])/gridnum)
plot_x0, plot_x1 = np.meshgrid(x0,x1)
plot_x =  np.array([plot_x0.ravel(), plot_x1.ravel()]).T

plot_y, sigma = gpr.predict(plot_x, return_std=True)
plot_y = scaler_y.inverse_transform(plot_y)

Y = gpr.predict(X)
Y = scaler_y.inverse_transform(Y)
X0 = []
X1 = []
for x in X:
  X0.append(x[0])
  X1.append(x[1])



fig = plt.figure(figsize=(8, 5))

plt.plot(X0, X1, 'r.', markersize=3)
#"""
cnt = plt.contourf(plot_x0, plot_x1, plot_y.reshape(gridnum, -1))
# cnt = plt.contourf(plot_x0, plot_x1, sigma.reshape(gridnum, -1))

# cnt.set_clim(-150,100)
cbar = plt.colorbar(cnt)
#"""


"""
ax = fig.add_subplot(111, projection='3d')
ax.plot(X0, X1, Y.reshape(1,-1)[0], marker=".", color="r", linestyle='None')
surf = ax.plot_surface(plot_x0, plot_x1, plot_y.reshape(gridnum, -1), cmap=cm.coolwarm)
cbar = plt.colorbar(surf)
"""

# For surface plot
"""
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(plot_x0, plot_x1, pred_y.reshape(gridnum, -1), cmap='bwr', linewidth=0)
"""


plt.tick_params(labelsize=8)
plt.xlabel('$x0$', fontsize=8)
plt.ylabel('$x1$', fontsize=8)
plt.show()


