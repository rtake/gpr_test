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
    val += a[i]*(x-x0[i][0])*(x-x0[i][0]); 
    val += b[i]*(x-x0[i][0])*(y-x0[i][1]); 
    val += c[i]*(y-x0[i][1])*(y-x0[i][1]); 
    energy += A[i]*np.exp(val);

  return energy

x0 = []
x1 = []
for i in range(-3,3):
  for j in range(-3,3):
    x0.append(i)
    x1.append(j)

y = []
X = []
for i in x0:
  for j in x1:
    y.append([Muller_Brown(i,j)])
    X.append([i,j])

#y = y.reshape(-1,1)

scaler_y = StandardScaler().fit(y)
kernel = ConstantKernel() * RBF() + WhiteKernel() 
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0) 
gpr.fit(X, scaler_y.transform(y))


gridnum = 100
x0lim = [-3,3]
x1lim = [-3,3]
x0 = np.arange(x0lim[0], x0lim[1], (x0lim[1]-x0lim[0])/gridnum)
x1 = np.arange(x1lim[0], x1lim[1], (x1lim[1]-x1lim[0])/gridnum)
plot_x0, plot_x1 = np.meshgrid(x0,x1)
plot_x =  np.array([plot_x0.ravel(), plot_x1.ravel()]).T

plot_y = gpr.predict(plot_x)
#plot_y = scaler_y.inverse_transform(plot_y)

fig = plt.figure(figsize=(4, 3))
plt.xlabel('$x0$', fontsize=8)
plt.ylabel('$x1$', fontsize=8)
# plt.xlim(x0lim)
# plt.ylim(x1lim)


# For contour plot
# """

# plt.plot(X[:,0], X[:,1], 'r.', markersize=3)

# plt.contourf(plot_x0, plot_x1, plot_y.reshape(gridnum, -1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(plot_x0, plot_x1, plot_y.reshape(gridnum, -1), cmap=cm.coolwarm)

print (plot_y)

cbar = plt.colorbar(surf)
cbar.set_label('energy(harteree)')
plt.tick_params(labelsize=8)

# """


# For surface plot
"""
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(plot_x0, plot_x1, pred_y.reshape(gridnum, -1), cmap='bwr', linewidth=0)
"""

plt.show()


