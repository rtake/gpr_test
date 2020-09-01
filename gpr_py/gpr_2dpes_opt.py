import numpy as np
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct
from sklearn.preprocessing import StandardScaler


# Road data
csv_file = open("./h2o_2d_1000.csv", "r")
reader = csv.reader(csv_file)
reader_np = np.array(list(reader))

y = np.atleast_2d(reader_np[1:,1]).T.astype(np.float32)
X = reader_np[1:,2:].astype(np.float32)

scaler_y = StandardScaler().fit(y)  


# Initialize Gp regressor
kernel = ConstantKernel() * RBF() + WhiteKernel() 
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0) 


# Optimization
maxoptcycle=10000
converged=None
dim=2
stepsize=0.01
alpha = 0.01
threshold = 0.001
diff = [stepsize,-stepsize]

xlist = np.zeros((1,dim))
xlist[0][0] = 0.47
xlist[0][1] = 0.69


# regression
gpr.fit(X, scaler_y.transform(y))


# optimization
cycle = 0
while cycle < maxoptcycle:
  if converged != None: 
    break

  # gradient  
  gradient = np.zeros(dim)

  i = 0
  while i < dim:
    x = np.empty((0,dim))

    j = 0
    while j < 2:
      x0 = xlist[cycle].copy()
      x0[i] += diff[j]
      x = np.append(x, [x0], axis=0)
      j += 1

    y0 = gpr.predict(x)
    y0 = scaler_y.inverse_transform(y0)
    gradient[i] = (y0[0]-y0[1])/stepsize
    i += 1

  xlist = np.append(xlist, [xlist[cycle]-alpha*gradient], axis=0)
  # print (xlist)

  if all(gradient) < threshold:
    print ("converged")
    break

  cycle += 1


# Make grid
gridnum = 100
x0lim = [0,5]
x1lim = [0,5]
x0 = np.arange(x0lim[0], x0lim[1], (x0lim[1]-x0lim[0])/gridnum)
x1 = np.arange(x1lim[0], x1lim[1], (x1lim[1]-x1lim[0])/gridnum)
plot_x0, plot_x1 = np.meshgrid(x0,x1)
plot_x =  np.array([plot_x0.ravel(), plot_x1.ravel()]).T

plot_y = gpr.predict(plot_x)
plot_y = scaler_y.inverse_transform(plot_y)

# For plot
fig = plt.figure(figsize=(4, 3))
plt.xlabel('$x0$', fontsize=8)
plt.ylabel('$x1$', fontsize=8)
# plt.xlim(x0lim)
# plt.ylim(x1lim)


# For contour plot
# """

# plt.plot(X[:,0], X[:,1], 'r.', markersize=3)

plt.plot(xlist[:,0], xlist[:,1], 'r.', markersize=3)
plt.contourf(plot_x0, plot_x1, plot_y.reshape(gridnum, -1))

cbar = plt.colorbar()
cbar.set_label('energy(harteree)')
plt.tick_params(labelsize=8)

# """


# For surface plot
"""
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(plot_x0, plot_x1, pred_y.reshape(gridnum, -1), cmap='bwr', linewidth=0)
"""

plt.show()