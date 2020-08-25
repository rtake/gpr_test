import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct
from sklearn.preprocessing import StandardScaler

ndata = 10

csv_file = open("./h2_10point.csv", "r")
reader = csv.reader(csv_file)
reader_np = np.array(list(reader))

y = np.atleast_2d(reader_np[1:,1]).T.astype(np.float32) # energy
X = reader_np[1:,2:].astype(np.float32) # distance
# print (x)
# print (y)

scaler_y = StandardScaler().fit(y) 

kernel = ConstantKernel() * RBF() + WhiteKernel() 
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0) 
gpr.fit(X, scaler_y.transform(y))

params = kernel.get_params()
for key in sorted(params) : print("%s : %s" % (key, params[key]))
print (kernel.theta)

# print(gpr.kernel_)

plot_X = np.atleast_2d(np.linspace(0, 5, 1000)).T

pred_mu, pred_sigma = gpr.predict(plot_X, return_std=True)
pred_mu = scaler_y.inverse_transform(pred_mu)
pred_sigma = pred_sigma.reshape(-1, 1) * scaler_y.scale_
# print (pred_mu, pred_sigma)

fig = plt.figure(figsize=(8, 6))
plt.plot(X, y, 'r.', markersize=16)
plt.plot(plot_X, pred_mu, 'b')

plt.fill_between(plot_X.squeeze(), (pred_mu - 1.9600 * pred_sigma).squeeze(), (pred_mu + 1.9600 * pred_sigma).squeeze())
plt.xlabel('$distance$', fontsize=16)
plt.ylabel('$energy$', fontsize=16)
plt.xlim(0, 5)
plt.ylim(-1.2, 0)

plt.tick_params(labelsize=16)
plt.show()