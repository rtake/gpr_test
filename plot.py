import numpy as np
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y, z = np.loadtxt("./lml.csv", delimiter=',', unpack=True)

size = int(np.sqrt(x.size))
x = x.reshape(size, size)
y = y.reshape(size, size)
z = z.reshape(size, size)
print(x,y,z)

#x = np.linspace(-5, 5, 101)
#y = np.linspace(-5, 5, 101)
#x,y = np.meshgrid(x, y)

fig = plt.figure()

cont = plt.contour(x,y,z, levels=50)
cont.clabel(fmt='%1.1f', fontsize=8)
plt.colorbar()


"""
ax = Axes3D(fig)
ax.set_zlim(-100,0)
ax.plot(x, y, z, marker="o",linestyle='None')
"""

plt.xlabel("Theta0")
plt.ylabel("Theta1")

plt.show()