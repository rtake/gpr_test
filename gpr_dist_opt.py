import math
import numpy as np
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct
from sklearn.preprocessing import StandardScaler


def Cartesian_to_Distance(cartesian):
  natom = cartesian.shape[0] # number of atoms
  distances = np.empty(0)

  for i in range(0,natom):
    for j in range(i+1,natom):
      atom_i = cartesian[i]
      atom_j = cartesian[j]
      distance = math.sqrt( (atom_i[0]-atom_j[0])**2 + (atom_i[1]-atom_j[1])**2 + (atom_i[2]-atom_j[2])**2 )
      distances = np.append(distances, [distance], axis=0)
  
  return distances


# Road data
csv_file = open("./h2o_dist.csv", "r")
reader = csv.reader(csv_file)
reader_np = np.array(list(reader))

y = np.atleast_2d(reader_np[1:,1]).T.astype(np.float32)
X = reader_np[1:,2:].astype(np.float32)

scaler_y = StandardScaler().fit(y)  


# Optimization
maxoptcycle = 1000000
converged = False
natom = 3
dim = natom*3
stepsize = 0.01
alpha = 0.000001
threshold = 0.01

mols = np.zeros((0,natom,3))
m_0 = np.array([[0,0,0],[1,0,0],[0,1,0]])
mols = np.append(mols, [m_0], axis=0)


# regression
kernel = ConstantKernel() * RBF() + WhiteKernel() 
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0) 
gpr.fit(X, scaler_y.transform(y))


# optimization
for cycle in range(0,maxoptcycle):
  if converged == True: 
    break

  converged = True

  # gradient  
  gradient = np.zeros((natom,3))
  for i in range(0,natom):
    if i == 0:
      continue
    
    for j in range(0,3):
      distances_list = np.empty((0,int(natom*(natom-1)/2)))
      m0 = mols[cycle].copy()
      m1 = mols[cycle].copy()
      
      m0[i][j] += stepsize
      m1[i][j] -= stepsize

      distances_list = np.append(distances_list, [Cartesian_to_Distance(m0)], axis=0)
      distances_list = np.append(distances_list, [Cartesian_to_Distance(m1)], axis=0)

      # print (distances_list)
      y0 = gpr.predict(distances_list)
      y0 = scaler_y.inverse_transform(y0)
      gradient[i][j] = (y0[0]-y0[1])/stepsize

  mols = np.append(mols, [mols[cycle]-alpha*gradient], axis=0)
  
  print ("crd")
  print (mols[cycle])
  print ("grad")
  print (gradient)

  for grad in gradient:
    for g in grad:
      if abs(g) > threshold:
        converged = False

  if converged == True:
    print ("converged")
    break


