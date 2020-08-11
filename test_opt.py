import scipy.optimize as optimize

def f(x):
  return (x-1)*(x-1)+2 # x^2-2x+3

res = optimize.minimize(f, 10, method='CG')
print (res.x)