# include <bits/stdc++.h>
# include <Eigen/Core>
# include <Eigen/LU>

using namespace std;
using namespace Eigen;

typedef struct GPRData {
  int ndat; // the number of data
  int dim; // dimenson of data
  // int ndim_dat; // dimension of data
  double **x; // x[ndat][ndim]
  double *y; // y[ndat]
  double **K; // Kernel Matrix
  double sigma;
} GPRData;


typedef struct Kernel_0 {
  // ConstantKernel() * RBF() + WhiteKernel()
  double theta_0; // for Constant kernel
  double theta_1; // for RBF(Gaussian) kernel
  double theta_2; // for White kernel
} Kernel_0;


double Covfunc_RBF(double theta, int dim, double *x, double *x_prime);
double Covfunc_RBF_derivative(double theta, int dim, double *x, double *x_prime);
void Kernel_0_MakeKernelMatrix(GPRData *gpr, Kernel_0 *k_0);
double Covfunc_Kernel_0(Kernel_0 *k_0, int dim, double *x, double *x_prime);
void Covfunc_Kernel_0_derivative(Kernel_0 *k_0, int dim, double *x, double *x_prime, double *derivative);
void Kernel_0_fit(Kernel_0 *k_0, GPRData *gpr) ;
void Kernel_0_predict(Kernel_0 *k_0, double r, double *pred);
