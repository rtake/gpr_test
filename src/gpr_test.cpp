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
  double **K; // Covariance matrix
} GPRData;

// double Covfunc_RBF(double theta, double r) { return exp( (-theta*0.5)*r*r ); }

double Covfunc_RBF(double theta, int dim, double *x, double *x_prime) {
  double r=0;
  for(int i=0;i<dim;i++) { r += (x[i]-x_prime[i])*(x[i]-x_prime[i]); }
  r = sqrt(r);
  return exp( (-theta*0.5)*r*r );
}

// double Covfunc_RBF_derivative(double theta, double r) { return -0.5*r*r*exp( (-theta*0.5)*r*r ); }

double Covfunc_RBF_derivative(double theta, int dim, double *x, double *x_prime) {
  double r=0;
  for(int i=0;i<dim;i++) { r += (x[i]-x_prime[i])*(x[i]-x_prime[i]); }
  r = sqrt(r);
  return -0.5*r*r*exp( (-theta*0.5)*r*r );
}

typedef struct Kernel_0 {
  // ConstantKernel() * RBF() + WhiteKernel()
  double theta_0; // for Constant kernel
  double theta_1; // for RBF(Gaussian) kernel
  double theta_2; // for White kernel
} Kernel_0;

double Covfunc_Kernel_0(Kernel_0 *k_0, int dim, double *x, double *x_prime) {
  double r=0;
  for(int i=0;i<dim;i++) { r += (x[i]-x_prime[i])*(x[i]-x_prime[i]); }
  r = sqrt(r);
  return k_0->theta_0 * Covfunc_RBF(k_0->theta_1,dim,x,x_prime) + k_0->theta_2;
}

void Covfunc_Kernel_0_derivative(Kernel_0 *k_0, int dim, double *x, double *x_prime, double *derivative) {
  derivative[0] = Covfunc_RBF(k_0->theta_1,dim,x,x_prime) + k_0->theta_2; 
  derivative[1] = k_0->theta_0 * Covfunc_RBF_derivative(k_0->theta_1,dim,x,x_prime) + k_0->theta_2;
  derivative[2] = k_0->theta_0 * Covfunc_RBF(k_0->theta_1,dim,x,x_prime) + 1;
}


void Kernel_0_fit(Kernel_0 *k_0, GPRData *gpr) {
  int maxcycle = 10000, converged = 0, n=gpr->ndat, dim=gpr->dim, nparam=3;
  double alpha = 0.1;
  VectorXd t; 
  MatrixXd Grad0(n,n), Grad1(n,n), Grad2(n,n), K(n,n);

  for(int cycle = 0;cycle < maxcycle;cycle++) {
    double *updates;
    updates = (double*)malloc(sizeof(double)*nparam);
    
    for(int i=0;i<n;i++) {
      for(int j=0;j<n;j++) {
        double *grad;
        grad = (double*)malloc(sizeof(double)*nparam);

        Covfunc_Kernel_0_derivative(k_0, dim, gpr->x[i], gpr->x[j], grad);
        Grad0(i,j) = grad[0];
        Grad1(i,j) = grad[1];
        Grad2(i,j) = grad[2];

        free(grad);
      }
    }

    // updates = np.array([-np.trace(self.precision.dot(grad)) + t.dot(self.precision.dot(grad).dot(self.precision).dot(t)) for grad in gradients])
  
    updates[0] = -((K.inverse())*Grad0).trace() + t.transpose()*(K.inverse())*Grad0*(K.inverse())*t;
    updates[1] = -((K.inverse())*Grad1).trace() + t.transpose()*(K.inverse())*Grad0*(K.inverse())*t;
    updates[2] = -((K.inverse())*Grad2).trace() + t.transpose()*(K.inverse())*Grad0*(K.inverse())*t;

    k_0->theta_0 -= alpha * updates[0];
    k_0->theta_1 -= alpha * updates[1];
    k_0->theta_2 -= alpha * updates[2];
  }

}


void Kernel_0_predict(Kernel_0 *k_0, double r, double *pred) {
  double mean = 0, var = 0;

  pred[0] = mean;
  pred[1] = var;
}

int main(void) {
  int kerneltype = 0;
  GPRData *gpr;
  gpr = (GPRData*)malloc(sizeof(GPRData));

  // load data

  if(kerneltype == 0) {
    Kernel_0 *k_0;
    k_0 = (Kernel_0*)malloc(sizeof(Kernel_0));
    k_0->theta_0 = 1;
    k_0->theta_1 = 1;
    k_0->theta_2 = 1;

    Kernel_0_fit(k_0, gpr);

    free(k_0);
  }

  free(gpr);
  return 0;
}