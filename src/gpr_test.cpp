# include <bits/stdc++.h>
# include <Eigen/Core>
# include <Eigen/LU>

# include "gpr_test.hpp"

using namespace std;
using namespace Eigen;


double Covfunc_RBF(double theta, int dim, double *x, double *x_prime) {
  double r=0;
  for(int i=0;i<dim;i++) { r += (x[i]-x_prime[i])*(x[i]-x_prime[i]); }
  r = sqrt(r);
  return exp( (-theta*0.5)*r*r );
}


double Covfunc_RBF_derivative(double theta, int dim, double *x, double *x_prime) {
  double r=0;
  for(int i=0;i<dim;i++) { r += (x[i]-x_prime[i])*(x[i]-x_prime[i]); }
  r = sqrt(r);
  return -0.5*r*r*exp( (-theta*0.5)*r*r );
}


void Kernel_0_MakeKernelMatrix(GPRData *gpr, Kernel_0 *k_0) {
  for(int i=0;i<gpr->ndat;i++) {
    for(int j=0;j<gpr->ndat;j++) {
      gpr->K[i][j] = Covfunc_Kernel_0(k_0, gpr->dim, gpr->x[i], gpr->x[j]);
      if(i == j) { gpr->K[i][j] += gpr->sigma * gpr->sigma; }
    }
  }
}


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
  int maxcycle=10000, converged=0, n=gpr->ndat, dim=gpr->dim, nparam=3;
  double alpha=0.1;
  VectorXd y = Map<VectorXd>(gpr->y, n); // training data
  MatrixXd Grad0(n,n), Grad1(n,n), Grad2(n,n), K;

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

    Kernel_0_MakeKernelMatrix(gpr, k_0);
    K = Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(&gpr->K[0][0], n, n);
 
    updates[0] = -((K.inverse())*Grad0).trace() + ((K.inverse()*y).transpose())*Grad0*(K.inverse()*y);
    updates[1] = -((K.inverse())*Grad1).trace() + ((K.inverse()*y).transpose())*Grad1*(K.inverse()*y);
    updates[2] = -((K.inverse())*Grad2).trace() + ((K.inverse()*y).transpose())*Grad2*(K.inverse()*y);
    
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
  GPRData gpr;

  /* Initialize GPR */
  gpr.dim = 1;
  gpr.ndat = 10;
  gpr.x = (double**)malloc(sizeof(double*)*gpr.ndat*gpr.dim);
  gpr.y = (double*)malloc(sizeof(double)*gpr.ndat);
  gpr.K = (double**)malloc(sizeof(double*)*gpr.ndat*gpr.ndat);
  
  /* Load data */
  for(int i=0;i<gpr.ndat;i++) {
    gpr.x[i][0] = (double)i;
    gpr.y[i] = sin(gpr.x[i][0]);
  }

  if(kerneltype == 0) {
    Kernel_0 k_0;
    k_0.theta_0 = 1;
    k_0.theta_1 = 1;
    k_0.theta_2 = 1;

    Kernel_0_fit(&k_0, &gpr);
  }

  free(gpr.x);
  free(gpr.y);
  free(gpr.K);

  return 0;
}