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


void Kernel_0_MakeKernelMatrix(GaussianProcessRegressor *gpr, Kernel_0 *k_0) {
  for(int i=0;i<gpr->ndat;i++) {
    for(int j=0;j<gpr->ndat;j++) {
      gpr->K[i][j] = Covfunc_Kernel_0(k_0, gpr->dim, gpr->x[i], gpr->x[j]);
      if(i == j) { gpr->K[i][j] += gpr->sigma * gpr->sigma; }
    }
  }
}


void GaussianProcessRegressor_MakeKernelMatrix(GaussianProcessRegressor *gpr) {
  for(int i=0;i<gpr->ndat;i++) {
    for(int j=0;j<gpr->ndat;j++) {
      if(gpr->kerneltype == 0) { gpr->K[i][j] = gpr->theta[0] * Covfunc_RBF(gpr->theta[1], gpr->dim, gpr->x[i], gpr->x[j]) + gpr->theta[2]; }
      else if(gpr->kerneltype == 1) { gpr->K[i][j] = gpr->theta[0] * Covfunc_RBF(gpr->theta[1], gpr->dim, gpr->x[i], gpr->x[j]); }

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


void Kernel_0_fit(Kernel_0 *k_0, GaussianProcessRegressor *gpr) {
  int maxcycle=10, converged=0, n=gpr->ndat, dim=gpr->dim, nparam=3;
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
    printf("cycle %d update: %lf, %lf, %lf\n", cycle, updates[0], updates[1], updates[2]);
    
    k_0->theta_0 -= alpha * updates[0];
    k_0->theta_1 -= alpha * updates[1];
    k_0->theta_2 -= alpha * updates[2];
    printf("cycle %d theta: %lf, %lf, %lf\n", cycle, k_0->theta_0, k_0->theta_1, k_0->theta_2);
  }
}


void GaussianProcessRegressor_fit(GaussianProcessRegressor *gpr) {
  int maxcycle=10, converged=0, n=gpr->ndat, dim=gpr->dim, nparam=gpr->nparam;
  double alpha=0.1;
  
  if(gpr->kerneltype == 0) {
    VectorXd y = Map<VectorXd>(gpr->y, n); // training data
    MatrixXd Grad0(n,n), Grad1(n,n), Grad2(n,n), K;

    for(int cycle=0;cycle<maxcycle;cycle++) {
      double *updates;
      updates = (double*)malloc(sizeof(double)*nparam);
    
      for(int i=0;i<n;i++) {
        for(int j=0;j<n;j++) {
          Grad0(i,j) = Covfunc_RBF(gpr->theta[0], dim, gpr->x[i], gpr->x[j]) + gpr->theta[2]; 
          Grad1(i,j) = gpr->theta[0] * Covfunc_RBF_derivative(gpr->theta[1], dim, gpr->x[i], gpr->x[j]) + gpr->theta[2];
          Grad2(i,j) = gpr->theta[0] * Covfunc_RBF(gpr->theta[0], dim, gpr->x[i], gpr->x[j]) + 1;
        }
      }

      GaussianProcessRegressor_MakeKernelMatrix(gpr);
      K = Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(&gpr->K[0][0], n, n);

      updates[0] = -((K.inverse())*Grad0).trace() + ((K.inverse()*y).transpose())*Grad0*(K.inverse()*y);
      updates[1] = -((K.inverse())*Grad1).trace() + ((K.inverse()*y).transpose())*Grad1*(K.inverse()*y);
      updates[2] = -((K.inverse())*Grad2).trace() + ((K.inverse()*y).transpose())*Grad2*(K.inverse()*y);
      printf("cycle %d update: %lf, %lf, %lf\n", cycle, updates[0], updates[1], updates[2]);
    
      for(int i=0;i<nparam;i++) { gpr->theta[i] -= alpha * updates[i]; }
      printf("cycle %d theta: %lf, %lf, %lf\n", cycle, gpr->theta[0], gpr->theta[1], gpr->theta[2]);
    }
  } else if(gpr->kerneltype == 1) {
    VectorXd y = Map<VectorXd>(gpr->y, n); // training data
    MatrixXd Grad0(n,n), Grad1(n,n), K;

    for(int cycle=0;cycle<maxcycle;cycle++) {
      double *updates;
      updates = (double*)malloc(sizeof(double)*nparam);
    
      for(int i=0;i<n;i++) {
        for(int j=0;j<n;j++) {
          Grad0(i,j) = Covfunc_RBF(gpr->theta[0], dim, gpr->x[i], gpr->x[j]); 
          Grad1(i,j) = gpr->theta[0] * Covfunc_RBF_derivative(gpr->theta[1], dim, gpr->x[i], gpr->x[j]);
        }
      }

      GaussianProcessRegressor_MakeKernelMatrix(gpr);
      K = Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(&gpr->K[0][0], n, n);

      updates[0] = -((K.inverse())*Grad0).trace() + ((K.inverse()*y).transpose())*Grad0*(K.inverse()*y);
      updates[1] = -((K.inverse())*Grad1).trace() + ((K.inverse()*y).transpose())*Grad1*(K.inverse()*y);
      printf("cycle %d update: %lf, %lf, %lf\n", cycle, updates[0], updates[1], updates[2]);
    
      for(int i=0;i<nparam;i++) { gpr->theta[i] -= alpha * updates[i]; }
      printf("cycle %d theta: %lf, %lf, %lf\n", cycle, gpr->theta[0], gpr->theta[1], gpr->theta[2]);
    }
  }

}


void Kernel_0_predict(Kernel_0 *k_0, double r, double *pred) {
  double mean = 0, var = 0;

  pred[0] = mean;
  pred[1] = var;
}


void GaussianProcessRegressor_Malloc(GaussianProcessRegressor *gpr) {
  int n=gpr->ndat, m=gpr->dim;

  gpr->x = (double**)malloc(sizeof(double*)*n);
  gpr->x[0] = (double*)malloc(sizeof(double)*n*m); 
  for(int i=0;i<n;i++) { gpr->x[i] = gpr->x[0] + i*m; }

  gpr->y = (double*)malloc(sizeof(double)*n);

  gpr->K = (double**)malloc(sizeof(double*)*n);
  gpr->K[0] = (double*)malloc(sizeof(double*)*n*n);
  for(int i=0;i<n;i++) { gpr->K[i] = gpr->K[0] + i*n; }
}


void GaussianProcessRegressor_Free(GaussianProcessRegressor *gpr) {
  free(gpr->x);
  free(gpr->y);
  free(gpr->K);
}


int main(void) {
  GaussianProcessRegressor *gpr;
  gpr = (GaussianProcessRegressor*)malloc(sizeof(GaussianProcessRegressor));

  /* Initialize GPR */
  gpr->dim = 1;
  gpr->ndat = 10;
  gpr->kerneltype = 1;
  GaussianProcessRegressor_Malloc(gpr); 

  /* Load data */
  for(int i=0;i<gpr->ndat;i++) {
    gpr->x[i][0] = (double)i;
    gpr->y[i] = sin(gpr->x[i][0]);
  }

  if(kerneltype == 0) {
    Kernel_0 *k_0;
    k_0 = (Kernel_0*)malloc(sizeof(Kernel_0));

    k_0->theta_0 = 1;
    k_0->theta_1 = 1;
    k_0->theta_2 = 1;

    Kernel_0_fit(k_0, gpr);
    printf("%lf, %lf, %lf\n", k_0->theta_0, k_0->theta_1, k_0->theta_2);

    free(k_0);
  } else if(gpr->kerneltype == 1) {
    gpr->sigma = 0.7;
    gpr->theta[0] = 1;
    gpr->theta[1] = 1;

    GaussianProcessRegressor_fit(gpr);
    printf("%lf, %lf, %lf\n", gpr->theta[0], gpr->theta[1]);
  }

  GaussianProcessRegressor_Free(gpr);
  free(gpr);

  return 0;
}