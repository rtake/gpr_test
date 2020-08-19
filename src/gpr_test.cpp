# include <bits/stdc++.h>
# include <Eigen/Core>
# include <Eigen/LU>
# include <Eigen/Dense>


using namespace std;
using namespace Eigen;


typedef struct GaussianProcessRegressor {
  int ndat; // the number of data
  int dim; // dimenson of data
  // int ndim_dat; // dimension of data
  double **x; // x[ndat][ndim]
  double *y; // y[ndat]
  double **K; // Kernel Matrix
  double **K_derivative;
  double sigma;

  int kerneltype;
  int nparam; // number of parameter
  double *theta;
  /*
  kerneltype == 0  ->  ConstantKernel() * RBF() + WhiteKernel(), nparam = 3
  kerneltype == 1  ->  ConstantKernel() * RBF(), nparam = 2
  */
} GaussianProcessRegressor;


double Covfunc_RBF(double l, int dim, double *x, double *x_prime) {
  double r=0;
  for(int i=0;i<dim;i++) { r += (x[i]-x_prime[i])*(x[i]-x_prime[i]); }
  r = sqrt(r);
  return exp( -(r*r)/(2*l*l) );
}


double Covfunc_RBF_derivative(double l, int dim, double *x, double *x_prime) {
  double r=0;
  for(int i=0;i<dim;i++) { r += (x[i]-x_prime[i])*(x[i]-x_prime[i]); }
  r = sqrt(r);
  return (r*r)/(l*l*l)*Covfunc_RBF(l,dim,x,x_prime);
}


void GaussianProcessRegressor_MakeKernelMatrix(GaussianProcessRegressor *gpr) {
  for(int i=0;i<gpr->ndat;i++) {
    for(int j=0;j<gpr->ndat;j++) {
      if(gpr->kerneltype == 0) { gpr->K[i][j] = gpr->theta[0] * Covfunc_RBF(gpr->theta[1], gpr->dim, gpr->x[i], gpr->x[j]) + gpr->theta[2]; }
      else if(gpr->kerneltype == 1) { gpr->K[i][j] = gpr->theta[0] * Covfunc_RBF(gpr->theta[1], gpr->dim, gpr->x[i], gpr->x[j]); }
      // printf("%d %d --> %lf\n", i, j, gpr->K[i][j]); 
      if(i == j) { gpr->K[i][j] += gpr->sigma * gpr->sigma; }
    }
  }
}


double GaussianProcessRegressor_LogMarginalLikelihood(GaussianProcessRegressor *gpr) {
  VectorXd y = Map<VectorXd>(gpr->y, gpr->ndat); 
  MatrixXd K = Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(&gpr->K[0][0], gpr->ndat, gpr->ndat);
  double lml_0, lml_1, lml_2, pi=acos(-1);

  /*
  LLT<MatrixXd> lltofK(K);
  MatrixXd L = lltofK.matrixL();
  VectroXd alpha = 
  */

  lml_0 = -0.5*(y.transpose())*(K.inverse())*y; // here
  lml_1 = -0.5*log(K.determinant()); // ok
  lml_2 = -0.5*gpr->ndat*log(2*pi); // ok
  
  /*
  cout << "y.transpose()" << endl << y.transpose() << endl;
  cout << "K.inverse()" << endl << K.inverse() << endl;
  cout << "y" << endl << y << endl;
  printf("LML : %lf, %lf, %lf\n", lml_0, lml_1, lml_2);
  */
 
  return lml_0 + lml_1 + lml_2;
}


void GaussianProcessRegressor_fit(GaussianProcessRegressor *gpr) {
  int maxcycle=10, converged=0, n=gpr->ndat, dim=gpr->dim, nparam=gpr->nparam;
  double alpha=0.01;
  
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

      printf("cycle %d : %lf\n", cycle, GaussianProcessRegressor_LogMarginalLikelihood(gpr));
      cout << K << endl;

      updates[0] = ((K.inverse())*Grad0).trace() + ((K.inverse()*y).transpose())*Grad0*(K.inverse()*y);
      updates[1] = ((K.inverse())*Grad1).trace() + ((K.inverse()*y).transpose())*Grad1*(K.inverse()*y);
      updates[2] = ((K.inverse())*Grad2).trace() + ((K.inverse()*y).transpose())*Grad2*(K.inverse()*y);
      printf("update: %lf, %lf, %lf\n", updates[0], updates[1], updates[2]);
    
      for(int i=0;i<nparam;i++) { gpr->theta[i] -= alpha * updates[i]; }
      printf("theta: %lf, %lf, %lf\n", gpr->theta[0], gpr->theta[1], gpr->theta[2]);
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

      printf("cycle %d : %lf\n", cycle, GaussianProcessRegressor_LogMarginalLikelihood(gpr));
      printf("theta(log-transformed): %lf, %lf\n", log(gpr->theta[0]), log(gpr->theta[1]));
      cout << K << endl; // ok

      updates[0] = -((K.inverse())*Grad0).trace() + ((K.inverse()*y).transpose())*Grad0*(K.inverse()*y);
      updates[1] = -((K.inverse())*Grad1).trace() + ((K.inverse()*y).transpose())*Grad1*(K.inverse()*y);
      printf("update: %lf, %lf\n", updates[0], updates[1]);
    
      for(int i=0;i<nparam;i++) { gpr->theta[i] += alpha * updates[i]; }
      
    }
  }

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


void GaussianProcessRegressor_StandardScaler(GaussianProcessRegressor *gpr) {
  int n=gpr->ndat;
  double y_mean=0, y_var=0;

  for(int i=0;i<n;i++) { y_mean += gpr->y[i]; }
  y_mean /= n;
  for(int i=0;i<n;i++) { y_var += (gpr->y[i]-y_mean)*(gpr->y[i]-y_mean); }
  y_var = y_var/n;
  for(int i=0;i<n;i++) { gpr->y[i] = (gpr->y[i]-y_mean)/sqrt(y_var); }

  // printf("mean : %lf, var : %lf\n", y_mean, y_var);
}


void GaussianProcessRegressor_Free(GaussianProcessRegressor *gpr) {
  free(gpr->x);
  free(gpr->y);
  free(gpr->K);
  free(gpr->theta);
}

/*
void Kernel_0_MakeKernelMatrix(GaussianProcessRegressor *gpr, Kernel_0 *k_0) {
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
*/


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

  /* Scaling training data */
  /*
  for(int i=0;i<gpr->ndat;i++) { printf("%lf,", gpr->y[i]); }
  cout << endl;  
  */

  GaussianProcessRegressor_StandardScaler(gpr);
  
  /*
  for(int i=0;i<gpr->ndat;i++) { printf("%lf,", gpr->y[i]); }
  cout << endl;
  */  
  
  if(gpr->kerneltype == 0) {
    /*
    Kernel_0 *k_0;
    k_0 = (Kernel_0*)malloc(sizeof(Kernel_0));

    k_0->theta_0 = 1;
    k_0->theta_1 = 1;
    k_0->theta_2 = 1;

    Kernel_0_fit(k_0, gpr);
    printf("%lf, %lf, %lf\n", k_0->theta_0, k_0->theta_1, k_0->theta_2);

    free(k_0);
    */
  } else if(gpr->kerneltype == 1) {
    gpr->nparam = 2;
    gpr->theta = (double*)malloc(sizeof(double)*gpr->nparam);
    gpr->sigma = 0.1;
    gpr->theta[0] = 1;
    gpr->theta[1] = 1;

    GaussianProcessRegressor_fit(gpr);
    printf("%lf, %lf\n", log(gpr->theta[0]), log(gpr->theta[1]));

    FILE *fp = fopen("lml.csv", "w");

    for(double i=-5;i<=5;i+=0.1) {
      for(double j=-5;j<=5;j+=0.1) {
        GaussianProcessRegressor gpr0 = *gpr;
        gpr0.theta[0] = exp(j); // x
        gpr0.theta[1] = exp(i); // y
        GaussianProcessRegressor_MakeKernelMatrix(&gpr0);
        fprintf(fp,"%lf,%lf,%lf\n", j, i, GaussianProcessRegressor_LogMarginalLikelihood(&gpr0));
      } 
    }
  }

  GaussianProcessRegressor_Free(gpr);
  free(gpr);

  return 0;
}