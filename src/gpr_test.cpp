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
  double alpha;

  int kerneltype;
  int nparam; // number of parameter
  double *theta;
  /*
  kerneltype == 0  ->  ConstantKernel() * RBF() + WhiteKernel(), nparam = 3
  kerneltype == 1  ->  ConstantKernel() * RBF(), nparam = 2
  */
   
  /* For scaling (GaussianProcessRegressor_StandardScaler) */
  double mu_train; // 
  double sigma_train; // 

  /* for prediction */
  int npred;
  double **test;
  double *mu_pred; // 
  double *sigma_pred;
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
      if(gpr->kerneltype == 0) {
        gpr->K[i][j] = gpr->theta[0] * Covfunc_RBF(gpr->theta[1], gpr->dim, gpr->x[i], gpr->x[j]);
        if(i == j) { gpr->K[i][j] += gpr->theta[2]; }
      }
      else if(gpr->kerneltype == 1) {
        gpr->K[i][j] = gpr->theta[0] * Covfunc_RBF(gpr->theta[1], gpr->dim, gpr->x[i], gpr->x[j]);
        if(i == j) { gpr->K[i][j] += gpr->alpha * gpr->alpha; }  
      }      
    }
  }
}


double GaussianProcessRegressor_LogMarginalLikelihood(GaussianProcessRegressor *gpr) {
  VectorXd y = Map<VectorXd>(gpr->y, gpr->ndat); 
  MatrixXd K = Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(&gpr->K[0][0], gpr->ndat, gpr->ndat);
  double lml_0, lml_1, lml_2, pi=acos(-1);

  lml_0 = -0.5*(y.transpose())*(K.inverse())*y;
  lml_1 = -0.5*log(K.determinant()); 
  lml_2 = -0.5*gpr->ndat*log(2*pi); 

  printf("lml_0 : %17.12lf, lml_1 : %17.12lf, lml_2 : %17.12lf\n", lml_0, lml_1, lml_2);
  return lml_0 + lml_1 + lml_2;
}


void GaussianProcessRegressor_fit(GaussianProcessRegressor *gpr) {
  int maxcycle=1000000, n=gpr->ndat, dim=gpr->dim, nparam=gpr->nparam, converged=0;
  double alpha=0.00001, threshold=0.001;
  VectorXd y; // training data
  
  if(gpr->kerneltype == 0) {
    MatrixXd Grad0(n,n), Grad1(n,n), Grad2(n,n), K;

    for(int cycle=0;cycle<maxcycle;cycle++) {

      /* evaluate gradient for Kernel matrix */
      double updates[nparam];
      for(int i=0;i<n;i++) {
        for(int j=0;j<n;j++) {
          Grad0(i,j) = Covfunc_RBF(gpr->theta[0], dim, gpr->x[i], gpr->x[j]); 
          Grad1(i,j) = gpr->theta[0] * Covfunc_RBF_derivative(gpr->theta[1], dim, gpr->x[i], gpr->x[j]);

          if(i == j) { Grad2(i,j) = gpr->theta[2]; }
          else { Grad2(i,j) = 0; }
        }
      }

      /* Make Kernel matrix  for current parameters */
      GaussianProcessRegressor_MakeKernelMatrix(gpr);
      K = Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(&gpr->K[0][0], n, n); 
      y = Map<VectorXd>(gpr->y, n);

      
      /* calculate gradient */
      updates[0] = -0.5*((K.inverse())*Grad0).trace() + 0.5*((K.inverse()*y).transpose())*Grad0*(K.inverse()*y);
      updates[1] = -0.5*((K.inverse())*Grad1).trace() + 0.5*((K.inverse()*y).transpose())*Grad1*(K.inverse()*y);
      updates[2] = -0.5*((K.inverse())*Grad2).trace() + 0.5*((K.inverse()*y).transpose())*Grad2*(K.inverse()*y);
      

      /* output */
      printf("\ncycle %d : %lf\n", cycle, GaussianProcessRegressor_LogMarginalLikelihood(gpr));
      cout << K << endl;
      printf("theta: %lf, %lf, %lf\n", gpr->theta[0], gpr->theta[1], gpr->theta[2]);
      printf("theta(log-transformed): %lf, %lf, %lf\n", log(gpr->theta[0]), log(gpr->theta[1]), log(gpr->theta[2]));
      printf("update(gradient): %lf, %lf, %lf\n", updates[0], updates[1], updates[2]);


      /* update parameters */    
      for(int i=0;i<nparam;i++) { gpr->theta[i] += alpha * updates[i]; }


      /* converge check */
      converged = 1;
      for(int i=0;i<nparam;i++) {
        if(abs(updates[i]) > threshold) { converged *= 0;  }
      }

      if(converged == 1) {
        printf("converged\n");
        break;
      }
    }

  } else if(gpr->kerneltype == 1) {
    MatrixXd Grad0(n,n), Grad1(n,n), K;

    for(int cycle=0;cycle<maxcycle;cycle++) {

      /* evaluate gradient for Kernel matrix */
      double updates[nparam];
      for(int i=0;i<n;i++) {
        for(int j=0;j<n;j++) {
          Grad0(i,j) = Covfunc_RBF(gpr->theta[0], dim, gpr->x[i], gpr->x[j]); 
          Grad1(i,j) = gpr->theta[0] * Covfunc_RBF_derivative(gpr->theta[1], dim, gpr->x[i], gpr->x[j]);
        }
      }

      /* Make Kernel matrix  for current parameters */
      GaussianProcessRegressor_MakeKernelMatrix(gpr);
      K = Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(&gpr->K[0][0], n, n);
      y = Map<VectorXd>(gpr->y, n);
      
      /* calculate gradient */
      updates[0] = -((K.inverse())*Grad0).trace() + ((K.inverse()*y).transpose())*Grad0*(K.inverse()*y);
      updates[1] = -((K.inverse())*Grad1).trace() + ((K.inverse()*y).transpose())*Grad1*(K.inverse()*y);
      

      /* update parameters */    
      for(int i=0;i<nparam;i++) { gpr->theta[i] += alpha * updates[i]; }


      /* output */
      /*
      printf("cycle %d : %lf\n", cycle, GaussianProcessRegressor_LogMarginalLikelihood(gpr));
      printf("theta(log-transformed): %lf, %lf\n", log(gpr->theta[0]), log(gpr->theta[1]));
      printf("update(gradient): %lf, %lf\n", updates[0], updates[1]);
      */

      /* converge check */
      converged = 1;
      for(int i=0;i<nparam;i++) {
        if(abs(updates[i]) > threshold) { converged *= 0;  }
      }

      if(converged == 1) {
        printf("converged\n");
        break;
      }
      
    }
  }

}


void GaussianProcessRegressor_Malloc(GaussianProcessRegressor *gpr) {
  int n=gpr->ndat, m=gpr->dim, t=gpr->npred;

  gpr->theta = (double*)malloc(sizeof(double)*gpr->nparam);

  gpr->x = (double**)malloc(sizeof(double*)*n);
  gpr->x[0] = (double*)malloc(sizeof(double)*n*m); 
  for(int i=0;i<n;i++) { gpr->x[i] = gpr->x[0] + i*m; }

  gpr->y = (double*)malloc(sizeof(double)*n);
  // free(gpr->y);
  printf("gpr->y : %p, size : %d\n", gpr->y, n);

  gpr->K = (double**)malloc(sizeof(double*)*n);
  gpr->K[0] = (double*)malloc(sizeof(double*)*n*n);
  for(int i=0;i<n;i++) { gpr->K[i] = gpr->K[0] + i*n; }

  gpr->test = (double**)malloc(sizeof(double*)*t);
  gpr->test[0] = (double*)malloc(sizeof(double)*t*m);
  for(int i=0;i<t;i++) { gpr->test[i] = gpr->test[0] + i*m; }

  gpr->mu_pred = (double*)malloc(sizeof(double)*t);
  gpr->sigma_pred = (double*)malloc(sizeof(double)*t);
}


void GaussianProcessRegressor_StandardScaler(GaussianProcessRegressor *gpr, int key) {
  int n=gpr->ndat;

  if(key == 1) {
    /* Standarization for training data */
    double y_mean=0, y_var=0;

    for(int i=0;i<n;i++) { y_mean += gpr->y[i]; }
    y_mean /= n;
    
    for(int i=0;i<n;i++) { y_var += (gpr->y[i]-y_mean)*(gpr->y[i]-y_mean); }
    y_var = y_var/n;

    for(int i=0;i<n;i++) { gpr->y[i] = (gpr->y[i]-y_mean)/sqrt(y_var); }
    
    gpr->mu_train = y_mean;
    gpr->sigma_train = sqrt(y_var);
  } else if(key == -1) {
    /* Scale back the data to the original representation */

    for(int i=0;i<gpr->npred;i++) {
      gpr->mu_pred[i] = gpr->mu_pred[i] * gpr->sigma_train + gpr->mu_train;
      gpr->sigma_pred[i] = gpr->sigma_pred[i] * gpr->sigma_train;
    }
  }

}


void GaussianProcessRegressor_Free(GaussianProcessRegressor *gpr) {
  free(gpr->theta);
  free(gpr->x[0]);
  free(gpr->x);
  free(gpr->y);
  free(gpr->K);
  free(gpr->test);
  free(gpr->mu_pred);
  free(gpr->sigma_pred);
}


void GaussianProcessRegressor_predict(GaussianProcessRegressor *gpr) {
  MatrixXd K = Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(&gpr->K[0][0], gpr->ndat, gpr->ndat);
  VectorXd k(gpr->ndat), y = Map<VectorXd>(gpr->y, gpr->ndat);
  double k_self;

  if(gpr->kerneltype == 0) {
    
    for(int i=0;i<gpr->npred;i++) {
      for(int j=0;j<gpr->ndat;j++) { k[j] = gpr->theta[0] * Covfunc_RBF(gpr->theta[1], gpr->dim, gpr->test[i], gpr->x[j]) + gpr->theta[2]; }
      k_self = gpr->theta[0] * Covfunc_RBF(gpr->theta[1], gpr->dim, gpr->test[i], gpr->test[i]);
      gpr->mu_pred[i] = (k.transpose())*(K.inverse())*y;
      gpr->sigma_pred[i] = k_self - (k.transpose())*(K.inverse())*k;
    }
    
  } else if(gpr->kerneltype == 1) {

    for(int i=0;i<gpr->npred;i++) {
      for(int j=0;j<gpr->ndat;j++) { k[j] = gpr->theta[0] * Covfunc_RBF(gpr->theta[1], gpr->dim, gpr->test[i], gpr->x[j]); }
      k_self = gpr->theta[0] * Covfunc_RBF(gpr->theta[1], gpr->dim, gpr->test[i], gpr->test[i]);
      gpr->mu_pred[i] = (k.transpose())*(K.inverse())*y;
      gpr->sigma_pred[i] = k_self - (k.transpose())*(K.inverse())*k;
    }

  }

}


double **Loadcsv(int *nrow, int *ncol, char *file) {
  int _nrow=0, _ncol=0;
  char line[256];
  double **mat;
  FILE *fp;

  fp = fopen(file,"r");
  while( fgets(line,256,fp) ) {
    _ncol = 1; // count the number of column

    for(int i=0;i<256;i++) {
      if(line[i]==',') { _ncol++; }
    }

    _nrow++; // count the number of row
  }
  fclose(fp);

  *nrow = _nrow;
  *ncol = _ncol;

  mat = (double**)malloc(sizeof(double*)*_nrow);
  mat[0] = (double*)malloc(sizeof(double)*_nrow*_ncol);
  for(int i=1;i<_nrow;i++) { mat[i] = mat[0] + i*_ncol; }

  fp = fopen(file,"r");

  for(int i=0;i<_nrow;i++) {
    fgets(line,sizeof(line),fp);

    int cnt=0;
    sscanf(line, "%lf", &mat[i][cnt++]);
    for(int pos=0;pos<256;pos++) {
      if(line[pos] == ',') { sscanf(line+pos+1, "%lf", &mat[i][cnt++]); }
    }
  }

  fclose(fp);

  return mat;
}


int GaussianProcessRegression(char *input) {
  GaussianProcessRegressor *gpr;
  FILE *fp;
  char traincsv[256], testcsv[256], line[256];
  double **train, **test;
  int nrow_train, ncol_train, nrow_test, ncol_test;
  
  gpr = (GaussianProcessRegressor*)malloc(sizeof(GaussianProcessRegressor));
  
  // load input file
  fp = fopen(input,"r");
  while(fgets(line,256,fp)) {
    const char *pt = strstr(line,"=");

    if( strstr(line,"kernel") ) { sscanf(pt+1,"%d", &gpr->kerneltype); }
    else if( strstr(line,"train") ) { sscanf(pt+1,"%s", traincsv); }
    else if( strstr(line,"test") ) { sscanf(pt+1,"%s", testcsv); } 
  }

  train = Loadcsv(&nrow_train, &ncol_train, traincsv);
  test = Loadcsv(&nrow_test, &ncol_test, testcsv);

  if(ncol_test != ncol_train-1) {
    printf("%d,%d\n", ncol_test, ncol_train);
    printf("Dimension of train and test is not match\n");
    return -1;
  }

  gpr->dim = ncol_train-1;
  gpr->ndat = nrow_train; 
  gpr->npred = nrow_test;

  if(gpr->kerneltype == 0) { gpr->nparam = 3; }
  else if(gpr->kerneltype == 1) { gpr->nparam = 2; }

  GaussianProcessRegressor_Malloc(gpr);
  
  for(int i=0;i<nrow_train;i++) {
    for(int j=0;j<ncol_train-1;j++) { gpr->x[i][j] = train[i][j]; }
    gpr->y[i] = train[i][ncol_train-1];
  }

  for(int i=0;i<nrow_test;i++) {
    for(int j=0;j<ncol_test;j++) { gpr->test[i][j] = test[i][j]; }
  }

  free(train[0]);
  free(train);
  free(test[0]);
  free(test);

  /*
  for(int i=0;i<gpr->ndat;i++) {
    for(int j=0;j<gpr->dim;j++) {
      printf("%lf,", gpr->test[i][j]);
    }
    printf("\n");
    // printf("%lf\n", gpr->y[i]);
  }
  */

  GaussianProcessRegressor_StandardScaler(gpr,1);

  /*
  for(int i=0;i<gpr->ndat;i++) {
    for(int j=0;j<gpr->dim;j++) {
      printf("%lf,", gpr->x[i][j]);
    }
    printf("%lf\n", gpr->y[i]);
  }
  */

  if(gpr->kerneltype == 0) {
    /* set parameters for kernel */
    gpr->nparam = 3;
    gpr->theta[0] = 1;
    gpr->theta[1] = 1;
    gpr->theta[2] = 1e-5;
    gpr->alpha = 0;


    /* Optimize hyper-parameters */
    GaussianProcessRegressor_fit(gpr);

    GaussianProcessRegressor_predict(gpr);

    printf("Optimum hyperparameters : %lf, %lf\n", log(gpr->theta[0]), log(gpr->theta[1])); // Optimum parameters

    for(int i=0;i<gpr->ndat;i++) { printf("%lf, %lf\n", gpr->x[i][0], gpr->y[i]); }
    for(int i=0;i<gpr->npred;i++) { printf("%lf, %lf, %lf\n", gpr->test[i][0], gpr->mu_pred[i], gpr->sigma_pred[i]); }
  } else if(gpr->kerneltype == 1) {

    /* set parameters for kernel */
    gpr->nparam = 2;
    gpr->theta[0] = 1;
    gpr->theta[1] = 1;
    gpr->alpha = 0.1;


    /* Optimize hyper-parameters */
    GaussianProcessRegressor_fit(gpr);


    /* Predict mu and sigma for test data */
    GaussianProcessRegressor_predict(gpr);
    

    /* output */
    /*
    fp = fopen("lml.csv", "w");
    for(double i=-5;i<=5;i+=0.1) {
      for(double j=-5;j<=5;j+=0.1) {
        GaussianProcessRegressor gpr0 = *gpr;
        gpr0.theta[0] = exp(j); // xaxis
        gpr0.theta[1] = exp(i); // yaxis
        GaussianProcessRegressor_MakeKernelMatrix(&gpr0);
        fprintf(fp,"%lf,%lf,%lf\n", j, i, GaussianProcessRegressor_LogMarginalLikelihood(&gpr0));
      } 
    }
    fclose(fp);
    */

    printf("Optimum hyperparameters : %lf, %lf\n", log(gpr->theta[0]), log(gpr->theta[1])); // Optimum parameters

    for(int i=0;i<gpr->ndat;i++) { printf("%lf, %lf\n", gpr->x[i][0], gpr->y[i]); }
    for(int i=0;i<gpr->npred;i++) { printf("%lf, %lf, %lf\n", gpr->test[i][0], gpr->mu_pred[i], gpr->sigma_pred[i]); }
  }
 
  GaussianProcessRegressor_Free(gpr);
  free(gpr);

  return 0;
}


int main(int argc, char *argv[]) {
  GaussianProcessRegression(argv[1]);
  return 0;
}