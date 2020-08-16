# include <bits/stdc++.h>
# include <Eigen/Core>
// # include <Eigen/Dense>
// # include <Eigen/LU>

using namespace std;
using namespace Eigen;
int main (void) {
  /*
  int mat[10][10];
  for(int i=0;i<10;i++) {
    for(int j=0;j<10;j++) {
      mat[i][j] = i+j;
    }
  }

  MatrixXi D = Map<Matrix<int,Dynamic,Dynamic,RowMajor>>(&mat[0][0],10,10);
  */
  
  double vec[10];
  for(int i=0;i<10;i++) { vec[i] = (double)i; }
  VectorXd dvec = Map<VectorXd>(&vec[0],10);

  return 0;
}
