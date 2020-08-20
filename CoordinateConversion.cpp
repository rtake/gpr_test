# include <bits/stdc++.h>
# include <Eigen/Core>
# include <Eigen/LU>
# include <Eigen/Dense>
# include <Eigen/Geometry>


using namespace std;
using namespace Eigen;

void DistancetoCartesian(double *distance, int natom, double (*cartesian)[3]) {
  int n=natom;
  double theta_0, theta_1, theta_2;
  MatrixXd A(3,3);
  VectorXd b(3), x(3);

  theta_0 = acos((distance[0]*distance[0]+distance[1]*distance[1]-distance[n-1]*distance[n-1])/(2*distance[0]*distance[1]));
  theta_1 = acos((distance[0]*distance[0]+distance[2]*distance[2]-distance[2+n-2]*distance[2+n-2])/(2*distance[0]*distance[2]));
  theta_2 = acos((distance[1]*distance[1]+distance[2]*distance[2]-distance[2+n-2+n-3]*distance[2+n-2+n-3])/(2*distance[1]*distance[2]));  

  b(0) = (distance[0]*distance[0] + distance[2]*distance[2] - distance[2+(n-2)]*distance[2+(n-2)])/2;
  b(1) = (distance[1]*distance[1] + distance[2]*distance[2] - distance[2+(n-2)+(n-3)]*distance[2+(n-2)+(n-3)])/2;

  cartesian[0][0] = 0; 
  cartesian[0][1] = 0;  
  cartesian[0][2] = 0;
  cartesian[1][0] = distance[0]; 
  cartesian[1][1] = 0;  
  cartesian[1][2] = 0;
  cartesian[2][0] = distance[1]*cos(theta_0);  
  cartesian[2][1] = distance[1]*sin(theta_0);  
  cartesian[2][2] = 0;
  cartesian[3][0] = b(0)/cartesian[1][0]; 
  cartesian[3][1] = (1/cartesian[2][1])*(b(1)-b(0)*cartesian[2][0]/cartesian[1][0]);
  cartesian[3][2] = sqrt(distance[2]*distance[2]-cartesian[3][0]*cartesian[3][0]-cartesian[3][1]*cartesian[3][1]);

  for(int i=0;i<3;i++) {
    for(int j=0;j<3;j++) {
      A(i,j) = cartesian[i+1][j];
    }
  }

  for(int i=4;i<n;i++) {
    b(0) = (distance[i-1]*distance[i-1] + distance[0]*distance[0] - distance[i-1+(n-2)]*distance[i-1+(n-2)])/2;
    b(1) = (distance[i-1]*distance[i-1] + distance[1]*distance[1] - distance[i-1+(n-2)+(n-3)]*distance[i-1+(n-2)+(n-3)])/2;
    b(2) = (distance[i-1]*distance[i-1] + distance[2]*distance[2] - distance[i-1+(n-2)+(n-3)+(n-4)]*distance[i-1+(n-2)+(n-3)+(n-4)])/2;
    x = A.colPivHouseholderQr().solve(b);
      
    for(int j=0;j<3;j++) { cartesian[i][j] = x(j); }

    cout << A << endl;
    cout << x << endl;
    cout << b << endl;
    cout << A * x << endl;
  }

}

int main(void) {
  int natom=5;
  double distance[natom*(natom-1)/2], cartesian[natom][3];

  distance[0] = 1.09188;
  distance[1] = 1.09188;
  distance[2] = 1.09188;
  distance[3] = 1.09188;

  distance[4] = 1.78303;
  distance[5] = 1.78303;
  distance[6] = 1.78303;

  distance[7] = 1.78303;
  distance[8] = 1.78303;

  distance[9] = 1.78303;

  DistancetoCartesian(distance, natom, cartesian);

  for(int i=0;i<natom;i++) {
    printf("X\t");
    for(int j=0;j<3;j++) { printf("%lf\t", cartesian[i][j]); }
    printf("\n");
  }

  return 0;
}