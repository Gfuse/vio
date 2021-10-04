/*
 * math_utils.cpp
 *
 *  Created on: Jul 20, 2012
 *      Author: cforster
 */

#include <vio/math_utils.h>
#include <ros/ros.h>
#include <Eigen/SVD>

namespace vk {

using namespace Eigen;
/*
 * R and T are the rotation and translation of the current frames with respect to the reference frame, T from ref to current
 * */
///TODO we need to add the restriction of the movement here
Vector3d
triangulateFeatureNonLin(const Matrix3d& R,  const Vector3d& t,
                         const Vector3d& feature1, const Vector3d& feature2 )
{
/*  Vector3d R_F2 = R * feature2;//T*A=R*A+t
  Vector2d b;
  *//*
   * b=cross(t,f1)
   *   corss(t,Rf2)
   * *//*
  b[0] = t.dot(feature1);
  b[1] = t.dot(R_F2);//cross(t,R*f2)
  Matrix2d A;
  *//**
   * A<< cross(f1,f1), -z
   *     z           , -corss(R*f2,R*f2)
   *
   * det(A)= -corss(R*f2,R*f2)*cross(f1,f1) + pow(z,2)
   *//*
  A(0,0) = feature1.dot(feature1);
  A(1,0) = feature1.dot(R_F2);//z
  A(0,1) = -A(1,0);
  A(1,1) = -R_F2.dot(R_F2);
  *//*
   * A_inv<< cross(f1,f1)/det(A) , z/det(A)
   *         -z/det(A)           , -corss(R*f2,R*f2)/det(A)
   *
   * lambda[0]= cross(t,f1)*cross(f1,f1)/det(A) + corss(t,Rf2)*z/det(A)
   * lambda[1]= -cross(t,f1)*z/det(A) -   corss(t,Rf2) * corss(R*f2,R*f2)/det(A)
   * *//*
  Vector2d lambda = A.inverse() * b;
  Vector3d xm = lambda[0] * feature1;
  Vector3d xn = t + lambda[1] * R_F2;
  return ( xm + xn )/2;*/
/*        Vector3d R_F2 = R * feature2;//T*A=R*A+t
        Vector3d z = feature1.cross(R_F2);
        Vector3d b0 = t.cross(feature1);
        Vector3d b1 = t.cross(R_F2);

        Vector3d xm = ((z.dot(b1)/pow(z.norm(), 2))*(feature1));
        Vector3d xn = t + ((z.dot(b0)/pow(z.norm(), 2))*R_F2);
        return ( xm + xn )/2;*/
        Vector3d m0_hat = R * feature2;//T*A=R*A+t
        Vector3d m1_hat = feature1;
        Eigen::Matrix<double,3,2> x;
        x<<m0_hat,m1_hat;
        Eigen::Matrix<double,2,3> svd_in = x.transpose() * (Eigen::Matrix3d::Identity() - (t/t.norm())*(t/t.norm()).transpose());
        Eigen::JacobiSVD<MatrixXd> svd( svd_in, ComputeFullU | ComputeFullV);
        Vector3d n_p_hat = svd.matrixV().col(1);
        Vector3d m0_p = m0_hat - (m0_hat.dot(n_p_hat)) * n_p_hat;
        Vector3d m1_p = m1_hat - (m1_hat.dot(n_p_hat)) * n_p_hat;
        Vector3d R_F2_p = m0_p;
        Vector3d f1_p = m1_p;
        Vector3d z = f1_p.cross(R_F2_p);
        Vector3d b0 = t.cross(f1_p);
        Vector3d b1 = t.cross(R_F2_p);
        double lamda0 = z.dot(b0)/pow(z.norm(), 2);
        double lamda1 = z.dot(b1)/pow(z.norm(), 2);
        Vector3d xm =  lamda1 * f1_p;
        Vector3d xn = t + lamda0 * R_F2_p;
        return ( xm + xn )/2;
}

bool
depthFromTriangulationExact(
    const Matrix3d& R_r_c,
    const Vector3d& t_r_c,
    const Vector3d& f_r,
    const Vector3d& f_c,
    double& depth_in_r,
    double& depth_in_c)
{
  // bearing vectors (f_r, f_c) do not need to be unit length
  const Vector3d f_c_in_r(R_r_c*f_c);
  const double a = f_c_in_r.dot(f_r) / t_r_c.dot(f_r);
  const double b = f_c_in_r.dot(t_r_c);
  const double denom = (a*b - f_c_in_r.dot(f_c_in_r));

  if(abs(denom) < 0.000001)
    return false;

  depth_in_c = (b-a*t_r_c.dot(t_r_c)) / denom;
  depth_in_r = (t_r_c + f_c_in_r*depth_in_c).norm();
  return true;
}

double
reprojError(const Vector3d& f1,
            const Vector3d& f2,
            double error_multiplier2)
{
  Vector2d e = project2d(f1) - project2d(f2);
  return error_multiplier2 * e.norm();
}

double
computeInliers(const vector<Vector3d>& features2, // c2
               const vector<Vector3d>& features1, // c1
               const Matrix3d& R,                 // R_c1_to_c2
               const Vector3d& t,                 // T_c1_to_c2
               const double reproj_thresh,
               double error_multiplier2,
               vector<Vector3d>& xyz_vec,         // 3d point: with respect to the reference frame
               vector<int>& inliers,
               vector<int>& outliers)
{
  inliers.clear();
  outliers.clear();
  xyz_vec.clear();
  double tot_error = 0;
  //triangulate all features and compute reprojection errors and inliers
  for(size_t j=0; j<features1.size(); ++j)
  {
    if(t.z()>0.0){
        xyz_vec.push_back(triangulateFeatureNonLin(R, t, features1[j]/*reference*/, features2[j]/*current*/ ));
        double e1 = reprojError(features1[j], xyz_vec.back(), error_multiplier2);
        double e2 = reprojError(features2[j], SE3(R,t).inverse()*xyz_vec.back(), error_multiplier2);
        if(e1 > reproj_thresh || e2 > reproj_thresh)
            outliers.push_back(j);
        else
        {
            inliers.push_back(j);
            tot_error += e1+e2;
        }
    }
  }
  return tot_error;
}

void
computeInliersOneView(const vector<Vector3d> & feature_sphere_vec,
                      const vector<Vector3d> & xyz_vec,
                      const Matrix3d &R,
                      const Vector3d &t,
                      const double reproj_thresh,
                      const double error_multiplier2,
                      vector<int>& inliers,
                      vector<int>& outliers)
{
  inliers.clear(); //inliers.reserve(xyz_vec.size());
  outliers.clear(); //outliers.reserve(xyz_vec.size());
  for(size_t j = 0; j < xyz_vec.size(); j++ )
  {
    double e = reprojError(feature_sphere_vec[j],
                           R.transpose() * ( xyz_vec[j] - t ),
                           error_multiplier2);
    if(e < reproj_thresh)
      inliers.push_back(j);
    else
      outliers.push_back(j);
  }
}

Vector3d
dcm2rpy(const Matrix3d &R)
{
  Vector3d rpy;
  rpy[1] = atan2( -R(2,0), sqrt( pow( R(0,0), 2 ) + pow( R(1,0), 2 ) ) );
  if( fabs( rpy[1] - M_PI/2 ) < 0.00001 )
  {
    rpy[2] = 0;
    rpy[0] = -atan2( R(0,1), R(1,1) );
  }
  else
  {
    if( fabs( rpy[1] + M_PI/2 ) < 0.00001 )
    {
      rpy[2] = 0;
      rpy[0] = -atan2( R(0,1), R(1,1) );
    }
    else
    {
      rpy[2] = atan2( R(1,0)/cos(rpy[1]), R(0,0)/cos(rpy[1]) );
      rpy[0] = atan2( R(2,1)/cos(rpy[1]), R(2,2)/cos(rpy[1]) );
    }
  }
  return rpy;
}

Matrix3d
rpy2dcm(const Vector3d &rpy)
{
  Matrix3d R1;
  R1(0,0) = 1.0; R1(0,1) = 0.0; R1(0,2) = 0.0;
  R1(1,0) = 0.0; R1(1,1) = cos(rpy[0]); R1(1,2) = -sin(rpy[0]);
  R1(2,0) = 0.0; R1(2,1) = -R1(1,2); R1(2,2) = R1(1,1);

  Matrix3d R2;
  R2(0,0) = cos(rpy[1]); R2(0,1) = 0.0; R2(0,2) = sin(rpy[1]);
  R2(1,0) = 0.0; R2(1,1) = 1.0; R2(1,2) = 0.0;
  R2(2,0) = -R2(0,2); R2(2,1) = 0.0; R2(2,2) = R2(0,0);

  Matrix3d R3;
  R3(0,0) = cos(rpy[2]); R3(0,1) = -sin(rpy[2]); R3(0,2) = 0.0;
  R3(1,0) = -R3(0,1); R3(1,1) = R3(0,0); R3(1,2) = 0.0;
  R3(2,0) = 0.0; R3(2,1) = 0.0; R3(2,2) = 1.0;

  return R3 * R2 * R1;
}

Quaterniond
angax2quat(const Vector3d& n, const double& angle)
{
  // n must be normalized!
  double s(sin(angle/2));
  return Quaterniond( cos(angle/2), n[0]*s, n[1]*s, n[2]*s );
}


Matrix3d
angax2dcm(const Vector3d& n, const double& angle)
{
  // n must be normalized
  Matrix3d sqewn(sqew(n));
  return Matrix3d(Matrix3d::Identity() + sqewn*sin(angle) + sqewn*sqewn*(1-cos(angle)));
}

double
sampsonusError(const Vector2d &v2Dash, const Matrix3d& Essential, const Vector2d& v2)
{
  Vector3d v3Dash = unproject2d(v2Dash);
  Vector3d v3 = unproject2d(v2);

  double dError = v3Dash.transpose() * Essential * v3;

  Vector3d fv3 = Essential * v3;
  Vector3d fTv3Dash = Essential.transpose() * v3Dash;

  Vector2d fv3Slice = fv3.head<2>();
  Vector2d fTv3DashSlice = fTv3Dash.head<2>();

  return (dError * dError / (fv3Slice.dot(fv3Slice) + fTv3DashSlice.dot(fTv3DashSlice)));
}

} // end namespace vk
