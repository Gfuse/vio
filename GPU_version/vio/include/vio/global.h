// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SVO_GLOBAL_H_
#define SVO_GLOBAL_H_

#include <list>
#include <vector>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se2.h>
#include <sophus/se3.h>
#include <boost/shared_ptr.hpp>
#include<Eigen/StdVector>
#ifndef RPG_SVO_VIKIT_IS_VECTOR_SPECIALIZED //Guard for rpg_vikit
#define RPG_SVO_VIKIT_IS_VECTOR_SPECIALIZED
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector2d)
#endif
#include <ros/console.h>
#define SVO_DEBUG_STREAM(x) ROS_DEBUG_STREAM(x)
#define SVO_INFO_STREAM(x) ROS_INFO_STREAM(x)
#define SVO_WARN_STREAM(x) ROS_WARN_STREAM(x)
#define SVO_WARN_STREAM_THROTTLE(rate, x) ROS_WARN_STREAM_THROTTLE(rate, x)
#define SVO_ERROR_STREAM(x) ROS_ERROR_STREAM(x)


namespace vio
{
  using namespace Eigen;
  using namespace Sophus;

  const double EPS = 0.0000000001;
  const double PI = 3.14159265;
  class SE2_5: public SE2{
  public:
        SE2_5(SE2&& se2):T2_(se2){
        }
        SE2_5(SE2& se2):T2_(se2){
        }
        SE2_5(SE3&& se3){
            Quaterniond q=se3.unit_quaternion().normalized();
            double pitch=asin(2.0*(q.w()*q.y()-q.z()*q.x()));
            if (std::abs(pitch) >= 1)
                pitch = std::copysign(M_PI / 2.0, pitch); // use 90 degrees if out of range
            else
                pitch = std::asin(pitch);
            Eigen::Matrix<double, 2,2> R;
            R<<cos(pitch),sin(pitch),
               -sin(pitch),cos(pitch);
            T2_ = SE2(R,Vector2d(se3.translation().x(),se3.translation().z()));
        }
        SE2_5(SE3& se3){
            Quaterniond q=se3.unit_quaternion().normalized();
            double pitch=asin(2.0*(q.w()*q.y()-q.z()*q.x()));
            if (std::abs(pitch) >= 1)
                pitch = std::copysign(M_PI / 2.0, pitch); // use 90 degrees if out of range
            else
                pitch = std::asin(pitch);
             Eigen::Matrix<double, 2,2> R;
             R<<cos(pitch),sin(pitch),
                  -sin(pitch),cos(pitch);
            T2_=SE2(R,Vector2d(se3.translation().x(),se3.translation().z()));
        }
        SE2_5(double y,double z,double roll){
            Eigen::Matrix<double, 2,2> R;
            R<<cos(roll),sin(roll),
               -sin(roll),cos(roll);
            T2_=SE2(R,Vector2d(y,z));
        };
        SE2 se2() const{
            return T2_;
        }
        SE2 inverse() const{
            double roll=atan2(T2_.so2().unit_complex().imag(),T2_.so2().unit_complex().real());
            return SE2(roll+M_PI,-1.0*T2_.translation());
        }
        // Rotation around x
        double pitch()const{
            return atan2(T2_.rotation_matrix()(0,1),T2_.rotation_matrix()(0,0));
        }
        SE3 se3() const{
            //Todo add 15 roll orientation
            Eigen::Matrix<double,3,3> R;
            R<<T2_.rotation_matrix()(0,0),0.0,T2_.rotation_matrix()(0,1),
                     0.0,1.0,0.0,
                    T2_.rotation_matrix()(1,0),0.0,T2_.rotation_matrix()(1,1);
            return SE3(R,Vector3d(T2_.translation()(0),1e-19,T2_.translation()(1)));
        }
        ~SE2_5(){}

  private:
      SE2 T2_;
  };

  class Frame;
  typedef boost::shared_ptr<Frame> FramePtr;
} // namespace vio

#endif // SVO_GLOBAL_H_
