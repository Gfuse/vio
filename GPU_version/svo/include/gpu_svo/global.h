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


namespace svo
{
  using namespace Eigen;
  using namespace Sophus;

  const double EPS = 0.0000000001;
  const double PI = 3.14159265;
  class SE2_5{
  public:
        double X_,Z_,tetha_,cos_theta_,sin_theta_;
        SE3* T;
        SE2_5(double x,double z,double tetha):X_(x),Z_(z),tetha_(tetha){
            cos_theta_=cos(tetha_);
            sin_theta_=sin(tetha_);
            Eigen::Matrix<double, 3,3> R;
            R<<cos_theta_,0.0,sin_theta_,
               0.0,1.0,0.0,
               -sin_theta_,0.0,cos_theta_;
            T=new SE3(R,Vector3d(X_,0.0,Z_));
        };
        void set(){
            X_=T->translation().x();
            Z_=T->translation().z();
            tetha_=asin(2*(T->unit_quaternion().w()*T->unit_quaternion().y()-
                           T->unit_quaternion().z()*T->unit_quaternion().x()));
            cos_theta_=cos(tetha_);
            sin_theta_=sin(tetha_);
        }
        ~SE2_5(){}
  };

  class Frame;
  typedef boost::shared_ptr<Frame> FramePtr;
} // namespace svo

#endif // SVO_GLOBAL_H_
