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

#ifndef SVO_INITIALIZATION_H
#define SVO_INITIALIZATION_H

#include <vio/global.h>
#include <vio/cl_class.h>
#include <vio/ukf.h>
#include <vio/homography.h>

namespace vio {

class FrameHandlerMono;

/// Bootstrapping the map from the first two views.
namespace initialization {

enum InitResult { FAILURE, NO_KEYFRAME, SUCCESS };

/// Tracks features using Lucas-Kanade tracker and then estimates a homography.
class KltHomographyInit {
  friend class vio::FrameHandlerMono;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FramePtr frame_ref_;
  KltHomographyInit(opencl* gpu_fast_,UKF* ukf,FILE* log=nullptr):gpu_fast_(gpu_fast_),T_cur_from_ref_(0.0,0.0,0.0),ukf_(ukf),log_(log) {};
  ~KltHomographyInit() {};
  InitResult addFirstFrame(FramePtr frame_ref);
  InitResult addSecondFrame(FramePtr frame_ref);
  void reset();

protected:
  vector<cv::Point2f> px_ref_;      //!< keypoints to be tracked in reference frame.
  vector<cv::Point2f> px_cur_;      //!< tracked keypoints in current frame.
  vector<Vector3d> f_ref_;          //!< bearing vectors corresponding to the keypoints in the reference image.
  vector<Vector3d> f_cur_;          //!< bearing vectors corresponding to the keypoints in the current image.
  vector<double> disparities_;      //!< disparity between first and second frame.
  vector<int> inliers_;             //!< inliers after the geometric check (e.g., Homography).
  vector<Vector3d> xyz_in_cur_;     //!< 3D points computed during the geometric check.
  SE2_5 T_cur_from_ref_;              //!< computed transformation between the first two frames.
  opencl* gpu_fast_;
  FILE* log_=nullptr;
  UKF* ukf_= nullptr;

  void computeHomography(
            const vector<Vector3d>& f_ref,
            const vector<Vector3d>& f_cur,
            double focal_length,
            double reprojection_threshold,
            vector<int>& inliers,
            vector<Vector3d>& xyz_in_cur,
            SE2_5& T_cur_from_ref)
    {
        vector<Vector2d > uv_ref(f_ref.size());
        vector<Vector2d > uv_cur(f_cur.size());
        for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
        {
            uv_ref[i] = vk::project2d(f_ref[i]);
            uv_cur[i] = vk::project2d(f_cur[i]);
        }
        vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
        Homography.computeSE3fromMatches();
        vector<int> outliers;
        assert(ukf_!= nullptr);
        auto euler =Homography.T_c2_from_c1.unit_quaternion().normalized().toRotationMatrix().eulerAngles(0,1,2);
        auto check=ukf_->get_location();
#if VIO_DEBUG
      fprintf(log_,"[%s] homography: x=%f y=%f z=%f roll=%f pitch=%f yaw=%f \t ekf: x=%f z=%f pitch=%f\n",
              vio::time_in_HH_MM_SS_MMM().c_str(),
              Homography.T_c2_from_c1.translation().x(),Homography.T_c2_from_c1.translation().y(),Homography.T_c2_from_c1.translation().z(),
              euler(0),euler(1),euler(2)
              ,check.second.se2().translation().x(),check.second.se2().translation().y(),check.second.pitch());
#endif
        if((check.second.se3().translation()-Homography.T_c2_from_c1.translation()).norm()>0.1)return;
        auto res=ukf_->UpdateSvo(Homography.T_c2_from_c1.translation().x(),Homography.T_c2_from_c1.translation().z(),euler(1));
        vk::computeInliers(f_cur, f_ref,
                           res.second.se3().rotation_matrix(), res.second.se3().translation(),
                               reprojection_threshold, focal_length,
                               xyz_in_cur, inliers, outliers);
#if VIO_DEBUG
      fprintf(log_,"[%s] homography after fuse: x=%f y=%f z=%f pitch=%f\t number of outlier: %d\n",
              vio::time_in_HH_MM_SS_MMM().c_str(),
              res.second.se3().translation().x(),res.second.se3().translation().y(),res.second.se3().translation().z(),res.second.pitch(),outliers.size());
#endif
        T_cur_from_ref=res.second;
    }
};

/// Detect Fast corners in the image.
void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec,
    opencl* gpu_fast);
void trackKlt(
            FramePtr frame_ref,
            FramePtr frame_cur,
            vector<cv::Point2f>& px_ref,
            vector<cv::Point2f>& px_cur,
            vector<Vector3d>& f_ref,
            vector<Vector3d>& f_cur,
            vector<double>& disparities);
} // namespace initialization
} // namespace vio

#endif // SVO_INITIALIZATION_H
