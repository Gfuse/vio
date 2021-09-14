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

#include <vio/config.h>
#include <vio/frame.h>
#include <vio/point.h>
#include <vio/feature.h>
#include <vio/initialization.h>
#include <vio/feature_detection.h>
#include <vio/math_utils.h>

namespace vio {
namespace initialization {

InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  reset();
  detectFeatures(frame_ref, px_ref_, f_ref_,gpu_fast_);
  if(px_ref_.size() < 100)
  {
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }
#if VIO_DEBUG
    fprintf(log_,"[%s] Init: frame zero: %f, %f %f\n",vio::time_in_HH_MM_SS_MMM().c_str(),
            frame_ref->T_f_w_.se2().translation().x(),
            frame_ref->T_f_w_.se2().translation().y(),
            frame_ref->T_f_w_.pitch());
#endif
  frame_ref_ = frame_ref;
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
  return SUCCESS;
}

InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  if(disparities_.size() < 1){
      ukf_->UpdateSvo(0.0,0.0,0.0);
      return NO_KEYFRAME;
  }
  double disparity = vk::getMedian(disparities_);
    if(disparity < Config::initMinDisparity()){
        auto result=ukf_->get_location();
        if(fabs(result.second.se2().translation().x())>0.2 || fabs(result.second.pitch())>0.0698132)ukf_->UpdateSvo(0.0,result.second.se2().translation().y(),0.0);
#if VIO_DEBUG
        fprintf(log_,"[%s] Init: px average disparity is:%f ,While minimum is: %f  KLT tracked : %d\n",vio::time_in_HH_MM_SS_MMM().c_str(),
                disparity,
                Config::initMinDisparity(),
                disparities_.size());
#endif
        return NO_KEYFRAME;
    }
  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);
  if(inliers_.size() < Config::initMinInliers()){
#if VIO_DEBUG
      fprintf(log_,"[%s] Init: Homography RANSAC (inlier) is:%d ,While %d inliers minimum required.  px average disparity is:%f ,While minimum is: %f  KLT tracked: %d\n",
              vio::time_in_HH_MM_SS_MMM().c_str(),
              inliers_.size(),
              Config::initMinInliers(),
              disparity,
              Config::initMinDisparity(),
              disparities_.size());
#endif
      return NO_KEYFRAME;
  }
    frame_cur->T_f_w_ = T_cur_from_ref_;
  // For each inlier create 3D point and add feature in both frames
    for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
    {
        Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
        Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
        if(frame_cur->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10))
        {
            Vector3d pos = frame_cur->getSE3Inv() * (xyz_in_cur_[*it]/* *scale */);
            Point* new_point = new Point(pos);

            Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
            frame_cur->addFeature(ftr_cur);
            new_point->addFrameRef(ftr_cur);

            Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
            frame_ref_->addFeature(ftr_ref);
            new_point->addFrameRef(ftr_ref);
        }
    }
#if VIO_DEBUG
    fprintf(log_,"[%s] Init finished: Homography RANSAC (inlier) is:%d ,While %d inliers minimum required.  px average disparity is:%f ,While minimum is: %f  KLT tracked: %d\n",
            vio::time_in_HH_MM_SS_MMM().c_str(),
            inliers_.size(),
            Config::initMinInliers(),
            disparity,
            Config::initMinDisparity(),
            disparities_.size());
#endif
  return SUCCESS;
}

void KltHomographyInit::reset()
{
  px_cur_.clear();
  frame_ref_.reset();
}

void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec,
    opencl* gpu_fast)
{
  Features new_features;
  feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), gpu_fast,Config::nPyrLevels());
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);
  // now for all maximum corners, initialize a new seed
  px_vec.clear();
  f_vec.clear();
  for(auto&& ftr:new_features){
      px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
      f_vec.push_back(ftr->f);
      delete ftr;
  }
}
void trackKlt(
            FramePtr frame_ref,
            FramePtr frame_cur,
            vector<cv::Point2f>& px_ref,
            vector<cv::Point2f>& px_cur,
            vector<Vector3d>& f_ref,
            vector<Vector3d>& f_cur,
            vector<double>& disparities)
    {
        disparities.clear();
        const double klt_win_size = 30.0;//30.0
        const int klt_max_iter = 30;//30
        const double klt_eps = 0.001;
        vector<uchar> status;
        vector<float> error;
        vector<float> min_eig_vec;
        cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
        if(frame_ref->img_pyr_[0].empty() || frame_cur->img_pyr_[0].empty())return;
        cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                                 px_ref, px_cur,
                                 status, error,
                                 cv::Size2i(klt_win_size, klt_win_size),
                                 4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);
        vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
        vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
        vector<Vector3d>::iterator f_ref_it = f_ref.begin();
        f_cur.clear(); //f_cur.reserve(px_cur.size());
        //disparities.reserve(px_cur.size());
        for(size_t i=0; px_ref_it != px_ref.end(); ++i)
        {
            if(!status[i])
            {
                px_ref_it = px_ref.erase(px_ref_it);
                px_cur_it = px_cur.erase(px_cur_it);
                f_ref_it = f_ref.erase(f_ref_it);
                continue;
            }
            f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
            disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
            ++px_ref_it;
            ++px_cur_it;
            ++f_ref_it;
        }
    };
} // namespace initialization
} // namespace vio
