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

#include <gpu_svo/config.h>
#include <gpu_svo/frame.h>
#include <gpu_svo/point.h>
#include <gpu_svo/feature.h>
#include <gpu_svo/initialization.h>
#include <gpu_svo/feature_detection.h>
#include <gpu_svo/math_utils.h>
#include <gpu_svo/homography.h>

namespace svo {
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
  frame_ref_ = frame_ref;
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
  return SUCCESS;
}

InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  if(disparities_.size() < Config::initMinTracked())
        return NO_KEYFRAME;
  double disparity = vk::getMedian(disparities_);
  if(disparity < Config::initMinDisparity()){
      SVO_INFO_STREAM("Init: px average disparity is: "<<disparity<<" While: "<<Config::initMinDisparity()<<" minimum required."
      <<"  KLT tracked : "<<disparities_.size());
      return NO_KEYFRAME;
  }
  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);

  if(inliers_.size() < Config::initMinInliers())
  {
    SVO_INFO_STREAM("Init Homography RANSAC (inlier) is: "<<inliers_.size()<<" While: "<<Config::initMinInliers()<<" inliers minimum required.\n"
       <<"Init: px average disparity is: "<<disparity<<" While: "<<Config::initMinDisparity()<<" minimum required.");
      return NO_KEYFRAME;
  }

  // Rescale the map such that the mean scene depth is equal to the specified scale
  vector<double> depth_vec;
  for(size_t i=0; i<xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  double scene_depth_median = vk::getMedian(depth_vec);
  double scale = Config::mapScale()/scene_depth_median;
  frame_cur->T_f_w_ = SE2_5(T_cur_from_ref_.se2() * frame_ref_->T_f_w_.se2());
  frame_cur->T_f_w_ = SE2_5(SE2(frame_cur->T_f_w_.se2().rotation_matrix(),
                                -frame_cur->T_f_w_.se2().rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()))));
  // For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->getSE3Inv();
    for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
    {
        Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
        Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
        if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
        {
            Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
            Point* new_point = new Point(pos);

            Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
            frame_cur->addFeature(ftr_cur);
            new_point->addFrameRef(ftr_cur);

            Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
            frame_ref_->addFeature(ftr_ref);
            new_point->addFrameRef(ftr_ref);
        }
    }
    SVO_INFO_STREAM("Init Homography RANSAC (inlier) is: "<<inliers_.size()
                    <<"\nKLT disparity is: "<<disparity<<" px average disparity."
                    <<"\nThe number of feature: "<<frame_cur->fts_.size());
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
  px_vec.clear(); px_vec.reserve(new_features.size());
  f_vec.clear(); f_vec.reserve(new_features.size());
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
  const double klt_win_size = 30.0;//30.0
  const int klt_max_iter = 30;//30
  const double klt_eps = 0.001;
  vector<uchar> status;
  vector<float> error;
  vector<float> min_eig_vec;
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
  cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                           px_ref, px_cur,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size),
                           4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  f_cur.clear(); f_cur.reserve(px_cur.size());
  disparities.clear(); disparities.reserve(px_cur.size());
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
}

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
  vk::computeInliers(f_cur, f_ref,
                     Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                     reprojection_threshold, focal_length,
                     xyz_in_cur, inliers, outliers);
  T_cur_from_ref=SE2_5(Homography.T_c2_from_c1);
}


} // namespace initialization
} // namespace svo
