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
#include <gpu_svo/frame_handler_mono.h>
#include <gpu_svo/map.h>
#include <gpu_svo/frame.h>
#include <gpu_svo/feature.h>
#include <gpu_svo/point.h>
#include <gpu_svo/pose_optimizer.h>
#include <gpu_svo/sparse_img_align.h>
#include <gpu_svo/depth_filter.h>
#include <gpu_svo/for_it.hpp>
#ifdef USE_BUNDLE_ADJUSTMENT
#include <gpu_svo/bundle_adjustment.h>
#endif

namespace svo {

FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam,Sophus::SE3& SE_init) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL)
{
    gpu_fast_= new opencl();
    gpu_fast_->make_kernel("fast_gray");
    cv::Mat img=cv::Mat(cv::Size(700, 500),CV_8UC1);
    gpu_fast_->write_buf(0,0,img);
    cl_int2 corners_[350000];
    gpu_fast_->write_buf(0,1,350000,corners_);
    int icorner[1]={0};
    gpu_fast_->write_buf(0,2,1,icorner);
    gpu_fast_->write_buf(0,3,(1 << 18));
    klt_homography_init_=new initialization::KltHomographyInit(gpu_fast_);
  initialize();
  imu_integPtr_=std::make_shared<Imu_Integration>(SE_init);
}

void FrameHandlerMono::initialize()
{
  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(),gpu_fast_, Config::nPyrLevels()));
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
  depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
  depth_filter_->startThread();
}

FrameHandlerMono::~FrameHandlerMono()
{
  delete depth_filter_;
}

void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp)
{
  if(!startFrameProcessingCommon(timestamp))
    return;

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();
  overlap_kfs_.clear();

  // create new frame
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));

  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    res = processFrame();
  else if(stage_ == STAGE_SECOND_FRAME)
    res = processSecondFrame();
  else if(stage_ == STAGE_FIRST_FRAME)
    res = processFirstFrame();
  else if(stage_ == STAGE_RELOCALIZING)
    res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));

  // set last frame
  last_frame_ = new_frame_;
  new_frame_.reset();
  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
}

FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  new_frame_->T_f_w_ = SE2_5(0.0,0.0,0.0);
  if(klt_homography_init_->addFirstFrame(new_frame_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_SECOND_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
  new_frame_->T_f_w_ = last_frame_->T_f_w_;
  initialization::InitResult res = klt_homography_init_->addSecondFrame(new_frame_);
  if(res == initialization::NO_KEYFRAME)
    return RESULT_NO_KEYFRAME;

  new_frame_->setKeyframe();
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);

  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);
  if(!imu_integPtr_->init(new_frame_)){
      return RESULT_FAILURE;
  }

  // add frame to map
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  klt_homography_init_->reset();
  SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
  return RESULT_IS_KEYFRAME;
}


FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
  SE3 tem=imu_integPtr_->preintegrate_predict();
  new_frame_->T_f_w_ = SE2_5(tem.translation().x(),tem.translation().z(),atan(tem.rotation_matrix()(0,2)/tem.rotation_matrix()(0,0)));
  //new_frame_->T_f_w_ = last_frame_->T_f_w_;
  // sparse image align
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::LevenbergMarquardt, false, false);
  img_align.run(last_frame_, new_frame_);
  // map reprojection & feature alignment
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  size_t sfba_n_edges_final=0;

  double sfba_thresh, sfba_error_init, sfba_error_final;
  if(!imu_integPtr_->predict(new_frame_,last_frame_,sfba_n_edges_final,reprojector_.n_matches_)){
        return RESULT_FAILURE;
  }
  pose_optimizer::optimizeGaussNewton(
            Config::poseOptimThresh(), Config::poseOptimNumIter(), true,
            new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);

  std::cerr<<"Reprojection Map: "<<"\t nPoint:"<<overlap_kfs_.back().second
  <<"\t nCell = "<<reprojector_.n_trials_<<"\t \t nMatches = "<<reprojector_.n_matches_
  <<" \t Reprojected points after opmization: "<<sfba_n_edges_final<<'\n';

  if(sfba_n_edges_final < Config::qualityMinFts() || reprojector_.n_matches_ < Config::qualityMinFts()){
      return RESULT_NO_KEYFRAME;
  }

  // structure optimization
//  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  //SVO_STOP_TIMER("point_optimizer");

  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    return RESULT_FAILURE;
  }
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  {
    depth_filter_->addFrame(new_frame_);
    return RESULT_NO_KEYFRAME;
  }
  new_frame_->setKeyframe();

  // new keyframe selected
  for(auto&& it:new_frame_->fts_){
      if(it->point != NULL)it->point->addFrameRef(it);
  }
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

  // init new depth-filters
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  map_.addKeyframe(new_frame_);
  std::cerr<<"Svo pipline was finished\n";
  return RESULT_IS_KEYFRAME;
}

FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
    const SE3& T_cur_ref,
    FramePtr ref_keyframe)
{
  if(ref_keyframe == nullptr)
  {
    return RESULT_FAILURE;
  }
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
  usleep(2000);
  if(img_align_n_tracked > Config::qualityMinFts())
  {
    SE2_5 T_f_w_last = last_frame_->T_f_w_;
    last_frame_ = ref_keyframe;
    FrameHandlerMono::UpdateResult res = processFrame();
    if(res != RESULT_FAILURE)
    {
      stage_ = STAGE_DEFAULT_FRAME;
    }
    else
      new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
    return res;
  }
  return RESULT_FAILURE;
}

bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
  FramePtr ref_keyframe;
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  if(res != RESULT_FAILURE) {
    last_frame_ = new_frame_;
    return true;
  }
  return false;
}

void FrameHandlerMono::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  core_kfs_.clear();
  overlap_kfs_.clear();
  depth_filter_->reset();
}

void FrameHandlerMono::setFirstFrame(const FramePtr& first_frame)
{
  resetAll();
  last_frame_ = first_frame;
  last_frame_->setKeyframe();
  map_.addKeyframe(last_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
}

bool FrameHandlerMono::needNewKf(double scene_depth_mean)
{
  for(auto&& it:overlap_kfs_)
  {
    Vector3d relpos = new_frame_->w2f(it.first->pos());
    if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist())
      return false;
  }
  return true;
}

void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size()-1);
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
}

} // namespace svo
