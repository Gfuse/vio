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
#include <vio/frame_handler_mono.h>
#include <vio/map.h>
#include <vio/frame.h>
#include <vio/feature.h>
#include <vio/point.h>
#include <vio/pose_optimizer.h>
#include <vio/sparse_img_align.h>
#include <vio/depth_filter.h>
#include <vio/for_it.hpp>
#include <assert.h>

namespace vio {

FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam,Eigen::Matrix<double,3,1>& init) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL),
  ukfPtr_(init),
  time_(ros::Time::now())
{
    gpu_fast_= new opencl(cam_);
    gpu_fast_->make_kernel("fast_gray");
    cv::Mat img=cv::Mat(cv::Size(700, 500),CV_8UC1);
    gpu_fast_->write_buf(0,0,img);
    cl_int2 corners_[350000]={0};
    gpu_fast_->write_buf(0,1,350000,corners_);
    int icorner[1]={0};
    gpu_fast_->write_buf(0,2,1,icorner);
    gpu_fast_->write_buf(0,3,(1 << 18));
    gpu_fast_->make_kernel("compute_residual");
    gpu_fast_->write_buf(1,0,img);
    gpu_fast_->write_buf(1,1,img);
    cl_float3 pose[1]={0.0};
    gpu_fast_->write_buf(1,2,3,pose);
    gpu_fast_->write_buf(1,3,3,pose);
    cl_float3 f[300]={0};
    gpu_fast_->write_buf(1,4,300,f);
    cl_float2 px[300]={0};
    gpu_fast_->write_buf(1,5,300,px);
    float  e[300]={0};
    gpu_fast_->write_buf(1,7,300,e);
    float H[9*300]={0};
    gpu_fast_->write_buf(1,8,9*300,H);
    cl_float3 J[300];
    gpu_fast_->write_buf(1,9,300,J);
    float chi2[300]={0.01};
    gpu_fast_->write_buf(1,10,300,chi2);
    klt_homography_init_=new initialization::KltHomographyInit(gpu_fast_);
    initialize();
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

void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp,const ros::Time& time)
{
  if(!startFrameProcessingCommon(timestamp)){
      return;
  }

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();
  overlap_kfs_.clear();

  // create new frame
  new_frame_=boost::make_shared<Frame>(cam_, img.clone(), timestamp);
  time_=time;
  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    res = processFrame();
  else if(stage_ == STAGE_SECOND_FRAME)
    res = processSecondFrame();
  else if(stage_ == STAGE_FIRST_FRAME)
    res = processFirstFrame();

  // set last frame
  if(!new_frame_->fts_.empty() || stage_ != STAGE_DEFAULT_FRAME)last_frame_ = new_frame_;
  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
  if(stage_ == STAGE_RELOCALIZING){
      depthFilter()->stopThread();
      start();
  }
}

FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  auto init_f= ukfPtr_.get_location();
  new_frame_->T_f_w_=init_f.second;
  new_frame_->Cov_ = init_f.first;
  if(klt_homography_init_->addFirstFrame(new_frame_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;
  new_frame_->setKeyframe();
  map_.reset();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_SECOND_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
  auto init_f= ukfPtr_.get_location();
  new_frame_->T_f_w_=init_f.second;
  new_frame_->Cov_ = init_f.first;
  initialization::InitResult res = klt_homography_init_->addSecondFrame(new_frame_);
  auto result=ukfPtr_.UpdateSvo(new_frame_->T_f_w_.se2().translation()(0),new_frame_->T_f_w_.se2().translation()(1),new_frame_->T_f_w_.pitch(),1000,time_);
  new_frame_->T_f_w_ = result.second;
  new_frame_->Cov_ = result.first;
  if(res == initialization::NO_KEYFRAME){
      return RESULT_NO_KEYFRAME;
  }else if(res == initialization::FAILURE){
      stage_ = STAGE_FIRST_FRAME;
      return RESULT_FAILURE;
  }

  new_frame_->setKeyframe();
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);

  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);
  // add frame to map
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  klt_homography_init_->reset();
  SVO_INFO_STREAM("Init: triangulated initial map. Vio is starting ...");
  return RESULT_IS_KEYFRAME;
}


FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
  auto init_f= ukfPtr_.get_location();
  new_frame_->T_f_w_=init_f.second;
  new_frame_->Cov_ = init_f.first;
  // sparse image align
  //SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),30, SparseImgAlign::LevenbergMarquardt, false, false);
  //assert(new_frame_.get()== nullptr);
  SparseImgAlignGpu img_align(Config::kltMaxLevel(), Config::kltMinLevel(),30, SparseImgAlignGpu::GaussNewton, false,gpu_fast_);
  if(img_align.run(last_frame_, new_frame_)==0)return  RESULT_FAILURE;
 // reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  //  reprojector_.reprojectMap1(new_frame_, last_frame_, overlap_kfs_);
    reprojector_.reprojectMap2(new_frame_, overlap_kfs_);
  //assert (false);

  std::cout<<"Reprojection Map nPoint: "<<overlap_kfs_.back().second
             <<"\tnCell: "<<reprojector_.n_trials_<<"\t nMatches: "<<reprojector_.n_matches_
             <<"\t distance between two frames: "<<
             (last_frame_->T_f_w_.se2().translation()-new_frame_->T_f_w_.se2().translation()).norm()<<'\n';
  if( reprojector_.n_trials_ < 10)return RESULT_FAILURE;
  size_t sfba_n_edges_final=0;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
            Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
            new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  if((last_frame_->T_f_w_.se2().translation()-new_frame_->T_f_w_.se2().translation()).norm()>0.5 ||
       fabs(last_frame_->T_f_w_.pitch()-new_frame_->T_f_w_.pitch())>M_PI){
        return RESULT_FAILURE;
    }
  auto result=ukfPtr_.UpdateSvo(new_frame_->T_f_w_.se2().translation()(0),
                                new_frame_->T_f_w_.se2().translation()(1),new_frame_->T_f_w_.pitch(),sfba_n_edges_final,time_);
  new_frame_->T_f_w_ =result.second;
  new_frame_->Cov_ = result.first;
  if(sfba_n_edges_final < Config::qualityMinFts() || reprojector_.n_matches_ < Config::qualityMinFts()){
      return RESULT_FAILURE;
  }

  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
      return RESULT_FAILURE;
  }
  double depth_mean, depth_min;

  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);

  if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)//edited
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
  //std::cout<<"Svo pipline was finished\n";
  return RESULT_IS_KEYFRAME;
}

FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
    FramePtr ref_keyframe)
{
  return RESULT_FAILURE;
}

bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
    /*
  FramePtr ref_keyframe;
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  if(res != RESULT_FAILURE) {
    last_frame_ = new_frame_;
    return true;
  }*/
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
  if(scene_depth_mean >10.0)return true;
  for(auto&& it:overlap_kfs_)
  {
    if(fabs(it.first->T_f_w_.pitch()-new_frame_->T_f_w_.pitch())>0.0523599)return true;
    if(fabs((it.first->T_f_w_.translation()-new_frame_->T_f_w_.translation()).norm()) > Config::kfSelectMinDist())
        return true;
  }
  return false;
}

void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size()-1);
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
}
void FrameHandlerMono::UpdateIMU(double* value,const ros::Time& time){
    if(value== nullptr)return;
    ukfPtr_.UpdateIMU(value[0],value[1],value[2],time);
}
void FrameHandlerMono::UpdateCmd(double* value,const ros::Time& time){
    if(value== nullptr)return;
    ukfPtr_.UpdateCmd(value[0],value[1],value[2],time);
}

} // namespace vio
