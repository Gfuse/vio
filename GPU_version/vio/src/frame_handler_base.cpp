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

#include <vio/abstract_camera.h>
#include <stdlib.h>
#include <Eigen/StdVector>
#include <boost/bind.hpp>
#include <fstream>
#include <vio/frame_handler_base.h>
#include <vio/config.h>
#include <vio/feature.h>
#include <vio/matcher.h>
#include <vio/map.h>
#include <vio/point.h>

namespace vio
{

// definition of global and static variables which were declared in the header

FrameHandlerBase::FrameHandlerBase() :
  stage_(STAGE_PAUSED),
  set_reset_(false),
  set_start_(false),
  num_obs_last_(0),
  tracking_quality_(TRACKING_INSUFFICIENT)
{
}

FrameHandlerBase::~FrameHandlerBase()
{
}

bool FrameHandlerBase::startFrameProcessingCommon(const double timestamp)
{
  if(set_start_)
  {
    resetAll();
    stage_ = STAGE_FIRST_FRAME;
  }

  if(stage_ == STAGE_PAUSED)
    return false;


  // some cleanup from last iteration, can't do before because of visualization
  map_.emptyTrash();
  return true;
}

int FrameHandlerBase::finishFrameProcessingCommon(
    const size_t update_id,
    const UpdateResult dropout,
    const size_t num_observations)
{

  if(stage_ == STAGE_DEFAULT_FRAME)
  num_obs_last_ = num_observations;

  if(dropout == RESULT_FAILURE &&
      (stage_ == STAGE_DEFAULT_FRAME || stage_ == STAGE_RELOCALIZING ))
  {
    //stage_ = STAGE_RELOCALIZING;
    tracking_quality_ = TRACKING_INSUFFICIENT;
  }
  if(set_reset_)
    resetAll();

  return 0;
}

void FrameHandlerBase::resetCommon()
{
  map_.reset();
  stage_ = STAGE_PAUSED;
  set_reset_ = false;
  set_start_ = false;
  tracking_quality_ = TRACKING_INSUFFICIENT;
  num_obs_last_ = 0;
}

void FrameHandlerBase::setTrackingQuality(const size_t num_observations)
{
  tracking_quality_ = TRACKING_GOOD;
  if(num_observations < Config::qualityMinFts())
  {
    //SVO_WARN_STREAM_THROTTLE(0.5, "Tracking less than "<< Config::qualityMinFts() <<" features!");
    tracking_quality_ = TRACKING_INSUFFICIENT;
  }
  const int feature_drop = static_cast<int>(std::min(num_obs_last_, Config::maxFts())) - num_observations;
  if(feature_drop > Config::qualityMaxFtsDrop())
  {
    //SVO_WARN_STREAM("Lost "<< feature_drop <<" features!");
    tracking_quality_ = TRACKING_INSUFFICIENT;
  }
}

bool ptLastOptimComparator(Point* lhs, Point* rhs)
{
  return (lhs->last_structure_optim_ < rhs->last_structure_optim_);
}

void FrameHandlerBase::optimizeStructure(
    FramePtr frame,
    size_t max_n_pts,
    int max_iter)
{
  deque<std::shared_ptr<Point>> pts;
  for(auto&& it:frame->fts_){
      if(it->point!=NULL)pts.push_back(it->point);
  }
  //max_n_pts = min(max_n_pts, pts.size());
  //nth_element(pts.begin(), pts.begin() + max_n_pts, pts.end(), ptLastOptimComparator);
  for(auto&& pt:pts){
      pt->optimize(max_iter);
      pt->last_frame_overlap_id_= frame->id_;
  }
}


} // namespace vio
