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
  num_obs_last_(0)
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
  num_obs_last_ = 0;
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
      if(it->point!=NULL && !it->point->pos_.hasNaN() && it->point->pos_.norm() !=0.){
          pts.push_back(it->point);
      }else{
          map_.safeDeletePoint(it->point);
      }
  }
  //max_n_pts = min(max_n_pts, pts.size());
  //nth_element(pts.begin(), pts.begin() + max_n_pts, pts.end(), ptLastOptimComparator);
  for(auto&& pt:pts){
      pt->optimize(max_iter);
      pt->last_frame_overlap_id_= frame->id_;
  }
}


} // namespace vio
