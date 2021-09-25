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

#include <stdexcept>
#include <vio/math_utils.h>
#include <vio/point.h>
#include <vio/frame.h>
#include <vio/feature.h>
 
namespace vio {

int Point::point_counter_ = 0;

Point::Point(const Vector3d& pos) :
  id_(point_counter_++),
  pos_(pos),
  normal_set_(false),
  n_obs_(0),
  v_pt_(NULL),
  last_published_ts_(0),
  last_frame_overlap_id_(-10),
  type_(TYPE_UNKNOWN),
  n_failed_reproj_(0),
  n_succeeded_reproj_(0),
  last_structure_optim_(0)
{}

Point::Point(const Vector3d& pos, std::shared_ptr<Feature> ftr) :
  id_(point_counter_++),
  pos_(pos),
  normal_set_(false),
  n_obs_(1),
  v_pt_(NULL),
  last_published_ts_(0),
  last_frame_overlap_id_(-10),
  type_(TYPE_UNKNOWN),
  n_failed_reproj_(0),
  n_succeeded_reproj_(0),
  last_structure_optim_(0)
{
  obs_.push_front(ftr);
}

Point::~Point()
{}

void Point::addFrameRef(std::shared_ptr<Feature> ftr)
{
    boost::unique_lock<boost::mutex> lock(point_mut_);
  obs_.push_front(ftr);
  ++n_obs_;
}

std::shared_ptr<Feature> Point::findFrameRef(FramePtr frame)
{
    boost::unique_lock<boost::mutex> lock(point_mut_);
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
    if((*it)->frame == frame)
      return *it;
  return NULL;    // no keyframe found
}

bool Point::deleteFrameRef(FramePtr frame)
{
    boost::unique_lock<boost::mutex> lock(point_mut_);
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
  {
    if(!(*it)->frame){
        obs_.erase(it);
        continue;
    }
    if((*it)->frame->id_ == frame->id_)
    {
      obs_.erase(it);
      return true;
    }
  }
  return false;
}

void Point::initNormal()
{
  assert(!obs_.empty());
  const std::shared_ptr<Feature> ftr = obs_.back();
  assert(ftr->frame != NULL);
  normal_ = ftr->frame->se3().rotation_matrix().transpose()*(-ftr->f);
  double x=pow(20/(pos_-Vector3d(ftr->frame->pos()(0),0.0,ftr->frame->pos()(1))).norm(),2);
  normal_information_ = DiagonalMatrix<double,3,3>(x, 1.0, 1.0);
  normal_set_ = true;
}

bool Point::getCloseViewObs(const Vector2d& framepos, std::shared_ptr<Feature>& ftr,int id) const
{
    boost::unique_lock<boost::mutex> lock(point_mut_);
    // TODO: get frame with same point of view AND same pyramid level!
  ftr= nullptr;
  if(id_<1)return false;
  if(obs_.size()<1)return false;
  Vector3d obs_dir(Vector3d(framepos(0),1e-9,framepos(1)) + pos_); obs_dir.normalize();
  double max_cos_angle = 1.0;
  try{
      for(auto&& ob:obs_){
          if(!ob->frame)continue;
          if(ob->frame->id_==id){
              ftr = ob;
              return true;
          }
          Vector2d t=ob->frame->pos();
          Vector3d dir(Vector3d(t.x(),1e-9,t.y()) + pos_);
          double cos_angle = obs_dir.dot(dir.normalized());
          if(cos_angle < max_cos_angle)
          {
              max_cos_angle = cos_angle;
              ftr = ob;
          }
      }
  } catch (std::exception& e) {
      std::cerr << "Exception caught : " << e.what() << std::endl;
      exit(0);
  }
 
  if(ftr== nullptr)return false;
  return true;
}

void Point::optimize(const size_t n_iter)
{
  boost::unique_lock<boost::mutex> lock(point_mut_);
  Vector3d old_point = pos_;
  double chi2 = 0.0;
  Matrix3d A;
  Vector3d b;

  for(size_t i=0; i<n_iter; i++)
  {
    A.setZero();
    b.setZero();
    double new_chi2 = 0.0;

    // compute residuals
    for(auto it=obs_.begin(); it!=obs_.end(); ++it)
    {
      Matrix23d J;
      const Vector3d p_in_f((*it)->frame->se3()*pos_);
      Point::jacobian_xyz2uv(p_in_f, (*it)->frame->se3().rotation_matrix(), J);
      const Vector2d e(vk::project2d((*it)->f) - vk::project2d(p_in_f));
      new_chi2 += e.squaredNorm();
      A.noalias() += J.transpose() * J;
      b.noalias() -= J.transpose() * e;
    }

    // solve linear system
    const Vector3d dp(A.ldlt().solve(b));

    // check if error increased
    if((i > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dp[0]))
    {

      pos_ = old_point; // roll-back
      break;
    }
    // update the model
    Vector3d new_point = pos_ + dp;
    old_point = pos_;
    pos_ = new_point;
    chi2 = new_chi2;

    // stop when converged
    if(vk::norm_max(dp) <= EPS)
      break;
  }

}

} // namespace vio
