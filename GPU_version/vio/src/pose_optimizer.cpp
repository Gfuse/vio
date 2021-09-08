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
#include <vio/pose_optimizer.h>
#include <vio/frame.h>
#include <vio/feature.h>
#include <vio/point.h>
#include <vio/robust_cost.h>
#include <vio/math_utils.h>
#include <sophus/se2.h>

namespace vio {
namespace pose_optimizer {

void optimizeGaussNewton(
    const double reproj_thresh,
    const size_t n_iter,
    const bool verbose,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs,
    FILE* log)
{
  // init
  double chi2(0.0);
  vector<double> chi2_vec_init, chi2_vec_final;
  vk::robust_cost::TukeyWeightFunction weight_function;
  SE2_5 T_old(frame->T_f_w_);
  Matrix3d A;
  Vector3d b;

  // compute the scale of the error for robust estimation
  std::vector<float> errors;
  for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point == NULL)
      continue;
    //Reprojection error
    Vector2d e = vk::project2d((*it)->f)
               - vk::project2d(Vector3d(frame->se3()*(*it)->point->pos_));

    e *= 1.0 / (1<<(*it)->level);
    errors.push_back(e.norm());
  }

  if(errors.empty())
    return;
  vk::robust_cost::MADScaleEstimator scale_estimator;
  estimated_scale = 0.5*scale_estimator.compute(errors);
#if VIO_DEBUG
    fprintf(log,"[%s] Init estimate scale:%f\n",vio::time_in_HH_MM_SS_MMM().c_str(),estimated_scale);
#endif

  num_obs = errors.size();
  chi2_vec_init.reserve(num_obs);
  chi2_vec_final.reserve(num_obs);
  double scale = estimated_scale;
  bool direction=true;
  size_t n_direction=0;
  for(size_t iter=0; iter<n_iter; iter++)
  {

    b.setZero();
    A.setZero();
    double new_chi2(0.0);

    // compute residual
    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
      if((*it)->point == NULL)
        continue;
      Matrix23d J;
      Vector3d xyz_f(Vector3d(frame->se3()*(*it)->point->pos_));
      frame->jacobian_xyz2uv_((*it)->f,(*it)->point->pos_,J);
      Vector2d e = vk::project2d((*it)->f) - vk::project2d(xyz_f);
/*#if VIO_DEBUG
        fprintf(log,"[%s] xyz_f:%f, %f, %f  Reprojection Error: %f, %f Feature: %f, %f, %f\n",
                vio::time_in_HH_MM_SS_MMM().c_str(),xyz_f.x(),xyz_f.y(),xyz_f.z(),e.x(),e.y(),
                        (*it)->point->pos_.x(),(*it)->point->pos_.y(),(*it)->point->pos_.z());
#endif*/
      double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
      e *= sqrt_inv_cov;
      if(iter == 0)
        chi2_vec_init.push_back(e.squaredNorm()); // just for debug
      J *= sqrt_inv_cov;
      double weight = weight_function.value(e.norm()/scale);
      A.noalias() += J.transpose()*J*weight;
      b.noalias() -= J.transpose()*e*weight;
      new_chi2 += e.squaredNorm()*weight;
    }

    // solve linear system
    Vector3d dT(A.ldlt().solve(b));

    // check if error increased
    if((iter > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dT[0]))
    {
#if VIO_DEBUG
          fprintf(log,"[%s] it:%d \t FAILURE \t new_chi2: %f\n",
                  vio::time_in_HH_MM_SS_MMM().c_str(),iter,new_chi2);
#endif
      frame->T_f_w_ = T_old; // roll-back
      break;
    }
    scale *=0.5;
    dT=0.1*dT;
    // update the model
    T_old = frame->T_f_w_;
    frame->T_f_w_=SE2_5(T_old.se2().translation().y()+dT.x(),T_old.se2().translation().y()+dT.y(),T_old.pitch()+dT.z());

    chi2 = new_chi2;
#if VIO_DEBUG
      fprintf(log,"[%s] it:%d \t Success \t new_chi2: %f \t dT = %f, %f, %f\n",
              vio::time_in_HH_MM_SS_MMM().c_str(),iter,new_chi2,dT.x(),dT.y(),dT.z());
#endif

    // stop when converged
    if(vk::norm_max(dT) <= EPS)
      break;
  }

  // Set covariance as inverse information matrix. Optimistic estimator!
  //const double pixel_variance=1.0;
  //frame->Cov_ = pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();

  // Remove Measurements with too large reprojection error
  double reproj_thresh_scaled = reproj_thresh / frame->cam_->errorMultiplier2();
  size_t n_deleted_refs = 0;
  for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point == NULL)
      continue;
    Vector2d e = vk::project2d((*it)->f) - vk::project2d(Vector3d(frame->se3()*(*it)->point->pos_));
    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
    e *= sqrt_inv_cov;
    chi2_vec_final.push_back(e.squaredNorm());
    if(e.norm() > reproj_thresh_scaled)
    {
      // we don't need to delete a reference in the point since it was not created yet
      (*it)->point = NULL;
      ++n_deleted_refs;
    }
  }
  error_init=0.0;
  error_final=0.0;
  if(!chi2_vec_init.empty())
    error_init = sqrt(vk::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
  if(!chi2_vec_final.empty())
    error_final = sqrt(vk::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();

  estimated_scale *= frame->cam_->errorMultiplier2();
  if(verbose)
    std::cout << "n deleted obs = " << n_deleted_refs
              << "\t scale = " << estimated_scale
              << "\t error init = " << error_init
              << "\t error end = " << error_final << std::endl;
  num_obs -= n_deleted_refs;
}

} // namespace pose_optimizer
} // namespace vio
