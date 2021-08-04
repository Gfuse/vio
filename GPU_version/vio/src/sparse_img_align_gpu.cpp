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

#include <vio/sparse_img_align_gpu.h>

namespace vio {

SparseImgAlignGpu::SparseImgAlignGpu(
    int max_level, int min_level, int n_iter,
    Method method, bool verbose,opencl* residual) :
        max_level_(max_level),
        min_level_(min_level),
        residual_(residual)
{
  n_iter_ = n_iter;
  n_iter_init_ = n_iter_;
  method_ = method;
  verbose_ = verbose;
  eps_ = 0.0001;
}

size_t SparseImgAlignGpu::run(FramePtr ref_frame, FramePtr cur_frame)
{
  reset();
  if(ref_frame->fts_.empty())
  {
    return 0;
  }
  double* camera=ref_frame->cam_->params();
  double ref_pos[3]={ref_frame->pos()(0),ref_frame->pos()(1),ref_frame->T_f_w_.pitch()};
  double cur_pos[3]={cur_frame->pos()(0),cur_frame->pos()(1),cur_frame->T_f_w_.pitch()};
  residual_->reload_buf(1,2,cur_pos);
  residual_->reload_buf(1,3,ref_pos);
  residual_->reload_buf(1,7,camera);
  cl_double3 features[ref_frame->fts_.size()];
  cl_double2 featue_px[ref_frame->fts_.size()];
  feature_counter_ = 0; // is used to compute the index of the cached jacobian
  for(auto it=ref_frame->fts_.begin(); it!=ref_frame->fts_.end();++it){
        Vector3d xyz_ref=(*it)->f*((*it)->point->pos_ - Eigen::Vector3d(ref_pos[0],0.0,ref_pos[1])).norm();
        features[feature_counter_].x=xyz_ref(0);
        features[feature_counter_].y=xyz_ref(1);
        features[feature_counter_].z=xyz_ref(2);
        featue_px[feature_counter_].x=(*it)->px(0);
        featue_px[feature_counter_].y=(*it)->px(1);
        ++feature_counter_;
  }
  residual_->reload_buf(1,4,features);
  residual_->reload_buf(1,5,featue_px);
  SE2 T_cur(cur_frame->T_f_w_.se2());///TODO temporary, we can remove it
  for(level_=max_level_; level_>=min_level_; --level_)
  {
      cv::Mat& cur_img = cur_frame->img_pyr_.at(level_);
      cv::Mat& ref_img = ref_frame->img_pyr_.at(level_);
      residual_->reload_buf(1,0,cur_img);
      residual_->reload_buf(1,1,ref_img);
      residual_->reload_buf(1,6,&level_);
    mu_ = 1.0;
    optimize(T_cur);
  }
  cur_frame->T_f_w_ = SE2_5(T_cur);

  return n_meas_/patch_area_;
}

double SparseImgAlignGpu::computeResiduals(
    bool linearize_system,
    bool compute_weight_scale)
{
  double chi2[1]={0};
  residual_->reload_buf(1,11,chi2);
  /// TODO GPU compute based on the feature
  residual_->run(1,feature_counter_);
  if(compute_weight_scale && iter_ == 0)  {
      float error[feature_counter_];
      residual_->read(1,8,error);
      std::vector<float> errors(error,error+feature_counter_);
      scale_ = scale_estimator_->compute(errors);
  }

  double out[1]={0.0};
  residual_->read(1,11,out);
  return out[0];
}

int SparseImgAlignGpu::solve()
{
    double H[9]={0};
    double J[3]={0};
    residual_->read(1,9,H);
    residual_->read(1,10,J);
    x_ = Eigen::Matrix<double,3,3>(H).ldlt().solve(Eigen::Matrix<double,3,1>(J));
    if((bool) std::isnan((double) x_[0]))
      return 0;
    return 1;
}
void SparseImgAlignGpu::update()
{
/// TODO the update situation may have a smarter solution
  double pos[3]={0};
  residual_->read(1,2,pos);
  SE2 update =  SE2(pos[2],Eigen::Vector2d(pos[0],pos[1])) * SE2::exp(-0.25*x_);
  pos[0]=update.translation()(0);
  pos[1]=update.translation()(1);
  pos[2]=atan(update.so2().unit_complex().imag()/update.so2().unit_complex().real());
  residual_->reload_buf(1,2,pos);
}

void SparseImgAlignGpu::startIteration()
{}

void SparseImgAlignGpu::finishIteration()
{
}

} // namespace vio

