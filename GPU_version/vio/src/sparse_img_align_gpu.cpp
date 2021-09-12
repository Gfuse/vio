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
  eps_ = 1e-10;
}

size_t SparseImgAlignGpu::run(FramePtr ref_frame, FramePtr cur_frame, FILE* log)
{
  reset();
  /*if(ref_frame->fts_.empty()) // more than 10
  {
    return 0;
  }*/
  cl_float3 ref_pos[1]={(float)ref_frame->pos()(0),(float)ref_frame->pos()(1),(float)ref_frame->T_f_w_.pitch()};
/*#if VIO_DEBUG
    fprintf(log,"[%s] ref pose : , x:%f, z=%f, pitch:%f\n",vio::time_in_HH_MM_SS_MMM().c_str(),
            ref_pos[0].x, ref_pos[0].y, ref_pos[0].z);
#endif*/
  cl_float3 cur_pos[1]={(float)cur_frame->pos()(0),(float)cur_frame->pos()(1),(float)cur_frame->T_f_w_.pitch()};
/*#if VIO_DEBUG
    fprintf(log,"[%s] cur pose : , x:%f, z=%f, pitch:%f\n",vio::time_in_HH_MM_SS_MMM().c_str(),
            cur_pos[0].x, cur_pos[0].y, cur_pos[0].z);
#endif*/
  residual_->reload_buf(1,2,cur_pos);
  residual_->reload_buf(1,3,ref_pos);
  cl_float3 features[ref_frame->fts_.size()];
  cl_float2 featue_px[ref_frame->fts_.size()];
  feature_counter_ = 0; // is used to compute the index of the cached jacobian
  for(auto it=ref_frame->fts_.begin(); it!=ref_frame->fts_.end();++it){
      if(*it == nullptr)continue;
      if((*it)->point == nullptr)continue;
        Vector3d xyz_ref=(*it)->f*((*it)->point->pos_ - Eigen::Vector3d(ref_pos[0].x,0.0,ref_pos[0].y)).norm();
        features[feature_counter_].x=xyz_ref(0);
        features[feature_counter_].y=xyz_ref(1);
        features[feature_counter_].z=xyz_ref(2);
        featue_px[feature_counter_].x=(*it)->px(0);
        featue_px[feature_counter_].y=(*it)->px(1);
        ++feature_counter_;
  }
  if(!feature_counter_) // more than 10
  {
      return 0;
  }
  residual_->reload_buf(1,4,features);
  residual_->reload_buf(1,5,featue_px);
  SE2 T_cur(cur_frame->T_f_w_.se2());///TODO temporary, we can remove it
  for(level_=max_level_; level_>=min_level_; --level_)
  {
      cv::Mat cur_img = cur_frame->img_pyr_.at(level_);
      cv::Mat ref_img = ref_frame->img_pyr_.at(level_);
      residual_->reload_buf(1,0,cur_img);
      residual_->reload_buf(1,1,ref_img);
      residual_->write_buf(1,6,level_);
      mu_ = 1.0;
      optimize(T_cur);
  }
  cl_float3 pos[1]={0};
  residual_->read(1,2,1,pos);
  if(isnan(pos[0].x) || isnan(pos[0].y) || isnan(pos[0].z))return 0;
  cur_frame->T_f_w_ = SE2_5(pos[0].x,pos[0].y,pos[0].z);
  return 1;
}

double SparseImgAlignGpu::computeResiduals(
    bool linearize_system,
    bool compute_weight_scale)
{
    float H[9*300]={0};
    cl_float3 J[300]={0.0};
    float chi2_[300]={0.0};
    float scale=(float)scale_;
    residual_->write_buf(1,11,scale);
    residual_->reload_buf(1,10,chi2_);
    residual_->reload_buf(1,8,H);
    residual_->reload_buf(1,9,J);
    residual_->run(1,feature_counter_);
    float error[300]={0.0};
    float chi[300]={0.0};
    residual_->read(1,10,feature_counter_,chi);
    residual_->read(1,7,feature_counter_,error);
    for(int i = 1; i < feature_counter_; ++i){
        error[0] += error[i];
        chi[0] += chi[i];
    }
    scale_ = error[0]*(1.48f / feature_counter_);
    return chi[0]/(feature_counter_*8);
}

int SparseImgAlignGpu::solve()
{
    float H[300*9]={0.0};
    cl_float3 J[300]={0.0};
    residual_->read(1,8,feature_counter_*9,H);
    residual_->read(1,9,feature_counter_,J);
    for(int i = 1; i < feature_counter_; ++i){
        H[0] += H[i*9];
        H[1] += H[i*9+1];
        H[2] += H[i*9+2];
        H[3] += H[i*9+3];
        H[4] += H[i*9+4];
        H[5] += H[i*9+5];
        H[6] += H[i*9+6];
        H[7] += H[i*9+7];
        H[8] += H[i*9+8];
        J[0].x -= J[i].x;
        J[0].y -= J[i].y;
        J[0].z -= J[i].z;
    }
    double Hd[9]={(double)H[0],(double)H[1],(double)H[2],
                  (double)H[3],(double)H[4],(double)H[5],
                  (double)H[6],(double)H[7],(double)H[8]};
    double Jd[3]={(double)J[0].x,(double)J[0].y,(double)J[0].z};
    x_ = Eigen::Matrix<double,3,3>(Hd).ldlt().solve(Eigen::Vector3d(Jd[0],Jd[1],Jd[2]));
    if((bool) std::isnan((double) x_[0]) || (bool) std::isnan((double) x_[1]) ||(bool) std::isnan((double) x_[2]))
        return 0;
    double norm=x_.norm();
    if(norm<=0 ||norm > 1.0)x_=Eigen::Vector3d(0.1,0.1,0.1);
    return 1;
}
void SparseImgAlignGpu::update()
{
    cl_float3 pos[1]={0.0,0.0,0.0};
    residual_->read(1,2,1,pos);
    Sophus::SE2 update =  Sophus::SE2(pos[0].z,Eigen::Vector2d(pos[0].x,pos[0].y)) * Sophus::SE2::exp(-1.0*x_);
    pos[0].x=(float)update.translation()(0);
    pos[0].y=(float)update.translation()(1);
    pos[0].z=(float)atan2(update.so2().unit_complex().imag(),update.so2().unit_complex().real());
    residual_->reload_buf(1,2,pos);
}

void SparseImgAlignGpu::startIteration()
{}

void SparseImgAlignGpu::finishIteration()
{
}

} // namespace vio

