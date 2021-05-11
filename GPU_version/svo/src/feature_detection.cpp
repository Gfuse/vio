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

#include <gpu_svo/feature_detection.h>
#include <gpu_svo/feature.h>
#include <vikit/vision.h>

namespace svo {
namespace feature_detection {

AbstractDetector::AbstractDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        cell_size_(cell_size),
        n_pyr_levels_(n_pyr_levels),
        grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
        grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
        grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
{
}

void AbstractDetector::resetGrid()
{
  std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

void AbstractDetector::setExistingFeatures(const Features& fts)
{
  std::for_each(fts.begin(), fts.end(), [&](Feature* i){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  });
}

void AbstractDetector::setGridOccpuancy(const Vector2d& px)
{
  grid_occupancy_.at(
      static_cast<int>(px[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(px[0]/cell_size_)) = true;
}

FastDetector::FastDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    opencl* gpu_fast_,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels),gpu_fast_(gpu_fast_)
{
}

void FastDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
    Features& fts)
{
  Corners corners(grid_n_cols_*grid_n_rows_, Corner(0,0,detection_threshold,0,0.0f));
  for(int L=0; L<n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    cv::Mat img=img_pyr.at(L);
    gpu_fast_->reload_buf(0,0,img);
    int icorner[1]={0};
    gpu_fast_->reload_buf(0,2,icorner);
    gpu_fast_->run(0,img_pyr[L].cols,img_pyr[L].rows);
    cl_int2 fast_corners[350000];
    gpu_fast_->read(0,1,fast_corners);
    int count[1]={0};
    gpu_fast_->read(0,2,count);
    for(uint i=0;i<count[0];++i)
    {
      if(fast_corners[i].x<0 || fast_corners[i].x>img_pyr[L].cols || fast_corners[i].y>img_pyr[L].rows || fast_corners[i].y<0)
          continue;

      const int k = static_cast<int>((fast_corners[i].y*scale)/cell_size_)*grid_n_cols_
                  + static_cast<int>((fast_corners[i].x*scale)/cell_size_);

      if(grid_occupancy_[k])
        continue;

      const float score = vk::shiTomasiScore(img_pyr[L], fast_corners[i].x, fast_corners[i].y);

      if(score > corners.at(k).score){
          corners.at(k) = Corner(fast_corners[i].x*scale, fast_corners[i].y*scale, score, L, 0.0f);
          // Create feature for every corner that has high enough corner score
          if(corners.at(k).score > detection_threshold)
              fts.push_back(new Feature(frame, Vector2d(corners.at(k).x, corners.at(k).y), corners.at(k).level));
      }
    }

  }
  resetGrid();
}

} // namespace feature_detection
} // namespace svo

