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

#include <vio/feature_detection.h>
#include <vio/feature.h>
#include <vio/vision.h>

namespace vio {
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
  for(auto&& i:fts){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  };
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
    std::shared_ptr<Frame> frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
    list<shared_ptr<Feature>>& fts,
    cv::Mat* descriptors){
  Corners corners(grid_n_cols_*grid_n_rows_, Corner(0,0,0.0,0,0.0f));
  std::vector<cv::KeyPoint> keypoints;
  fts.clear();
  for(int L=0; L<n_pyr_levels_; ++L)
  {
    const int scale = (1<<L);
    cv::Mat img=img_pyr.at(L);
    if(!gpu_fast_->reload_buf(0,0,img)){
        ROS_ERROR("Failed to write into GPU goodbye :)");
        exit(0);
    };
    cl_int icorner[1]={0};
    if(!gpu_fast_->reload_buf(0,2,icorner)){
          ROS_ERROR("Failed to write into GPU goodbye :)");
          exit(0);
      };
    gpu_fast_->run(0,img.cols,img.rows);
    cl_int count[1]={0};
    gpu_fast_->read(0,2,1,count);
    cl_int2 fast_corners[350000]={0};
    gpu_fast_->read(0,1,count[0],fast_corners);
    if(count[0]<1 ){
          ROS_ERROR("Can not communicate with GPU");
          exit(0);
    }
    for(uint i=0;i<count[0];++i)
    {
      if(fast_corners[i].x<0 || fast_corners[i].x>img_pyr[L].cols || fast_corners[i].y>img_pyr[L].rows || fast_corners[i].y<0){
          continue;
      }
      const int k = static_cast<int>((fast_corners[i].y*scale)/cell_size_)*grid_n_cols_
                  + static_cast<int>((fast_corners[i].x*scale)/cell_size_);

      if(grid_occupancy_[k] || k > corners.size()){
          continue;
      }
      const float score = vk::shiTomasiScore(img_pyr[L], fast_corners[i].x, fast_corners[i].y);
      if(score > corners.at(k).score){
          corners.at(k) = Corner(fast_corners[i].x, fast_corners[i].y, score, L, 0.0f);
          fts.push_back(make_shared<Feature>(frame, Vector2d(corners.at(k).x, corners.at(k).y)*scale, corners.at(k).level));
          if(descriptors)keypoints.push_back(cv::KeyPoint(fast_corners[i].x*scale, fast_corners[i].y*scale, 7));
      }
    }
  }
  if(descriptors){
      cv::Ptr<cv::xfeatures2d::FREAK> extractor = cv::xfeatures2d::FREAK::create(true, true, 22.0f, 4);
      extractor->compute(frame->img(), keypoints, *descriptors);
  }
  resetGrid();
};

} // namespace feature_detection
} // namespace vio

