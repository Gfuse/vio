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

#ifndef SVO_VISUALIZER_H_
#define SVO_VISUALIZER_H_

#include <queue>
#include <ros/ros.h>
#include <gpu_svo/global.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

namespace svo {

class Frame;
class Point;
class Map;
class FrameHandlerMono;

typedef boost::shared_ptr<Frame> FramePtr;

/// This class bundles all functions to publish visualisation messages.
class Visualizer
{
public:
  ros::NodeHandle pnh_;
  size_t trace_id_;
  ros::Publisher pub_frames_;
  ros::Publisher pub_points_;
  ros::Publisher pub_pose_;
  ros::Publisher pub_info_;
  tf::TransformBroadcaster br_;
  SE3 T_world_from_vision_;

  Visualizer();

  ~Visualizer() {};

  void publishMinimal(
      const cv::Mat& img,
      const FramePtr& frame,
      const FrameHandlerMono& slam,
      const double timestamp);

  void visualizeMarkers(
      const FramePtr& frame,
      const set<FramePtr>& core_kfs,
      const Map& map);

  void removeDeletedPts(const Map& map);

};

} // end namespace svo

#endif /* SVO_VISUALIZER_H_ */
