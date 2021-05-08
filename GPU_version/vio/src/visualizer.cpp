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

#include <vio_gpu/visualizer.h>
#include <gpu_svo/frame_handler_mono.h>
#include <gpu_svo/point.h>
#include <gpu_svo/map.h>
#include <gpu_svo/feature.h>
#include <vio_gpu/Info.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <vikit/timer.h>
#include <vikit/output_helper.h>
#include <deque>

namespace svo {

Visualizer::
Visualizer() :
    pnh_("~"),
    trace_id_(0),
    T_world_from_vision_(Matrix3d::Identity(), Vector3d::Zero())
{
  // Init ROS Marker Publishers
  pub_frames_ = pnh_.advertise<visualization_msgs::Marker>("keyframes", 10);
  pub_points_ = pnh_.advertise<visualization_msgs::Marker>("points", 1000);
  pub_pose_ = pnh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose",10);
  pub_info_ = pnh_.advertise<vio_gpu::Info>("info", 10);
}

void Visualizer::publishMinimal(
    const cv::Mat& img,
    const FramePtr& frame,
    const FrameHandlerMono& slam,
    const double timestamp)
{
  ++trace_id_;
  std_msgs::Header header_msg;
  header_msg.frame_id = "/cam";
  header_msg.seq = trace_id_;
  header_msg.stamp = ros::Time(timestamp);

  // publish info msg.
  if(pub_info_.getNumSubscribers() > 0)
  {
      vio_gpu::Info msg_info;
    msg_info.header = header_msg;
    msg_info.processing_time = slam.lastProcessingTime();
    msg_info.keyframes.reserve(slam.map().keyframes_.size());
    for(list<FramePtr>::const_iterator it=slam.map().keyframes_.begin(); it!=slam.map().keyframes_.end(); ++it)
      msg_info.keyframes.push_back((*it)->id_);
    msg_info.stage = static_cast<int>(slam.stage());
    msg_info.tracking_quality = static_cast<int>(slam.trackingQuality());
    if(frame != NULL)
      msg_info.num_matches = slam.lastNumObservations();
    else
      msg_info.num_matches = 0;
    pub_info_.publish(msg_info);
  }
  if(pub_pose_.getNumSubscribers() > 0 && slam.stage() == FrameHandlerBase::STAGE_DEFAULT_FRAME)
  {
    Quaterniond q;
    Vector3d p;
    Matrix<double,6,6> Cov;
    // publish cam in world frame (Estimated odometry in the worls frame)
    SE3 T_world_from_cam(T_world_from_vision_*frame->T_f_w_.inverse());// do not confused with inverse. the T_f_w_ is the from camera to initial position with reference of camera
    q = Quaterniond(T_world_from_cam.rotation_matrix()*T_world_from_vision_.rotation_matrix().transpose());
    p = T_world_from_cam.translation();
    Cov = T_world_from_cam.Adj()*frame->Cov_*T_world_from_cam.inverse().Adj();
    geometry_msgs::PoseWithCovarianceStampedPtr msg_pose(new geometry_msgs::PoseWithCovarianceStamped);
    msg_pose->header = header_msg;
    msg_pose->pose.pose.position.x = p[0];
    msg_pose->pose.pose.position.y = p[1];
    msg_pose->pose.pose.position.z = p[2];
    msg_pose->pose.pose.orientation.x = q.x();
    msg_pose->pose.pose.orientation.y = q.y();
    msg_pose->pose.pose.orientation.z = q.z();
    msg_pose->pose.pose.orientation.w = q.w();
    for(size_t i=0; i<36; ++i)
      msg_pose->pose.covariance[i] = Cov(i%6, i/6);
    pub_pose_.publish(msg_pose);
  }
}

void Visualizer::visualizeMarkers(
    const FramePtr& frame,
    const set<FramePtr>& core_kfs,
    const Map& map)
{
  if(frame == NULL)
    return;

  vk::output_helper::publishTfTransform(
      frame->T_f_w_*T_world_from_vision_.inverse(),
      ros::Time(frame->timestamp_), "cam_pos", "world", br_);

  if(pub_frames_.getNumSubscribers() > 0 || pub_points_.getNumSubscribers() > 0)
  {
    vk::output_helper::publishPointMarker(
        pub_points_, T_world_from_vision_*frame->pos(), "trajectory",
        ros::Time::now(), trace_id_, 0, 0.006, Vector3d(0.,0.,0.5));
    removeDeletedPts(map);
  }
}

void Visualizer::removeDeletedPts(const Map& map)
{
  if(pub_points_.getNumSubscribers() > 0)
  {
    for(list<Point*>::const_iterator it=map.trash_points_.begin(); it!=map.trash_points_.end(); ++it)
      vk::output_helper::publishPointMarker(pub_points_, Vector3d(), "pts", ros::Time::now(), (*it)->id_, 2, 0.006, Vector3d());
  }
}

} // end namespace gpu_svo
