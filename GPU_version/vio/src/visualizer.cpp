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

#include <gpu_svo/visualizer.h>
#include <gpu_svo/frame_handler_mono.h>
#include <gpu_svo/point.h>
#include <gpu_svo/map.h>
#include <gpu_svo/feature.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <gpu_svo/timer.h>
#include <gpu_svo/output_helper.h>
#include <deque>

namespace svo {

Visualizer::
Visualizer() :
    pnh_("~"),
    trace_id_(0),
    T_world_from_vision_(Matrix3d::Identity(), Vector3d::Zero())
{
  // Init ROS Marker Publishers
  //pub_frames_ = pnh_.advertise<visualization_msgs::Marker>("keyframes", 10);
  pub_points_ = pnh_.advertise<visualization_msgs::Marker>("points", 1000);
  pub_pose_with_cov_ = pnh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose_cov",10);
}

void Visualizer::publishMinimal(
    const cv::Mat& img,
    const FramePtr& frame,
    const FrameHandlerMono& slam,
    const double timestamp)
{
  ++trace_id_;
  std_msgs::Header header_msg;
  header_msg.frame_id = "/world";
  header_msg.seq = trace_id_;
  header_msg.stamp = ros::Time(timestamp);
  if(pub_pose_with_cov_.getNumSubscribers() > 0 && slam.stage() == FrameHandlerBase::STAGE_DEFAULT_FRAME)
  {
    // publish cam in world frame (Estimated odometry in the worls frame)
    Quaterniond q(AngleAxisd(frame->T_f_w_.pitch(), Vector3d::UnitZ()));
    geometry_msgs::PoseWithCovarianceStampedPtr msg_pose_with_cov(new geometry_msgs::PoseWithCovarianceStamped);
      msg_pose_with_cov->header = header_msg;
      msg_pose_with_cov->pose.pose.position.x = frame->se3().translation().x();
      msg_pose_with_cov->pose.pose.position.y = frame->se3().translation().z();
      msg_pose_with_cov->pose.pose.position.z = 0.0;
      msg_pose_with_cov->pose.pose.orientation.x = q.x();
      msg_pose_with_cov->pose.pose.orientation.y = q.y();
      msg_pose_with_cov->pose.pose.orientation.z = q.z();
      msg_pose_with_cov->pose.pose.orientation.w = q.w();
        msg_pose_with_cov->pose.covariance = {frame->Cov_(0,0),frame->Cov_(0,1),0.0,0.0,frame->Cov_(0,2),0.0,
                                              frame->Cov_(1,0),frame->Cov_(1,1),0.0,0.0,frame->Cov_(1,2),0.0,
                                              0.0,0.0,0.0000001,0.0,0.0,0.0,
                                              0.0,0.0,0.0,0.0000001,0.0,0.0,
                                              frame->Cov_(2,0),frame->Cov_(2,1),0.0,0.0,frame->Cov_(2,2),0.0,
                                              0.0,0.0,0.0,0.0,0.0,0.0000001};
    pub_pose_with_cov_.publish(msg_pose_with_cov);
    if(pub_frames_.getNumSubscribers() > 0 || pub_points_.getNumSubscribers() > 0){
          publishPointMarker(
                  pub_points_, Eigen::Vector3d(frame->se3().translation().x(),frame->se3().translation().z(),0.0), "trajectory",
                  ros::Time::now(), trace_id_, 0, 0.01, Vector3d(0.,0.,0.5));
    }
  }

}

void Visualizer::visualizeMarkers(
    const FramePtr& frame,
    const set<FramePtr>& core_kfs,
    const Map& map)
{
  if(frame == NULL)
    return;

  publishTfTransform(
      frame->se3()*T_world_from_vision_.inverse(),
      ros::Time(frame->timestamp_), "cam_pos", "world", br_);

  if(pub_frames_.getNumSubscribers() > 0 || pub_points_.getNumSubscribers() > 0)
  {
    publishPointMarker(
        pub_points_, T_world_from_vision_*Vector3d(frame->pos()(0),0.0,frame->pos()(1)), "trajectory",
        ros::Time::now(), trace_id_, 0, 0.006, Vector3d(0.,0.,0.5));
    removeDeletedPts(map);
  }
}

void Visualizer::removeDeletedPts(const Map& map)
{
  if(pub_points_.getNumSubscribers() > 0)
  {
    for(list<Point*>::const_iterator it=map.trash_points_.begin(); it!=map.trash_points_.end(); ++it)
      publishPointMarker(pub_points_, Vector3d(), "pts", ros::Time::now(), (*it)->id_, 2, 0.006, Vector3d());
  }
}

} // end namespace gpu_svo
