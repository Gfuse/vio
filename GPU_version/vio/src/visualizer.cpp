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
#include <geometry_msgs/PoseStamped.h>
#include <gpu_svo/timer.h>
#include <vio_gpu/output_helper.h>
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
    Matrix<double,6,6> Cov;
    // publish cam in world frame (Estimated odometry in the worls frame)
    SE3 T_world_from_cam(frame->T_f_w_.T->inverse());// do not confused with inverse. the T_f_w_ is the from camera to initial position with reference of camera
    double yaw=asin(2*(frame->T_f_w_.T->unit_quaternion().w()*frame->T_f_w_.T->unit_quaternion().y()-
            frame->T_f_w_.T->unit_quaternion().z()*frame->T_f_w_.T->unit_quaternion().x()));
    Quaterniond q;
    q = AngleAxisd(0.0, Vector3d::UnitX())
          * AngleAxisd(0.0, Vector3d::UnitY())
          * AngleAxisd(yaw-1.5708, Vector3d::UnitZ());
    Cov = T_world_from_cam.Adj()*frame->Cov_*T_world_from_cam.inverse().Adj();
    geometry_msgs::PoseWithCovarianceStampedPtr msg_pose_with_cov(new geometry_msgs::PoseWithCovarianceStamped);
      msg_pose_with_cov->header = header_msg;
      msg_pose_with_cov->pose.pose.position.x = -1.0*T_world_from_cam.translation().x();
      msg_pose_with_cov->pose.pose.position.y = -1.0*T_world_from_cam.translation().z();
      msg_pose_with_cov->pose.pose.position.z = 0.0;
      msg_pose_with_cov->pose.pose.orientation.x = q.x()/q.norm();
      msg_pose_with_cov->pose.pose.orientation.y = q.y()/q.norm();
      msg_pose_with_cov->pose.pose.orientation.z = q.z()/q.norm();
      msg_pose_with_cov->pose.pose.orientation.w = q.w()/q.norm();
    for(size_t i=0; i<36; ++i)
        msg_pose_with_cov->pose.covariance[i] = Cov(i%6, i/6);
    pub_pose_with_cov_.publish(msg_pose_with_cov);
    if(pub_frames_.getNumSubscribers() > 0 || pub_points_.getNumSubscribers() > 0){
          vk::output_helper::publishPointMarker(
                  pub_points_, Eigen::Vector3d(-1.0*T_world_from_cam.translation().x(),-1.0*T_world_from_cam.translation().z(),0.0), "trajectory",
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

  vk::output_helper::publishTfTransform(
      *frame->T_f_w_.T*T_world_from_vision_.inverse(),
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
