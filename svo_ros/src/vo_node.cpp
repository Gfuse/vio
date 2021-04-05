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
#include <ros/package.h>
#include <string>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo_ros/visualizer.h>
#include <vikit/params_helper.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/String.h>
#include <message_filters/synchronizer.h>
#include <image_transport/image_transport.h>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <vikit/abstract_camera.h>
#include <vikit/camera_loader.h>
#include <vikit/user_input_thread.h>


namespace svo {

/// SVO Interface
class VoNode
{
public:
      svo::FrameHandlerMono* vo_;
      svo::Visualizer visualizer_;
      bool publish_markers_;                 //!< publish only the minimal amount of info (choice for embedded devices)
      vk::AbstractCamera* cam_;
      FILE* logtime;
      bool quit_;
      VoNode();
      ~VoNode();
      void imgCb(const sensor_msgs::ImageConstPtr& msg);
};

VoNode::VoNode() :
      vo_(NULL),
      publish_markers_(vk::getParam<bool>("svo/publish_markers", true)),
      cam_(NULL),
      quit_(false)
{
      // Create Camera
      if(!vk::camera_loader::loadFromRosNs("svo", cam_))
        throw std::runtime_error("Camera model not correctly specified.");

      // Get initial position and orientation
      visualizer_.T_world_from_vision_ = Sophus::SE3(
          vk::rpy2dcm(Vector3d(vk::getParam<double>("svo/init_rx", 0.0),
                               vk::getParam<double>("svo/init_ry", 0.0),
                               vk::getParam<double>("svo/init_rz", 0.0))),
          Eigen::Vector3d(vk::getParam<double>("svo/init_tx", 0.0),
                          vk::getParam<double>("svo/init_ty", 0.0),
                          vk::getParam<double>("svo/init_tz", 0.0)));

      // Init VO and start
      vo_ = new svo::FrameHandlerMono(cam_);
      vo_->start();
      logtime = fopen((std::string(getenv("HOME"))+"/Project/time.txt").c_str(), "w+");

}

VoNode::~VoNode()
{
    fclose(logtime);
    delete logtime;
    delete vo_;
    delete cam_;
}

void VoNode::imgCb(const sensor_msgs::ImageConstPtr& msg)
{
    auto start=std::chrono::steady_clock::now();
      try {
          cv::Mat img;
          img = cv_bridge::toCvShare(msg, "mono8")->image;
          vo_->addImage(img, msg->header.stamp.toSec());
          visualizer_.publishMinimal(img, vo_->lastFrame(), *vo_, msg->header.stamp.toSec());


          if(publish_markers_ && vo_->stage() != FrameHandlerBase::STAGE_PAUSED)
              visualizer_.visualizeMarkers(vo_->lastFrame(), vo_->coreKeyframes(), vo_->map());

          if(vo_->stage() == FrameHandlerMono::STAGE_PAUSED)
              usleep(100000);
          fprintf(logtime, "%s\n",std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()).c_str());
      } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
      }
}
} // namespace svo

int main(int argc, char **argv)
{
  ros::init(argc, argv, "svo");
  ros::NodeHandle nh;
  std::cout << "create vo_node" << std::endl;
  svo::VoNode vo_node;

  // subscribe to cam msgs
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber it_sub = it.subscribe(vk::getParam<std::string>("svo/cam_topic", "camera/image_raw"), 5, &svo::VoNode::imgCb, &vo_node);

  // start processing callbacks
  while(ros::ok() && !vo_node.quit_)
  {
    ros::spinOnce();
    // TODO check when last image was processed. when too long ago. publish warning that no msgs are received!
  }

  printf("SVO terminated.\n");
  return 0;
}
