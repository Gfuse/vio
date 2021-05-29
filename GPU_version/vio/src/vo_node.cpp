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
#include <gpu_svo/frame_handler_mono.h>
#include <vio_gpu/visualizer.h>
#include <gpu_svo/params_helper.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <image_transport/image_transport.h>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <gpu_svo/abstract_camera.h>
#include <vio_gpu/camera_loader.h>
#include <gpu_svo/frame.h>


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
      boost::thread* imu_th;
      VoNode();
      ~VoNode();
      void imgCb(const sensor_msgs::ImageConstPtr& msg);
      void imuCb(const sensor_msgs::ImuPtr& imu);
      void imu();
private:
    double* imu_;
};

VoNode::VoNode() :
      vo_(NULL),
      publish_markers_(vk::getParam<bool>("vio/publish_markers", true)),
      cam_(NULL),
      quit_(false)
{
      // Create Camera
      imu_=(double *)calloc(static_cast<std::size_t>(6), sizeof(double));
      if(!vk::camera_loader::loadFromRosNs("vio", cam_))
        throw std::runtime_error("Camera model not correctly specified.");
      // Get initial position and orientation
      visualizer_.T_world_from_vision_ = Sophus::SE3(
          vk::rpy2dcm(Vector3d(vk::getParam<double>("vio/init_rx", 0.0),
                               vk::getParam<double>("vio/init_ry", 0.0),
                               vk::getParam<double>("vio/init_rz", 0.0))),
          Eigen::Vector3d(vk::getParam<double>("vio/init_tx", 0.0),
                          vk::getParam<double>("vio/init_ty", 0.0),
                          vk::getParam<double>("vio/init_tz", 0.0)));
      // Init VO and start
      vo_ = new svo::FrameHandlerMono(cam_,visualizer_.T_world_from_vision_);
      vo_->start();
    imu_th=new boost::thread(&VoNode::imu,this);
    logtime = fopen((std::string(getenv("HOME"))+"/time.txt").c_str(), "w+");

}

VoNode::~VoNode()
{
    quit_=true;
    fclose(logtime);
    imu_th->join();
    delete imu_th;
    delete logtime;
    delete vo_;
    delete cam_;
}

void VoNode::imgCb(const sensor_msgs::ImageConstPtr& msg)
{
      try {
          auto start=std::chrono::steady_clock::now();
          cv::Mat img=cv_bridge::toCvShare(msg, "mono8")->image;
          vo_->addImage(img, msg->header.stamp.toSec());
          visualizer_.publishMinimal(img, vo_->lastFrame(), *vo_, msg->header.stamp.toSec());
          if(vo_->stage() == FrameHandlerMono::STAGE_PAUSED)
              usleep(100000);
          fprintf(logtime, "%s\n",std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()).c_str());

      } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
      }
}
void VoNode::imuCb(const sensor_msgs::ImuPtr &imu) {
    double imu_in[6];
    imu_in[0] = 0.1*imu_[0]+0.9*imu->linear_acceleration.x;
    imu_in[1] = 0.1*imu_[1]+0.9*imu->linear_acceleration.y;
    imu_in[2] = 0.1*imu_[2]+0.9*imu->linear_acceleration.z;
    imu_in[3] = 0.1*imu_[3]+0.9*imu->angular_velocity.x;
    imu_in[4] = 0.1*imu_[4]+0.9*imu->angular_velocity.y;
    imu_in[5] = 0.1*imu_[5]+0.9*imu->angular_velocity.z;
    memcpy(imu_, imu_in, static_cast<std::size_t>(6*sizeof(double)));
    vo_->imu_integPtr_->update(imu_in);
}
void VoNode::imu(){
    ros::NodeHandle nh;
    // subscribe to cam msgs
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber it_sub = it.subscribe(vk::getParam<std::string>("vio/cam_topic", "camera/image_raw"), 1, &svo::VoNode::imgCb, this);
    // start processing callbacks
    while(ros::ok() && !quit_)
    {
        ros::spinOnce();
    }

}
} // namespace svo

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vio");
  svo::VoNode vo_node;
  ros::NodeHandle nh;
  ros::Subscriber imu_sub=nh.subscribe(vk::getParam<std::string>("vio/imu_topic", "imu/raw"),1,&svo::VoNode::imuCb, &vo_node);
  // start processing callbacks
  while(ros::ok() && !vo_node.quit_)
  {
      ros::spinOnce();
      usleep(1000);
      // TODO check when last image was processed. when too long ago. publish warning that no msgs are received!
  }
  printf("SVO terminated.\n");
  return 0;
}
