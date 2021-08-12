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
#define VIO_DEBUG false
#include <ros/ros.h>
#include <string>
#include <chrono>
#include <vio/frame_handler_mono.h>
#include <vio/params_helper.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Twist.h>
#include <image_transport/image_transport.h>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <vio/abstract_camera.h>
#include <vio/camera_loader.h>
#include <vio/frame.h>
#include <vio/getOdom.h>
#include <vio/start.h>
#include <vio/stop.h>
#include <vio/depth_filter.h>
#if VIO_DEBUG
#include <vio/visualizer.h>
#endif

class VioNode
{
public:
      vio::FrameHandlerMono* vo_;
      vk::AbstractCamera* cam_;
      VioNode();
      ~VioNode();
      void imgCb(const sensor_msgs::ImageConstPtr& msg);
      void imuCb(const sensor_msgs::ImuPtr& imu);
    void cmdCb(const geometry_msgs::TwistPtr &imu);
    void imu_th();
    bool getOdom(vio::getOdom::Request& req, vio::getOdom::Response& res);
    uint trace_id_= 0;
    bool start_=false;
    bool start(vio::start::Request& req, vio::start::Response& res){
        if(req.on==1){
            start_=true;
            if(vo_->stage()!=vio::FrameHandlerBase::STAGE_DEFAULT_FRAME)
                vo_->start();
            else{
                vo_->depthFilter()->startThread();
                vo_->reset();
            }

            imu_the_=new boost::thread(&VioNode::imu_th,this);
            ++trace_id_;
            res.ret=0;
        }else{
            res.ret=100;
        }
        return true;
    };

    bool stop(vio::stop::Request& req, vio::stop::Response& res){
        if(req.off==1 && imu_the_!=NULL){
            start_=false;
            vo_->depthFilter()->stopThread();
            imu_the_->interrupt();
            if(imu_the_->get_id()==boost::this_thread::get_id())
                imu_the_->detach();
            else
                imu_the_->join();
            imu_the_=NULL;
            res.ret=0;
        }else{
            res.ret=100;
        }
        return true;
    };
private:
    double* imu_;
    double* cmd_;
    ros::Time imu_time_;
    boost::thread* imu_the_= nullptr;
#if VIO_DEBUG
    vio::Visualizer visualizer_;
#endif
};

VioNode::VioNode() :
      vo_(NULL),
      cam_(NULL)
{
    imu_=(double *)calloc(static_cast<std::size_t>(3), sizeof(double));
    cmd_=(double *)calloc(static_cast<std::size_t>(3), sizeof(double));
    if(!vk::camera_loader::loadFromRosNs("vio", cam_))
        throw std::runtime_error("Camera model not correctly specified.");
    Eigen::Matrix<double,3,1> init;
    init<<1e-19,1e-19,1e-19;
    vo_ = new vio::FrameHandlerMono(cam_,init);
    usleep(500);
}

VioNode::~VioNode()
{
    start_=false;
    usleep(5000);
    imu_the_->join();
    delete imu_the_;
    delete vo_;
    delete cam_;
}

void VioNode::imgCb(const sensor_msgs::ImageConstPtr& msg)
{
    if(!start_)return;
      try {
          auto start=std::chrono::steady_clock::now();
          cv::Mat img=cv_bridge::toCvShare(msg, "mono8")->image;
          vo_->addImage(img, msg->header.stamp.toSec(),msg->header.stamp);
      } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
      }
}
void VioNode::imuCb(const sensor_msgs::ImuPtr &imu) {
    if(!start_)return;
    double imu_in[3];
    if(abs(imu->linear_acceleration.x)>5.0){
        return;
    }else{
        imu_in[0]=imu->linear_acceleration.x;
    };
    if(abs(imu->linear_acceleration.y)>5.0){
        return;
    }else{
        imu_in[1]=imu->linear_acceleration.y;
    };
    imu_in[0] = 0.5*imu_[0]+0.5*imu->linear_acceleration.x;
    imu_in[1] = 0.5*imu_[1]+0.5*imu->linear_acceleration.y;
    imu_in[2] = 0.5*imu_[2]+0.5*imu->angular_velocity.z;
    memcpy(imu_, imu_in, static_cast<std::size_t>(3*sizeof(double)));
    vo_->UpdateIMU(imu_in,imu->header.stamp);
    imu_time_=imu->header.stamp;
#if VIO_DEBUG
    visualizer_.publishMinimal(vo_->ukfPtr_, imu->header.stamp.toSec());
#endif
}
void VioNode::cmdCb(const geometry_msgs::TwistPtr &cmd) {
    if(!start_)return;
    double _cmd[3]={cmd->linear.x,cmd->linear.y,cmd->angular.z};
    memcpy(cmd_, _cmd, static_cast<std::size_t>(3*sizeof(double)));
    vo_->UpdateCmd(_cmd,imu_time_);
}
void VioNode::imu_th(){
    ros::NodeHandle nh;
    ros::Subscriber imu_sub=nh.subscribe(vk::getParam<std::string>("vio/imu_topic", "imu/raw"),10,&VioNode::imuCb, this);
    ros::Subscriber cmd_sub=nh.subscribe(vk::getParam<std::string>("vio/cmd_topic", "cmd/raw"),10,&VioNode::cmdCb, this);
    while(start_ && !boost::this_thread::interruption_requested())
    {
        ros::spinOnce();
        usleep(1000);
    }

}
bool VioNode::getOdom(vio::getOdom::Request &req, vio::getOdom::Response &res) {
    if(req.get==1){
        res.header.stamp = ros::Time();
        res.header.frame_id = "/world";
        res.header.seq = trace_id_;
        auto odom=vo_->ukfPtr_.get_location();
        res.x=odom.second.translation()(0);
        res.y=odom.second.translation()(1);
        res.yaw=odom.second.pitch();
        res.cov={odom.first(0,0),odom.first(0,1),odom.first(0,2),
                 odom.first(1,0),odom.first(1,1),odom.first(1,2),
                 odom.first(2,0),odom.first(2,1),odom.first(2,2)};
    }
    return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vio");
  ros::NodeHandle nh;
  VioNode vio;
  ros::ServiceServer start = nh.advertiseService(vk::getParam<std::string>("vio/start", "start"),
                                                 &VioNode::start, &vio);
  ros::ServiceServer stop = nh.advertiseService(vk::getParam<std::string>("vio/stop", "stop"),
                                                &VioNode::stop, &vio);
  ros::ServiceServer getOdom = nh.advertiseService(vk::getParam<std::string>("vio/odom", "get_odom"),
                                                   &VioNode::getOdom, &vio);
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber it_sub = it.subscribe(vk::getParam<std::string>("vio/cam_topic", "camera/image_raw"), 1, &VioNode::imgCb, &vio);
  // start processing callbacks
  while(ros::ok())
  {
      ros::spinOnce();
  }
  printf("SVO terminated.\n");
  return 0;
}
