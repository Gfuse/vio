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
#include <gpu_svo/frame_handler_mono.h>
#include <gpu_svo/visualizer.h>
#include <gpu_svo/params_helper.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <image_transport/image_transport.h>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <gpu_svo/abstract_camera.h>
#include <gpu_svo/camera_loader.h>
#include <gpu_svo/frame.h>

class VoNode
{
public:
      svo::FrameHandlerMono* vo_;
      svo::Visualizer visualizer_;
      bool publish_markers_;                 //!< publish only the minimal amount of info (choice for embedded devices)
      vk::AbstractCamera* cam_;
      bool quit_;
      boost::thread* imu_th;
      VoNode();
      ~VoNode();
      void imgCb(const sensor_msgs::ImageConstPtr& msg);
      void imuCb(const sensor_msgs::ImuPtr& imu);
    void cmdCb(const geometry_msgs::TwistPtr &imu);
    void imu();
private:
    double* imu_;
#if VIO_DEBUG
    FILE* log_= nullptr;
    double imu_t_[6]={0.0,0.0,0.0,0.0,0.0,0.0};
#endif
    double* cmd_;
    ros::Time imu_time_;
};

VoNode::VoNode() :
      vo_(NULL),
      publish_markers_(vk::getParam<bool>("vio/publish_markers", true)),
      cam_(NULL),
      quit_(false)
{
      // Create Camera
      imu_=(double *)calloc(static_cast<std::size_t>(3), sizeof(double));
      cmd_=(double *)calloc(static_cast<std::size_t>(3), sizeof(double));
      if(!vk::camera_loader::loadFromRosNs("vio", cam_))
        throw std::runtime_error("Camera model not correctly specified.");
      // Get initial position and orientation
      visualizer_.T_world_from_vision_ = Sophus::SE3(
          vk::rpy2dcm(Eigen::Vector3d(vk::getParam<double>("vio/init_rx", 0.0),
                               vk::getParam<double>("vio/init_ry", 0.0),
                               vk::getParam<double>("vio/init_rz", 0.0))),
          Eigen::Vector3d(vk::getParam<double>("vio/init_tx", 0.0),
                          vk::getParam<double>("vio/init_ty", 0.0),
                          vk::getParam<double>("vio/init_tz", 0.0)));
    Eigen::Matrix<double,6,1> init;
    init<<1e-5,1e-5,1e-5,1e-5,1e-5,1e-5;
    vo_ = new svo::FrameHandlerMono(cam_,init);
    usleep(500);

#if VIO_DEBUG
log_ = fopen((std::string(getenv("HOME"))+"/imu_integration.txt").c_str(), "w+");
#endif
    vo_->start();
    imu_th=new boost::thread(&VoNode::imu,this);

}

VoNode::~VoNode()
{
    quit_=true;


#if VIO_DEBUG
    fclose(log_);
    delete log_;
#endif
    imu_th->join();
    delete imu_th;
    delete vo_;
    delete cam_;
}

void VoNode::imgCb(const sensor_msgs::ImageConstPtr& msg)
{
      try {
          auto start=std::chrono::steady_clock::now();
          cv::Mat img=cv_bridge::toCvShare(msg, "mono8")->image;
          vo_->addImage(img, msg->header.stamp.toSec(),msg->header.stamp);
          visualizer_.publishMinimal(img, vo_->lastFrame(), *vo_, msg->header.stamp.toSec());
          if(vo_->stage() == svo::FrameHandlerMono::STAGE_PAUSED)
              usleep(100000);
#if VIO_DEBUG
    fprintf(log_, ",,,,,,,%s,%f,%f,%f\n",std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()).c_str(),vo_->lastFrame()->T_f_w_.translation()(0),
            vo_->lastFrame()->T_f_w_.translation()(1),
            acos(vo_->lastFrame()->T_f_w_.getSE2().rotation_matrix()(0,0)));
#endif

      } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
      }
}
void VoNode::imuCb(const sensor_msgs::ImuPtr &imu) {
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
#if VIO_DEBUG
        if(imu_t_[5]!=0.0){
            double dt=1e-9*(imu->header.stamp.toNSec()-imu_t_[5]);
            imu_t_[0]+=imu_in[0]*dt;
            imu_t_[1]+=imu_t_[0]*dt;
            imu_t_[2]+=imu_in[1]*dt;
            imu_t_[3]+=imu_t_[2]*dt;
            imu_t_[4]+=imu_in[2]*dt;
            imu_t_[5]=imu->header.stamp.toNSec();
            fprintf(log_, "%f,%f,%f,%f\n",imu_t_[1],imu_t_[3],imu_t_[4],dt);
        }else{
            imu_t_[0]=imu_in[0];
            imu_t_[1]=0.0;
            imu_t_[2]=imu_in[1];
            imu_t_[3]=0.0;
            imu_t_[4]=imu_in[2];
            imu_t_[5]=imu->header.stamp.toNSec();
        }
#endif
    vo_->UpdateIMU(imu_in,imu->header.stamp);
}
void VoNode::cmdCb(const geometry_msgs::TwistPtr &cmd) {
        double _cmd[3]={cmd->linear.x,cmd->linear.y,cmd->angular.z};
        memcpy(cmd_, _cmd, static_cast<std::size_t>(3*sizeof(double)));
#if VIO_DEBUG
        fprintf(log_, ",,,,%f,%f,%f\n",_cmd[0],_cmd[1],_cmd[2]);
#endif
    vo_->UpdateCmd(_cmd,imu_time_);
}
void VoNode::imu(){
    ros::NodeHandle nh;
    // subscribe to cam msgs
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber it_sub = it.subscribe(vk::getParam<std::string>("vio/cam_topic", "camera/image_raw"), 1, &VoNode::imgCb, this);
    // start processing callbacks
    while(ros::ok() && !quit_)
    {
        ros::spinOnce();
    }

}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vio");
  VoNode vo_node;
  ros::NodeHandle nh;
  ros::Subscriber imu_sub=nh.subscribe(vk::getParam<std::string>("vio/imu_topic", "imu/raw"),10,&VoNode::imuCb, &vo_node);
  ros::Subscriber cmd_sub=nh.subscribe(vk::getParam<std::string>("vio/cmd_topic", "cmd/raw"),10,&VoNode::cmdCb, &vo_node);
  // start processing callbacks
  while(ros::ok() && !vo_node.quit_)
  {
      ros::spinOnce();
      usleep(1000);
  }
  printf("SVO terminated.\n");
  return 0;
}
