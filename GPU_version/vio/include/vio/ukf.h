//
// Created by root on 6/8/21.
//
#ifndef VIO_UKF_H
#define VIO_UKF_H

#include <iostream>
#include <fstream>
#include <vio/config.h>
#include <Eigen/Dense>
#include <boost/thread.hpp>
#include <boost/function.hpp>
class Base{
public:
    Base(const Eigen::Matrix<double, 3, 1> &init) {
        state_=init;
        cov_.setIdentity();
        cov_*=0.0001;
        state_h_=init;
        cov_h_.setIdentity();
        cov_h_*=0.0001;
    };
    virtual ~Base(){
    };
    void predict(double dx,double dy,double dpitch,double time){
        double dt=time-t_cmd;
        if(t_cmd==0.0){
            t_cmd=time;
            return;
        }
        t_cmd=time;
        state_h_(2)=state_(2)+dt*dpitch;
        state_h_(1)=state_(1)+2.0*dt*(dy*cos(state_(2))-dx*sin(state_(2)));
        state_h_(0)=state_(0)+dt*(dx*cos(state_(2))+dy*sin(state_(2)));
        Eigen::Matrix<double,3,3> R;
        R<<vio::Config::Cmd_Cov(),0,0,
           0,vio::Config::Cmd_Cov(),0,
           0,0,vio::Config::Cmd_Cov();
        Eigen::Matrix<double,3,3> G;
        G<<1.0,0,dt*(-dx*sin(state_(2))+dy*cos(state_(2))),
           0,1.0,2.0*dt*(-dy*sin(state_(2))-dx*cos(state_(2))),
           0,0,1.0;
        cov_h_=G*cov_*G.transpose()+R;
    }
    void correct(double ddx, double ddy, double dpitch, double time){
        double dt=time-t_imu;
        if(t_imu==0.0){
            t_imu=time;
            return;
        }
        t_imu=time;
        Eigen::Matrix<double,3,3> H;
        Eigen::Matrix<double,3,3> Q;
        Eigen::Matrix<double,3,1> E;
        Eigen::Matrix<double,3,3> k;
        H<<1.0,0,pow(dt,2)*(ddy*cos(state_(2))-ddx*sin(state_(2))),
           0,1.0,-2.0*pow(dt,2)*(ddx*cos(state_(2))+ddy*sin(state_(2))),
           0,0,1.0;
        Q<<vio::Config::ACC_Noise(),0,0,
           0,vio::Config::ACC_Noise(),0,
           0,0,vio::Config::GYO_Noise();
        E(0)=state_(0)+pow(dt,2)*(ddx*cos(state_(2))+ddy*sin(state_(2)))-state_h_(0);
        E(1)=state_(1)+2.0*pow(dt,2)*(ddy*cos(state_(2))-ddx*sin(state_(2)))-state_h_(1);
        E(2)=state_(2)+dt*dpitch-state_h_(2);
        k=cov_h_*H.transpose()*(H*cov_h_*H.transpose()+Q).inverse();
        state_=state_h_+k*E;
        cov_=(Eigen::MatrixXd::Identity(3,3)-k*H)*cov_h_;
        state_h_(2)=state_(2)+dt*dpitch;
        state_h_(1)=state_(1)+2.0*pow(dt,2)*(ddy*cos(state_(2))-ddx*sin(state_(2)));
        state_h_(0)=state_(0)+pow(dt,2)*(ddx*cos(state_(2))+ddy*sin(state_(2)));
        Eigen::Matrix<double,3,3> R;
        R<<vio::Config::ACC_Noise(),0,0,
                0,vio::Config::ACC_Noise(),0,
                0,0,vio::Config::GYO_Noise();
        Eigen::Matrix<double,3,3> G;
        G<<1.0,0,pow(dt,2)*(ddy*cos(state_(2))-ddx*sin(state_(2))),
                0,1.0,-2.0*pow(dt,2)*(ddx*cos(state_(2))+ddy*sin(state_(2))),
                0,0,1.0;
        cov_h_=G*cov_*G.transpose()+R;

    }
    void correct(double x, double z,double pitch){
        Eigen::Matrix<double,3,3> H;
        Eigen::Matrix<double,3,3> Q;
        Eigen::Matrix<double,3,1> E;
        Eigen::Matrix<double,3,3> k;
        H<<1.0,0,0,
           0,1.0,0,
           0,0,1.0;
        Q<<vio::Config::Svo_Ekf(),0,0,
           0,vio::Config::Svo_Ekf(),0,
           0,0,vio::Config::Svo_Ekf();
        E<<x-state_h_(0),z-state_h_(1),pitch-state_h_(2);
        k=cov_h_*H.transpose()*(H*cov_h_*H.transpose()+Q).inverse();
        state_=state_h_+(k*E);
        cov_=(Eigen::MatrixXd::Identity(3,3)-k*H)*cov_h_;
    }
    Eigen::Matrix<double,3,3> cov_;
    Eigen::Matrix<double, 3, 1> state_;
private:
    double t_cmd=0.0;
    Eigen::Matrix<double,3,3> cov_h_;
    Eigen::Matrix<double, 3, 1> state_h_;//state{x,y,pitch( rotation aound z)} in world frame
    double t_imu=0.0;

};

class UKF {
public:
    UKF(const Eigen::Matrix<double, 3, 1>& init) : filter_(new Base(init)) {
    };
    virtual ~UKF() {
        delete filter_;
    };
    void UpdateIMU(double x/*in imu frame*/,double y/*in imu frame*/,double theta/*in imu frame*/,const ros::Time& time) {
        if(!start_)return;
        if(abs(x)<1.0)x=pow(x,3);//picked up from your code
        if(abs(y)<1.0)y=pow(y,3);//picked up from your code
        boost::unique_lock<boost::mutex> lock(ekf_mut_);
        filter_->correct(x,y,theta,1e-9*time.toNSec());
    };
    //IMU frame y front, x right, z up -> left hands (theta counts from y)
    void UpdateCmd(double x/*in imu frame*/,double y/*in imu frame*/,double theta/*in imu frame*/,const ros::Time& time) {
        if(!start_)return;
        boost::unique_lock<boost::mutex> lock(ekf_mut_);
        filter_->predict(x,y,theta,1e-9*time.toNSec());
    };
    //Camera frame z back, x right, y down -> right hands (pitch counts from x)correct
    //Notice T_F_W is the position of the first frame with respect to the new frame while they are looking at one feature
    std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5> UpdateSvo(double x/*in camera frame*/,double z/*in camera frame*/,double pitch/*in camera frame*/) {
        boost::unique_lock<boost::mutex> lock(ekf_mut_);
        filter_->correct(x,z,-1.0*pitch);
        return std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5>(filter_->cov_,vio::SE2_5(filter_->state_(0),filter_->state_(1),-1.0*(filter_->state_(2))));
    };
    std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5> get_location(){
        boost::unique_lock<boost::mutex> lock(ekf_mut_);
        return std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5>(filter_->cov_,vio::SE2_5(filter_->state_(0),filter_->state_(1),-1.0*(filter_->state_(2))));
    }
    bool start_=false;
private:
     Base* filter_= nullptr;
    boost::mutex ekf_mut_;
};
#endif //VIO_UKF_H
