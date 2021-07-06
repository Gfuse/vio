//
// Created by root on 6/8/21.
//
#ifndef GPU_SVO_UKF_H
#define GPU_SVO_UKF_H

#include <iostream>
#include <fstream>
#include "gpu_svo/config.h"
#include <Eigen/Dense>
class Base{
public:
    Base(const Eigen::Matrix<double, 3, 1> &init) {
        state_=init;
        cov_.setIdentity();
        cov_*=0.0001;
    };
    virtual ~Base(){
    };
    void predict(double dx,double dz,double dpitch,double time){
        if(t_==0.0){
            t_=time;
            return;
        }
        double dt=time-t_;
        t_=time;
        state_h_(2)=state_(2)+dt*dpitch;
        state_h_(1)=state_(1)+dz*cos(state_(2))-dx*sin(state_(2));
        state_h_(0)=state_(0)+dx*cos(state_(2))+dz*sin(state_(2));
        Eigen::Matrix<double,3,3> R;
        R<<svo::Config::Cmd_Cov(),0,0,
           0,svo::Config::Cmd_Cov(),0,
           0,0,svo::Config::Cmd_Cov();
        Eigen::Matrix<double,3,3> G;
        G<<1.0,0,-dx*sin(state_(2))+dz*cos(state_(2)),
           0,1.0,-dz*sin(state_(2))-dx*cos(state_(2)),
           0,0,1.0;
        cov_h_=G*cov_*G.transpose()+R;
        predict_up=true;
    }
    void correct(double ddx, double ddz, double dpitch, double time){
        Eigen::Matrix<double,3,3> H;
        Eigen::Matrix<double,3,3> Q;
        Eigen::Matrix<double,3,1> E;
        Eigen::Matrix<double,3,3> k;
        double dt=time-t_;
        t_=time;
        H<<1.0,0,0.5*pow(dt,2)*(ddz*cos(state_(2))-ddx*sin(state_(2))),
           0,1.0,-0.5*pow(dt,2)*(ddx*cos(state_(2))+ddz*sin(state_(2))),
           0,0,1.0;
        Q<<svo::Config::ACC_Noise(),0,0,
           0,svo::Config::ACC_Noise(),0,
           0,0,svo::Config::GYO_Noise();
        E(0)=state_(0)+0.5*pow(dt,2)*(ddx*cos(state_(2))+ddz*sin(state_(2)))-state_h_(0);
        E(1)=state_(1)+0.5*pow(dt,2)*(ddz*cos(state_(2))-ddx*sin(state_(2)))-state_h_(1);
        E(2)=state_(2)+dt*dpitch-state_h_(2);
        k=cov_h_*H.transpose()*(H*cov_h_*H.transpose()+Q).inverse();
        state_=state_h_+k*E;
        if(fabs(state_(2)-yaw_t_1_)>M_PI_4)state_(2)-=M_PI;
        if(fabs(state_(2))>2*M_PI)state_(2)-=2.0*int(0.5*state_(2)/M_PI)*M_PI;
        cov_=(Eigen::MatrixXd::Identity(3,3)-k*H)*cov_h_;
        state_h_(2)=state_(2)+dt*dpitch;
        state_h_(1)=state_(1)+0.5*pow(dt,2)*(ddz*cos(state_(2))-ddx*sin(state_(2)));
        state_h_(0)=state_(0)+0.5*pow(dt,2)*(ddx*cos(state_(2))+ddz*sin(state_(2)));
        Eigen::Matrix<double,3,3> R;
        R<<svo::Config::ACC_Noise(),0,0,
                0,svo::Config::ACC_Noise(),0,
                0,0,svo::Config::GYO_Noise();
        Eigen::Matrix<double,3,3> G;
        G<<1.0,0,0.5*pow(dt,2)*(ddz*cos(state_(2))-ddx*sin(state_(2))),
                0,1.0,-0.5*pow(dt,2)*(ddx*cos(state_(2))+ddz*sin(state_(2))),
                0,0,1.0;
        cov_h_=G*cov_*G.transpose()+R;
        yaw_t_1_=state_(2);

    }
    void correct(double x, double z,double pitch,size_t match,double time){
        Eigen::Matrix<double,3,3> H;
        Eigen::Matrix<double,3,3> Q;
        Eigen::Matrix<double,3,1> E;
        Eigen::Matrix<double,3,3> k;
        t_=time;
        ++match;
        H<<1.0,0,0,
           0,1.0,0,
           0,0,1.0;
        Q<<svo::Config::Svo_Ekf(),0,0,
           0,svo::Config::Svo_Ekf(),0,
           0,0,svo::Config::Svo_Ekf();
        E<<x-state_h_(0),z-state_h_(1),pitch-state_h_(2);
        k=cov_h_*H.transpose()*(H*cov_h_*H.transpose()+Q).inverse();
        state_=state_h_+(k*E);
        if(fabs(state_(2)-yaw_t_1_)>M_PI_4)state_(2)-=M_PI;
        if(fabs(state_(2))>2*M_PI)state_(2)-=2.0*int(0.5*state_(2)/M_PI)*M_PI;
        cov_=(Eigen::MatrixXd::Identity(3,3)-k*H)*cov_h_;
        yaw_t_1_=state_(2);
        predict_up=false;
    }
    Eigen::Matrix<double,3,3> cov_;
    Eigen::Matrix<double, 3, 1> state_;
    bool predict_up=false;
private:
    double t_=0.0;
    Eigen::Matrix<double,3,3> cov_h_;
    Eigen::Matrix<double, 3, 1> state_h_;//state{x,z,pitch}
    double yaw_t_1_=0.0;

};

class UKF {
public:
    UKF(const Eigen::Matrix<double, 3, 1>& init) : filter_(new Base(init)) {
    };
    virtual ~UKF() {
        delete filter_;
    };
    void UpdateIMU(double x,double z,double pitch,const ros::Time& time) {
        while(lock)usleep(5);
        lock=true;
        if(abs(x)<1.0)x=pow(x,3);//picked up from your code
        if(abs(z)<1.0)z=pow(z,3);//picked up from your code
        if(filter_->predict_up)filter_->correct(x,z,pitch,1e-9*time.toNSec());
        lock=false;
    };
    void UpdateCmd(double x,double z,double pitch,const ros::Time& time) {
        while(lock)usleep(5);
        lock=true;
        filter_->predict(x,z,pitch,1e-9*time.toNSec());
        lock=false;
    };
    std::pair<Eigen::Matrix<double,3,3>,svo::SE2_5> UpdateSvo(double x,double z,double pitch,size_t match,ros::Time& time) {
        while(lock)usleep(5);
        lock=true;
        if(filter_->predict_up)filter_->correct(x,z,pitch,match,1e-9*time.toNSec());
        lock=false;
        return std::pair<Eigen::Matrix<double,3,3>,svo::SE2_5>(filter_->cov_,svo::SE2_5(filter_->state_(0),filter_->state_(1),filter_->state_(2)));
    };
private:
     Base* filter_= nullptr;
     bool lock=false;
};
#endif //GPU_SVO_UKF_H
