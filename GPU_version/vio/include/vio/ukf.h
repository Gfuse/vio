//
// Created by root on 6/8/21.
//
#ifndef VIO_UKF_H
#define VIO_UKF_H

#include <iostream>
#include <fstream>
#include "vio/config.h"
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
        state_h_(2)=state_(2)+dt_angular*dpitch;
        state_h_(1)=state_(1)+dt_liner*(dz*cos(state_(2))-dx*sin(state_(2)));
        state_h_(0)=state_(0)+dt_liner*(dx*cos(state_(2))+dz*sin(state_(2)));
        Eigen::Matrix<double,3,3> R;
        R<<vio::Config::Cmd_Cov(),0,0,
           0,vio::Config::Cmd_Cov(),0,
           0,0,vio::Config::Cmd_Cov();
        Eigen::Matrix<double,3,3> G;
        G<<1.0,0,dt_liner*(-dx*sin(state_(2))+dz*cos(state_(2))),
           0,1.0,dt_liner*(-dz*sin(state_(2))-dx*cos(state_(2))),
           0,0,1.0;
        cov_h_=G*cov_*G.transpose()+R;
        predict_up=true;
    }
    void correct(double ddx, double ddz, double dpitch, double time){
        Eigen::Matrix<double,3,3> H;
        Eigen::Matrix<double,3,3> Q;
        Eigen::Matrix<double,3,1> E;
        Eigen::Matrix<double,3,3> k;
        H<<1.0,0,pow(0.03*dt_liner,2)*(ddz*cos(state_(2))-ddx*sin(state_(2))),
           0,1.0,-pow(0.03*dt_liner,2)*(ddx*cos(state_(2))+ddz*sin(state_(2))),
           0,0,1.0;
        Q<<vio::Config::ACC_Noise(),0,0,
           0,vio::Config::ACC_Noise(),0,
           0,0,vio::Config::GYO_Noise();
        E(0)=state_(0)+pow(0.03*dt_liner,2)*(ddx*cos(state_(2))+ddz*sin(state_(2)))-state_h_(0);
        E(1)=state_(1)+pow(0.03*dt_liner,2)*(ddz*cos(state_(2))-ddx*sin(state_(2)))-state_h_(1);
        E(2)=state_(2)+0.03*dt_angular*dpitch-state_h_(2);
        k=cov_h_*H.transpose()*(H*cov_h_*H.transpose()+Q).inverse();
        state_=state_h_+k*E;
        cov_=(Eigen::MatrixXd::Identity(3,3)-k*H)*cov_h_;
        state_h_(2)=state_(2)+0.03*dt_angular*dpitch;
        state_h_(1)=state_(1)+pow(0.03*dt_liner,2)*(ddz*cos(state_(2))-ddx*sin(state_(2)));
        state_h_(0)=state_(0)+pow(0.03*dt_liner,2)*(ddx*cos(state_(2))+ddz*sin(state_(2)));
        Eigen::Matrix<double,3,3> R;
        R<<vio::Config::ACC_Noise(),0,0,
                0,vio::Config::ACC_Noise(),0,
                0,0,vio::Config::GYO_Noise();
        Eigen::Matrix<double,3,3> G;
        G<<1.0,0,pow(0.03*dt_liner,2)*(ddz*cos(state_(2))-ddx*sin(state_(2))),
                0,1.0,-pow(0.03*dt_liner,2)*(ddx*cos(state_(2))+ddz*sin(state_(2))),
                0,0,1.0;
        cov_h_=G*cov_*G.transpose()+R;

    }
    void correct(double x, double z,double pitch,size_t match,double time){
        Eigen::Matrix<double,3,3> H;
        Eigen::Matrix<double,3,3> Q;
        Eigen::Matrix<double,3,1> E;
        Eigen::Matrix<double,3,3> k;
        ++match;
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
        predict_up=false;
    }
    Eigen::Matrix<double,3,3> cov_;
    Eigen::Matrix<double, 3, 1> state_;
    bool predict_up=false;
private:
    Eigen::Matrix<double,3,3> cov_h_;
    Eigen::Matrix<double, 3, 1> state_h_;//state{x,z,pitch}
    double dt_liner=0.2;
    double dt_angular=0.2;

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
    std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5> UpdateSvo(double x,double z,double pitch,size_t match,ros::Time& time) {
        while(lock)usleep(5);
        lock=true;
        pitch=pitch+M_PI;
        if(filter_->predict_up)filter_->correct(-z,-x,pitch,match,1e-9*time.toNSec());
        lock=false;
        double pitch_f;
        pitch_f=filter_->state_(2)+M_PI;
        return std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5>(filter_->cov_,vio::SE2_5(-filter_->state_(1),-filter_->state_(0),pitch_f));
    };
    std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5> get_location(){
        double pitch_f;
        pitch_f=filter_->state_(2)+M_PI;
        return std::pair<Eigen::Matrix<double,3,3>,vio::SE2_5>(filter_->cov_,vio::SE2_5(-filter_->state_(1),-filter_->state_(0),pitch_f));
    }
private:
     Base* filter_= nullptr;
     bool lock=false;
};
#endif //VIO_UKF_H
