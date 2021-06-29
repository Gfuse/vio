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
    Base(const Eigen::Matrix<double, 6, 1> &init) {
        state_=init;
        cov_.setIdentity();
        cov_*=0.0001;
    };
    virtual ~Base(){
    };
    void predict(double ddx,double ddz,double dpitch,double time){
        if(t_==0.0){
            t_=time;
            return;
        }
        double t=time-t_;
        t_=time;
        state_h_<<state_(0)+t*state_(2)+0.5*pow(t,2)*(ddx*cos(state_(4))-ddz*sin(state_(4))),
                state_(1)+t*state_(3)+0.5*pow(t,2)*(ddz*cos(state_(4))+ddx*sin(state_(4))),
                state_(2)+t*(ddx*cos(state_(4))-ddz*sin(state_(4))),
                state_(3)+t*(ddz*cos(state_(4))+ddx*sin(state_(4))),
                state_(4)+0.5*t*(state_(5)+dpitch),
                dpitch;
        Eigen::Matrix<double,6,6> G;
        G<<1.0,0,t,0,-0.5*pow(t,2)*(ddz*cos(state_(4))+ddx*sin(state_(4))),0,
           0,1.0,0,t,0.5*pow(t,2)*(ddx*cos(state_(4))-ddz*sin(state_(4))),0,
           0,0,1.0,0,-t*(ddz*cos(state_(4))+ddx*sin(state_(4))),0,
           0,0,0,1.0,t*(ddx*cos(state_(4))-ddz*sin(state_(4))),0,
           0,0,0,0,1.0,t*dpitch,
           0,0,0,0,0,1.0;
        Eigen::Matrix<double,6,6> R;
        R<<10.0*svo::Config::ACC_Noise(),0,0,0,0,0,
           0,10.0*svo::Config::ACC_Noise(),0,0,0,0,
           0,0,svo::Config::ACC_Noise(),0,0,0,
           0,0,0,svo::Config::ACC_Noise(),0,0,
           0,0,0,0,10.0*svo::Config::GYO_Noise(),0,
           0,0,0,0,0,svo::Config::GYO_Noise();
        cov_h_=G*cov_*G.transpose()+R;
        predict_up=true;
    }
    void correct(double x, double z, double pitch, double time){
        Eigen::Matrix<double,3,6> H;
        Eigen::Matrix<double,3,3> Q;
        Eigen::Matrix<double,3,1> E;
        Eigen::Matrix<double,6,3> k;
        t_=time;
        H<<0,0,1.0,0,0,0,
           0,0,0,1.0,0,0,
           0,0,0,0,0,1.0;
        Q<<svo::Config::Cmd_Cov(),0,0,
           0,svo::Config::Cmd_Cov(),0,
           0,0,svo::Config::Cmd_Cov();
        E<<x-state_h_(2),z-state_h_(3),pitch-state_h_(5);
        k=cov_h_*H.transpose()*(H*cov_h_*H.transpose()+Q).inverse();
        state_=state_h_+k*E;
        cov_=(Eigen::MatrixXd::Identity(6,6)-k*H)*cov_h_;
        predict_up=false;
    }
    void correct(double x, double z,double pitch,size_t match,double time){
        Eigen::Matrix<double,3,6> H;
        Eigen::Matrix<double,3,3> Q;
        Eigen::Matrix<double,3,1> E;
        Eigen::Matrix<double,6,3> k;
        t_=time;
        ++match;
        H<<1.0,0,0,0,0,0,
           0,1.0,0,0,0,0,
           0,0,0,0,1.0,0;
        Q<<svo::Config::Svo_Ekf()/match,0,0,
           0,svo::Config::Svo_Ekf()/match,0,
           0,0,10*svo::Config::Svo_Ekf()/match;
        E<<x-state_h_(0),z-state_h_(1),pitch-state_h_(4);
        k=cov_h_*H.transpose()*(H*cov_h_*H.transpose()+Q).inverse();
        state_=state_h_+k*E;
        cov_=(Eigen::MatrixXd::Identity(6,6)-k*H)*cov_h_;
        predict_up=false;
    }
    Eigen::Matrix<double,6,6> cov_;
    Eigen::Matrix<double, 6, 1> state_;
    bool predict_up=false;
private:
    double t_=0.0;
    Eigen::Matrix<double,6,6> cov_h_;
    Eigen::Matrix<double, 6, 1> state_h_;//state{x,z,dx,dv,pitch}

};

class UKF {
public:
    UKF(const Eigen::Matrix<double, 6, 1>& init) : filter_(new Base(init)) {
    };
    virtual ~UKF() {
        delete filter_;
    };
    void UpdateIMU(double x,double z,double pitch,const ros::Time& time) {
        while(lock)usleep(5);
        lock=true;
        //if(abs(x)<1.0)x=pow(x,3);//picked up from your code
        //if(abs(z)<1.0)z=pow(z,3);//picked up from your code
        if(!filter_->predict_up)filter_->predict(x,z,pitch,1e-9*time.toNSec());
        lock=false;
    };
    void UpdateCmd(double x,double z,double pitch,const ros::Time& time) {
        while(lock)usleep(5);
        lock=true;
        filter_->correct(x,z,pitch,1e-9*time.toNSec());
        lock=false;
    };
    std::pair<Eigen::Matrix<double,3,3>,svo::SE2> UpdateSvo(double x,double z,double pitch,size_t match,ros::Time& time) {
        while(lock)usleep(5);
        lock=true;
        filter_->correct(-1.0*x,-1.0*z,pitch,match,1e-9*time.toNSec());
        lock=false;
        //std::cout<<"filter cov\n"<<filter_->cov_<<'\n';
        //std::cout<<"filter state\n"<<filter_->state_<<'\n';
        Eigen::Matrix<double,3,3> cov;
        cov<<filter_->cov_(0,0),filter_->cov_(0,1),filter_->cov_(0,4),
             filter_->cov_(1,0),filter_->cov_(1,1),filter_->cov_(1,4),
             filter_->cov_(4,0),filter_->cov_(4,1),filter_->cov_(4,4);
        svo::SE2 out(filter_->state_(4),Eigen::Vector2d(-1.0*filter_->state_(0),-1.0*filter_->state_(1)));
        return std::pair<Eigen::Matrix<double,3,3>,svo::SE2>(cov,out);
    };
private:
     Base* filter_= nullptr;
     bool lock=false;
};
#endif //GPU_SVO_UKF_H
