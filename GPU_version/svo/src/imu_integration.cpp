//
// Created by gfuse on 4/9/21.
//
//#define NDEBUG
#include "gpu_svo/imu_integration.h"
#include <memory>
#include <utility>
#include "gpu_svo/feature.h"
#include "gpu_svo/point.h"
//#include "gpu_svo/svoVisionFactorSmall.h"
#include "gpu_svo/svoVisionFactor.h"
//#include "gpu_svo/svoVisionFactorLandMark.h"
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Pose3.h>
#include <vikit/params_helper.h>
#include <gpu_svo/config.h>
#include <gpu_svo/for_it.hpp>
#define debug 0
Imu_Integration::Imu_Integration(Sophus::SE3& SE_init){
    graphPtr=std::make_shared<gtsam::NonlinearFactorGraph>();
    valuesPtr=std::make_shared<gtsam::Values>();
    parameterPtr = boost::shared_ptr<gtsam::PreintegrationCombinedParams>(new gtsam::PreintegrationCombinedParams(gtsam::Vector3(0.0,0.0,0.0)));
    parameterPtr->accelerometerCovariance=gtsam::I_3x3 * svo::Config::ACC_Noise(); // acc white noise in continuous
    parameterPtr->gyroscopeCovariance=gtsam::I_3x3 * svo::Config::GYO_Noise();// gyro white noise in continuous
    parameterPtr->integrationCovariance=gtsam::I_3x3 * svo::Config::IUC(); // integration uncertainty continuous
    parameterPtr->biasAccCovariance = gtsam::I_3x3 * svo::Config::ABC();   // acc bias in continuous
    parameterPtr->biasOmegaCovariance = gtsam::I_3x3 * svo::Config::GBC();  // gyro bias in continuous
    parameterPtr->biasAccOmegaInt = gtsam::I_6x6 * svo::Config::EBP();  // error in the bias used for preintegration
    imu_biasPtr=std::make_shared<gtsam::imuBias::ConstantBias>(gtsam::Vector3(1e-8,1e-8,1e-8),gtsam::Vector3(1e-8,1e-8,1e-8));
    preintegratedPtr = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(parameterPtr,*imu_biasPtr);
    ProjectNoisePtr = gtsam::noiseModel::Isotropic::Sigma(2, svo::Config::NRP()); // the noise used for projection
    optimizerParamPtr=std::make_shared<gtsam::ISAM2Params>();
    optimizerParamPtr->relinearizeThreshold = 0.1;
    optimizerParamPtr->relinearizeSkip = 1;
    optimizerParamPtr->evaluateNonlinearError=true;
    optimizerParamPtr->enableDetailedResults=true;
    if(debug)logtime = fopen((std::string(getenv("HOME"))+"/imu_preIntegrate.txt").c_str(), "w+");
    assert(preintegratedPtr);
}
Imu_Integration::~Imu_Integration(){

}
bool Imu_Integration::update(double* imu){
    if(syn)return false;
    if(imu_factor_id<1)return false;
    if(abs(imu[0])<1.0)imu[0]=pow(imu[0],3);//picked up from your code
    if(abs(imu[1])<1.0)imu[0]=pow(imu[1],3);//picked up from your code
    auto in=std::chrono::steady_clock::now();
    if(std::chrono::duration<double>(in-t_1).count()>20.0){
        t_1=in;
        preintegratedPtr->integrateMeasurement(Eigen::Vector3d(-1.0*(imu[0]-svo::Config::X_off()),0.0,-1.0*(imu[1]-svo::Config::Y_off())),Eigen::Vector3d(0.0,imu[5]-svo::Config::Teta_off(),0.0),0.0005);
    }else{
        preintegratedPtr->integrateMeasurement(Eigen::Vector3d(-1.0*(imu[0]-svo::Config::X_off()),0.0,-1.0*(imu[1]-svo::Config::Y_off())),Eigen::Vector3d(0.0,imu[5]-svo::Config::Teta_off(),0.0),std::chrono::duration<double>(in-t_1).count());
        t_1=in;
        ++imu_n;
    }
    return true;

}
bool Imu_Integration::reset(gtsam::LevenbergMarquardtOptimizer& optimizer,boost::shared_ptr<svo::Frame>& new_frame){
    gtsam::Values result = optimizer.optimize();
    statePtr=std::make_shared<gtsam::NavState>(result.at<gtsam::Pose3>(P(imu_factor_id)), result.at<gtsam::Vector3>(V(imu_factor_id)));
    imu_biasPtr=std::make_shared<gtsam::imuBias::ConstantBias>(result.at<gtsam::imuBias::ConstantBias>(B(imu_factor_id)));
    preintegratedPtr->resetIntegrationAndSetBias(*imu_biasPtr);
    ++imu_factor_id;
    syn=false;
    gtsam::Marginals marginals(*graphPtr,result);
    if(debug)fprintf(logtime, "%d,%f,%f,%f,%f,\n",
                     imu_factor_id-1,
                     statePtr->pose().x(),
                     statePtr->pose().y(),
                     statePtr->pose().z(),
                     statePtr->pose().rotation().pitch());
    new_frame->T_f_w_=svo::SE2_5(statePtr->pose().x(),statePtr->pose().z(),statePtr->pose().rotation().pitch());
    new_frame->Cov_=marginals.marginalCovariance(P(imu_factor_id-1));
    graphPtr->resize(0);
    valuesPtr->clear();
    valuesPtr->insert(P(imu_factor_id-1),gtsam::Pose3(new_frame->T_f_w_.T->matrix()));
    valuesPtr->insert(V(imu_factor_id-1),statePtr->v());
    valuesPtr->insert(B(imu_factor_id-1),*imu_biasPtr);
    graphPtr->addPrior(P(imu_factor_id-1),gtsam::Pose3(new_frame->T_f_w_.T->matrix()),marginals.marginalCovariance(P(imu_factor_id-1)));
    graphPtr->addPrior(V(imu_factor_id-1),statePtr->v(),marginals.marginalCovariance(V(imu_factor_id-1)));
    graphPtr->addPrior(B(imu_factor_id-1),*imu_biasPtr,marginals.marginalCovariance(B(imu_factor_id-1)));
    return true;
}
bool Imu_Integration::preintegrate_predict(svo::SE2_5& frame) {
    if(imu_n<1){
        syn= false;
        return false;
    }else{
        syn=true;
    }
    Integ_predict=preintegratedPtr->predict(*statePtr,*imu_biasPtr);
    frame =svo::SE2_5(Integ_predict.pose().x(),Integ_predict.pose().z(),Integ_predict.pose().rotation().pitch());
    return true;
}
void Imu_Integration::reset() {
    preintegratedPtr->resetIntegrationAndSetBias(*imu_biasPtr);
    imu_n=0;
    syn=false;
}
bool Imu_Integration::predict(boost::shared_ptr<svo::Frame>& new_frame,
                              boost::shared_ptr<svo::Frame>& last_frame,
                              const double cell_match){
    valuesPtr->insert(P(imu_factor_id), Integ_predict.pose());
    valuesPtr->insert(V(imu_factor_id), Integ_predict.v());
    valuesPtr->insert(B(imu_factor_id), *imu_biasPtr);
    gtsam::CombinedImuFactor imu_factor(P(imu_factor_id - 1), V(imu_factor_id - 1), P(imu_factor_id), V(imu_factor_id), B(imu_factor_id - 1), B(imu_factor_id),
                                        *preintegratedPtr);
    graphPtr->emplace_shared<gtsam::CombinedImuFactor>(imu_factor);
    imu_n=0;
    graphPtr->emplace_shared<svo::VisionFactor>(gtsam::Pose3(new_frame->T_f_w_.T->matrix()),
                                                gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 1.0, 1.0, 1.0).finished()/(cell_match+1)),
                                                P(imu_factor_id));
    gtsam::LevenbergMarquardtParams params0;
    params0.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;
    params0.setlambdaFactor(10.0);
    params0.setlambdaInitial(100);
    params0.setlambdaUpperBound(1e10);
    gtsam::LevenbergMarquardtOptimizer optimizer(*graphPtr, *valuesPtr,params0);
    reset(optimizer,new_frame);
    return true;
}
bool Imu_Integration::init(boost::shared_ptr<svo::Frame>& new_frame){
    statePtr=std::make_shared<gtsam::NavState>(gtsam::Pose3(new_frame->T_f_w_.T->matrix()),
                                               gtsam::Vector3(1e-9,1e-9,1e-9));
    valuesPtr->insert(P(0),statePtr->pose());
    valuesPtr->insert(V(0),statePtr->v());
    valuesPtr->insert(B(0),*imu_biasPtr);
    graphPtr->addPrior(P(0),statePtr->pose(),gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-5, 1e-15, 1e-5, 1e-15, 1e-8, 1e-15).finished()));
    graphPtr->addPrior(V(0),statePtr->v(),gtsam::noiseModel::Isotropic::Sigma(3, 1e-4));
    graphPtr->addPrior(B(0),*imu_biasPtr,gtsam::noiseModel::Isotropic::Sigma(6, 1e-2));
    ++imu_factor_id;
    return true;
}