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
#define debug 1
Imu_Integration::Imu_Integration(Sophus::SE3& SE_init){
    graphPtr=std::make_shared<gtsam::NonlinearFactorGraph>();
    valuesPtr=std::make_shared<gtsam::Values>();
    // Example: pitch and roll of aircraft in an ENU Cartesian frame.
    // If pitch and roll are zero for an aerospace frame,
    // that means Z is pointing down, i.e., direction of Z = (0,0,-1)
    //gtsam::Vector4 g=SE_init.matrix()*gtsam::Vector4(0, 0, svo::Config::Gravity(),1.0);
    //parameterPtr = boost::shared_ptr<gtsam::PreintegrationCombinedParams>(new gtsam::PreintegrationCombinedParams(gtsam::Vector3(g.x(),g.y(),g.z())));
    parameterPtr = boost::shared_ptr<gtsam::PreintegrationCombinedParams>(new gtsam::PreintegrationCombinedParams(gtsam::Vector3(0.0,0.0,0.0)));
    //parameterPtr->setBodyPSensor(gtsam::Pose3(SE_init.matrix()));// The pose of the sensor in the body frame
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
    auto in=std::chrono::steady_clock::now();
    if(std::chrono::duration<double>(in-t_1).count()>20.0){
        t_1=in;
        preintegratedPtr->integrateMeasurement(Eigen::Vector3d(imu[0]-svo::Config::X_off(),0.0,-1.0*imu[1]-svo::Config::Y_off()),Eigen::Vector3d(0.0,imu[5]-svo::Config::Teta_off(),0.0),0.005);
    }else{
        preintegratedPtr->integrateMeasurement(Eigen::Vector3d(imu[0]-svo::Config::X_off(),0.0,-1.0*imu[1]-svo::Config::Y_off()),Eigen::Vector3d(0.0,imu[5]-svo::Config::Teta_off(),0.0),std::chrono::duration<double>(in-t_1).count());
        t_1=in;
        ++imu_n;
    }
    return true;

}
bool Imu_Integration::reset(gtsam::ISAM2& optimizer,boost::shared_ptr<svo::Frame>& new_frame){
    gtsam::Values result = optimizer.calculateEstimate();
    // Overwrite the beginning of the preintegration for the next step.
    statePtr=std::make_shared<gtsam::NavState>(result.at<gtsam::Pose3>(P(imu_factor_id)), result.at<gtsam::Vector3>(V(imu_factor_id)));
    imu_biasPtr=std::make_shared<gtsam::imuBias::ConstantBias>(result.at<gtsam::imuBias::ConstantBias>(B(imu_factor_id)));
    preintegratedPtr->resetIntegrationAndSetBias(*imu_biasPtr);
    ++imu_factor_id;
    syn=false;
    new_frame->T_f_w_=svo::SE2_5(statePtr->t().x(),statePtr->t().z(),atan(statePtr->R()(0,2)/statePtr->R()(0,0)));
    new_frame->Cov_=optimizer.marginalCovariance(P(imu_factor_id-1));
    graphPtr->resize(0);
    valuesPtr->clear();
    valuesPtr->insert(P(imu_factor_id-1),statePtr->pose());
    valuesPtr->insert(V(imu_factor_id-1),statePtr->v());
    valuesPtr->insert(B(imu_factor_id-1),*imu_biasPtr);
    graphPtr->addPrior(P(imu_factor_id-1),statePtr->pose(),optimizer.marginalCovariance(P(imu_factor_id-1)));
    graphPtr->addPrior(V(imu_factor_id-1),statePtr->v(),optimizer.marginalCovariance(V(imu_factor_id-1)));
    graphPtr->addPrior(B(imu_factor_id-1),*imu_biasPtr,optimizer.marginalCovariance(B(imu_factor_id-1)));
    for(auto&& f_n:_for(new_frame->fts_)){
        if(f_n.item->point == NULL)continue;
        if(!result.exists(L(f_n.index)))continue;
        f_n.item->point->pos_=result.at<gtsam::Point3>(L(f_n.index));
    //    valuesPtr->insert(L(f_n.index),f_n.item->point->pos_);
//        graphPtr->emplace_shared<svo::VisionFactor>(f_n.item->f,f_n.item->level,ProjectNoisePtr,P(imu_factor_id),L(f_n.index));
    }
    return true;
}
bool Imu_Integration::reset(gtsam::LevenbergMarquardtOptimizer& optimizer,boost::shared_ptr<svo::Frame>& new_frame){
    gtsam::Values result = optimizer.optimize();
    if(debug)optimizer.print();
    // Overwrite the beginning of the preintegration for the next step.
    statePtr=std::make_shared<gtsam::NavState>(result.at<gtsam::Pose3>(P(imu_factor_id)), result.at<gtsam::Vector3>(V(imu_factor_id)));
    imu_biasPtr=std::make_shared<gtsam::imuBias::ConstantBias>(result.at<gtsam::imuBias::ConstantBias>(B(imu_factor_id)));
    preintegratedPtr->resetIntegrationAndSetBias(*imu_biasPtr);
    ++imu_factor_id;
    syn=false;
    gtsam::Marginals marginals(*graphPtr,result);
    if(debug)fprintf(logtime, "%f,%f,%f,%f,%f,%f,\n",
                     new_frame->T_f_w_.T->translation().x(),
                     new_frame->T_f_w_.T->translation().y(),
                     new_frame->T_f_w_.T->translation().z(),
                     statePtr->pose().x(),
                     statePtr->pose().y(),
                     statePtr->pose().z());
    new_frame->T_f_w_=svo::SE2_5(statePtr->t().x(),statePtr->t().z(),atan(statePtr->R()(0,2)/statePtr->R()(0,0)));
    new_frame->Cov_=marginals.marginalCovariance(P(imu_factor_id-1));
    graphPtr->resize(0);
    valuesPtr->clear();
    valuesPtr->insert(P(imu_factor_id-1),statePtr->pose());
    valuesPtr->insert(V(imu_factor_id-1),statePtr->v());
    valuesPtr->insert(B(imu_factor_id-1),*imu_biasPtr);
    graphPtr->addPrior(P(imu_factor_id-1),statePtr->pose(),marginals.marginalCovariance(P(imu_factor_id-1)));
    graphPtr->addPrior(V(imu_factor_id-1),statePtr->v(),marginals.marginalCovariance(V(imu_factor_id-1)));
    graphPtr->addPrior(B(imu_factor_id-1),*imu_biasPtr,marginals.marginalCovariance(B(imu_factor_id-1)));
    return true;
}
Sophus::SE3 Imu_Integration::preintegrate_predict() {
    if(imu_n<1){
        syn= false;
        usleep(10000);
        syn=true;
    }else{
        syn=true;
    }
    auto preint_imu_combined =dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*preintegratedPtr);
    Integ_predict=preint_imu_combined.predict(*statePtr,*imu_biasPtr);
    return Sophus::SE3(Integ_predict.R(),Integ_predict.t());
}
bool Imu_Integration::predict(boost::shared_ptr<svo::Frame>& new_frame,
                              boost::shared_ptr<svo::Frame>& last_frame,
                              std::size_t& num_obs,const double cell_match){
    if(debug)std::cout<<"----------------------------IMU infuse---------------------------------------\n";
    //if(debug)preintegratedPtr->print();
    auto preint_imu_combined =dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*preintegratedPtr);
    valuesPtr->insert(P(imu_factor_id), Integ_predict.pose());
    valuesPtr->insert(V(imu_factor_id), Integ_predict.v());
    valuesPtr->insert(B(imu_factor_id), *imu_biasPtr);
    gtsam::CombinedImuFactor imu_factor(P(imu_factor_id - 1), V(imu_factor_id - 1), P(imu_factor_id), V(imu_factor_id), B(imu_factor_id - 1), B(imu_factor_id),
                                            preint_imu_combined);
    graphPtr->emplace_shared<gtsam::CombinedImuFactor>(imu_factor);
    imu_n=0;
    graphPtr->emplace_shared<svo::VisionFactor>(gtsam::Pose3(new_frame->T_f_w_.T->matrix()),
                                                gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 1.0, 1.0, 1.0).finished()/(cell_match+1)),
                                                P(imu_factor_id));
    gtsam::LevenbergMarquardtParams params0;
    //params0.verbosity=gtsam::LevenbergMarquardtParams::LINEAR;
    params0.verbosityLM = gtsam::LevenbergMarquardtParams::SUMMARY;
    params0.setlambdaFactor(10.0);
    params0.setlambdaInitial(100);
    params0.setlambdaUpperBound(1e10);
    gtsam::LevenbergMarquardtOptimizer optimizer(*graphPtr, *valuesPtr,params0);
    reset(optimizer,new_frame);

    if(debug)std::cout<<"------------------------------------------------------------------------------\n";
    return true;
}
bool Imu_Integration::init(boost::shared_ptr<svo::Frame>& new_frame){
    statePtr=std::make_shared<gtsam::NavState>(gtsam::Pose3(new_frame->T_f_w_.T->matrix()),
                                               gtsam::Vector3(1e-9,1e-9,1e-9));
    /*
    statePtr=std::make_shared<gtsam::NavState>(gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0,0.0,0.0),gtsam::Point3(0.0,0.0,0.0)),
                                               gtsam::Vector3(1e-9,1e-9,1e-9));
                                               */
    valuesPtr->insert(P(0),statePtr->pose());
    valuesPtr->insert(V(0),statePtr->v());
    valuesPtr->insert(B(0),*imu_biasPtr);
    graphPtr->addPrior(P(0),statePtr->pose(),gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-5, 1e-15, 1e-5, 1e-15, 1e-8, 1e-15).finished()));
    graphPtr->addPrior(V(0),statePtr->v(),gtsam::noiseModel::Isotropic::Sigma(3, 1e-4));
    graphPtr->addPrior(B(0),*imu_biasPtr,gtsam::noiseModel::Isotropic::Sigma(6, 1e-2));
    /*
    for(auto&& f_n:_for(new_frame->fts_)){
        if(f_n.item->point == NULL)continue;
        valuesPtr->insert(L(f_n.index),f_n.item->point->pos_);
        graphPtr->emplace_shared<svo::VisionFactor>(f_n.item->f,f_n.item->level,ProjectNoisePtr,P(imu_factor_id),L(f_n.index));
    }
    graphPtr->addPrior(L(0),new_frame->fts_.front()->point->pos_,gtsam::noiseModel::Isotropic::Sigma(3, 1e-4));
     */
    ++imu_factor_id;
    return true;
}