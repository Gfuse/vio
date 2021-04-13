//
// Created by gfuse on 4/9/21.
//
//#define NDEBUG
#include "svo/imu_integration.h"
#include <memory>
#include <utility>
#include "svo/feature.h"
#include "svo/point.h"

Imu_Integration::Imu_Integration(Sophus::SE3& SE_init){
    statePtr=std::make_shared<gtsam::NavState>(gtsam::Pose3(gtsam::Rot3(SE_init.rotation_matrix()),SE_init.translation()),
                                               gtsam::Vector3(0.0,0.0,0.0));
    imu_biasPtr=std::make_shared<gtsam::imuBias::ConstantBias>();
    valuesPtr=std::make_unique<gtsam::Values>();
    valuesPtr->insert(P(0),statePtr->pose());
    valuesPtr->insert(V(0),statePtr->v());
    valuesPtr->insert(B(0),*imu_biasPtr.get());
    graphPtr=std::make_unique<gtsam::NonlinearFactorGraph>();
    graphPtr->addPrior(P(0),statePtr->pose(),gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.5, 0.5, 0.5).finished()));
    graphPtr->addPrior(V(0),statePtr->v(),gtsam::noiseModel::Isotropic::Sigma(3, 0.1));
    graphPtr->addPrior(B(0),*imu_biasPtr.get(),gtsam::noiseModel::Isotropic::Sigma(6, 1e-3));
    parameterPtr = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD();// TODO check the IMU frame
    parameterPtr->accelerometerCovariance=gtsam::I_3x3 * pow(0.0003924, 2); // acc white noise in continuous
    parameterPtr->gyroscopeCovariance=gtsam::I_3x3 * pow(0.00020568902491,2); // gyro white noise in continuous
    parameterPtr->integrationCovariance=gtsam::I_3x3 * 1e-8; // integration uncertainty continuous
    parameterPtr->biasAccCovariance = gtsam::I_3x3 * pow(0.004905, 2);;   // acc bias in continuous
    parameterPtr->biasOmegaCovariance = gtsam::I_3x3 * pow(0.000001454441043, 2);  // gyro bias in continuous
    parameterPtr->biasAccOmegaInt = gtsam::I_6x6 * 1e-5; // error in the bias used for preintegration
    preintegratedPtr = std::make_shared<gtsam::PreintegrationType>(parameterPtr,*imu_biasPtr.get());
    ProjectNoisePtr = gtsam::noiseModel::Isotropic::Sigma(2, 0.1); // error in the bias used for projection
    cameraPtr=boost::make_shared<gtsam::Cal3_S2>(15,12,20,3,12);
    assert(preintegratedPtr);
}
Imu_Integration::~Imu_Integration(){

}
bool Imu_Integration::update(float* imu){
    if(t_1<1e-4){
        preintegratedPtr->integrateMeasurement(Eigen::Vector3d(imu[0],imu[1],imu[2]),Eigen::Vector3d(imu[3],imu[4],imu[5]),0.005);
        t_1=imu[6];
    }else{
        preintegratedPtr->integrateMeasurement(Eigen::Vector3d(imu[0],imu[1],imu[2]),Eigen::Vector3d(imu[3],imu[4],imu[5]),imu[6]-t_1);
        t_1=imu[6];
    }
    return true;
}
bool Imu_Integration::reset(){
    preintegratedPtr->resetIntegrationAndSetBias(*imu_biasPtr.get());
    return true;
}
bool Imu_Integration::predict(boost::shared_ptr<svo::Frame>& new_frame,std::size_t& num_obs){
    // Adding IMU factor and and pre-integration.
    auto preint_imu_combined =dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*preintegratedPtr);
    gtsam::NavState Estimate_state = preintegratedPtr->predict(*statePtr.get(), *imu_biasPtr.get());
    ++imu_factor_id;
    valuesPtr->insert(P(imu_factor_id), Estimate_state.pose());
    valuesPtr->insert(V(imu_factor_id), Estimate_state.v());
    valuesPtr->insert(B(imu_factor_id), *imu_biasPtr.get());
    gtsam::CombinedImuFactor imu_factor(P(imu_factor_id - 1), V(imu_factor_id - 1), P(imu_factor_id), V(imu_factor_id), B(imu_factor_id - 1), B(imu_factor_id),
                                 preint_imu_combined);
    graphPtr->emplace_shared<gtsam::CombinedImuFactor>(imu_factor);
    for(auto&& f:enumerate(new_frame->fts_)){
       if(f.item->point==NULL)continue;
        valuesPtr->insert(F(f.index), new_frame->T_f_w_ * f.item->point->pos_);
        graphPtr->emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2> >(
                f.item->px, ProjectNoisePtr, P( imu_factor_id), F(f.index), cameraPtr);
    }

    // Now optimize.
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");
    gtsam::LevenbergMarquardtOptimizer optimizer(*graphPtr.get(), *valuesPtr.get(), params);
    gtsam::Values result = optimizer.optimize();
    // Overwrite the beginning of the preintegration for the next step.
    statePtr=std::make_shared<gtsam::NavState>(result.at<gtsam::Pose3>(P(imu_factor_id)), result.at<gtsam::Vector3>(V(imu_factor_id)));
    imu_biasPtr=std::make_shared<gtsam::imuBias::ConstantBias>(result.at<gtsam::imuBias::ConstantBias>(B(imu_factor_id)));
    preintegratedPtr->resetIntegrationAndSetBias(*imu_biasPtr.get());
    t_1=1e-18;
    new_frame->T_f_w_=Sophus::SE3(statePtr->R(),statePtr->t());
   // new_frame->Cov_=gtsam::
    //if(!reset())return false;
    return true;
}