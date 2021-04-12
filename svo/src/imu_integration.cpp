//
// Created by gfuse on 4/9/21.
//
//#define NDEBUG
#include "svo/imu_integration.h"
#include <memory>
#include <utility>

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
    parameterPtr = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD();
    parameterPtr->accelerometerCovariance=gtsam::I_3x3 * pow(0.0003924, 2); // acc white noise in continuous
    parameterPtr->gyroscopeCovariance=gtsam::I_3x3 * pow(0.00020568902491,2); // gyro white noise in continuous
    parameterPtr->integrationCovariance=gtsam::I_3x3 * 1e-8; // integration uncertainty continuous
    // PreintegrationCombinedMeasurements params:
    parameterPtr->biasAccCovariance = gtsam::I_3x3 * pow(0.004905, 2);;   // acc bias in continuous
    parameterPtr->biasOmegaCovariance = gtsam::I_3x3 * pow(0.000001454441043, 2);  // gyro bias in continuous
    parameterPtr->biasAccOmegaInt = gtsam::I_6x6 * 1e-5; // error in the bias used for preintegration
    preintegratedPtr = std::make_shared<gtsam::PreintegrationType>(parameterPtr,*imu_biasPtr.get());
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
    // Adding IMU factor and GPS factor and optimizing.
    auto preint_imu_combined =dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*preintegratedPtr);
    ++imu_factor_id;
    gtsam::CombinedImuFactor imu_factor(P(imu_factor_id - 1), V(imu_factor_id - 1), P(imu_factor_id), V(imu_factor_id), B(imu_factor_id - 1), B(imu_factor_id),
                                 preint_imu_combined);
    graphPtr->add(imu_factor);
    if(!reset())return false;
    return true;
}