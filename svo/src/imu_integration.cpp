//
// Created by gfuse on 4/9/21.
//
//#define NDEBUG
#include "svo/imu_integration.h"
#include <memory>
#include <utility>
#include "svo/feature.h"
#include "svo/point.h"
#include "svo/svoVisionFactor.h"
#include <gtsam/nonlinear/Marginals.h>

Imu_Integration::Imu_Integration(Sophus::SE3& SE_init){
    statePtr=std::make_shared<gtsam::NavState>(gtsam::Pose3(gtsam::Rot3(SE_init.rotation_matrix()),SE_init.translation()),
                                               gtsam::Vector3(0.0,0.0,0.0));
    imu_biasPtr=std::make_shared<gtsam::imuBias::ConstantBias>(gtsam::Vector3(1e-8,1e-8,1e-8),gtsam::Vector3(1e-8,1e-8,1e-8));
    valuesPtr=std::make_shared<gtsam::Values>();
    valuesPtr->insert(P(0),statePtr->pose());
    valuesPtr->insert(V(0),statePtr->v());
    valuesPtr->insert(B(0),*imu_biasPtr);
    graphPtr=std::make_shared<gtsam::NonlinearFactorGraph>();
    graphPtr->addPrior(P(0),statePtr->pose(),gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished()));
    graphPtr->addPrior(V(0),statePtr->v(),gtsam::noiseModel::Isotropic::Sigma(3, 0.01));
    graphPtr->addPrior(B(0),*imu_biasPtr,gtsam::noiseModel::Isotropic::Sigma(6, 1e-4));
    parameterPtr = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD();// TODO check the IMU frame
    parameterPtr->accelerometerCovariance=gtsam::I_3x3 * 1e-3; // acc white noise in continuous
    parameterPtr->gyroscopeCovariance=gtsam::I_3x3 * 1e-3; // gyro white noise in continuous
    parameterPtr->integrationCovariance=gtsam::I_3x3 * 1e-8; // integration uncertainty continuous
    parameterPtr->biasAccCovariance = gtsam::I_3x3 * 1e-8;   // acc bias in continuous
    parameterPtr->biasOmegaCovariance = gtsam::I_3x3 * 1e-8;  // gyro bias in continuous
    parameterPtr->biasAccOmegaInt = gtsam::I_6x6 * 1e-5; // error in the bias used for preintegration
    preintegratedPtr = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(parameterPtr,*imu_biasPtr);
    ProjectNoisePtr = gtsam::noiseModel::Isotropic::Sigma(2, 1e-8); // TODO change the noise used for projection
    assert(preintegratedPtr);
}
Imu_Integration::~Imu_Integration(){

}
bool Imu_Integration::update(double* imu){
    if(syn)return false;
    if(imu[6]>imu_stamp){
        auto in=std::chrono::steady_clock::now();
        if(std::chrono::duration<double>(in-t_1).count()>20.0){
            t_1=in;
            imu_stamp=imu[6];
        }else{
            preintegratedPtr->integrateMeasurement(Eigen::Vector3d(imu[0],imu[1],imu[2]),Eigen::Vector3d(imu[3],imu[4],imu[5]),std::chrono::duration<double>(in-t_1).count());
            t_1=in;
            imu_stamp=imu[6];
        }
        return true;
    }
    return false;

}
bool Imu_Integration::reset(gtsam::Values& result){
    // Overwrite the beginning of the preintegration for the next step.
    statePtr=std::make_shared<gtsam::NavState>(result.at<gtsam::Pose3>(P(imu_factor_id)), result.at<gtsam::Vector3>(V(imu_factor_id)));
    imu_biasPtr=std::make_shared<gtsam::imuBias::ConstantBias>(result.at<gtsam::imuBias::ConstantBias>(B(imu_factor_id)));
    preintegratedPtr->resetIntegrationAndSetBias(*imu_biasPtr);
    //graphPtr->resize(0);
    //valuesPtr->erase(P(imu_factor_id-1));
    //valuesPtr->erase(V(imu_factor_id-1));
    //valuesPtr->erase(B(imu_factor_id-1));
    //t_1=std::chrono::steady_clock::now();
    return true;
}
bool Imu_Integration::predict(boost::shared_ptr<svo::Frame>& new_frame,std::size_t& num_obs,const double reproj_thresh){
    // Adding IMU factor and and pre-integration.
    syn=true;
    auto preint_imu_combined =dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*preintegratedPtr);
    gtsam::NavState Estimate_state = preintegratedPtr->predict(*statePtr, *imu_biasPtr);
    std::cout<<"--------------------------------------------------------------------------PreIntegration----------------------------------------------"<<'\n';
    //preintegratedPtr->print();
    ++imu_factor_id;
    valuesPtr->insert(P(imu_factor_id), Estimate_state.pose());
    valuesPtr->insert(V(imu_factor_id), Estimate_state.v());
    valuesPtr->insert(B(imu_factor_id), *imu_biasPtr);
    gtsam::CombinedImuFactor imu_factor(P(imu_factor_id - 1), V(imu_factor_id - 1), P(imu_factor_id), V(imu_factor_id), B(imu_factor_id - 1), B(imu_factor_id),
                                 preint_imu_combined);
    graphPtr->emplace_shared<gtsam::CombinedImuFactor>(imu_factor);
    imu_factor.print();
    num_obs=0;
    for(auto&& f:new_frame->fts_){
       if(f->point==NULL)continue;
        graphPtr->emplace_shared<svo::VisionFactor>(f->f, f->point->pos_,f->level, ProjectNoisePtr,P(imu_factor_id));
        ++num_obs;
    }
    // Now optimize.
    std::cout<<"--------------------------------------------------------------------------graph----------------------------------------------"<<'\n';
    //graphPtr->print();
    std::cout<<"--------------------------------------------------------------------------values----------------------------------------------"<<'\n';
    valuesPtr->print();
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");
    gtsam::LevenbergMarquardtOptimizer optimizer(*graphPtr, *valuesPtr, params);
    gtsam::Values result = optimizer.optimize();
    gtsam::Marginals marginals(*graphPtr, result);
    std::cout<<"--------------------------------------------------------------------------marginals----------------------------------------------"<<'\n';
    //marginals.print();
    reset(result);
    syn=false;
    new_frame->T_f_w_=Sophus::SE3(statePtr->R(),statePtr->t());
    new_frame->Cov_=marginals.marginalCovariance(P(imu_factor_id));
    std::cout<<new_frame->Cov_<<'\n';
    double reproj_thresh_scaled = reproj_thresh / new_frame->cam_->errorMultiplier2();
    for(auto&& f:new_frame->fts_)
    {
        if(f->point == NULL)
            continue;
        Eigen::Vector2d e = vk::project2d(f->f) - vk::project2d(new_frame->T_f_w_ * f->point->pos_);
        double sqrt_inv_cov = 1.0 / (1<<f->level);
        e *= sqrt_inv_cov;
        if(e.norm() > reproj_thresh_scaled)
        {
            // we don't need to delete a reference in the point since it was not created yet
            f->point = NULL;
            --num_obs;
        }
    }
    return true;
}