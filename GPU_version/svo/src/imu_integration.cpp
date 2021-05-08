//
// Created by gfuse on 4/9/21.
//
//#define NDEBUG
#include "gpu_svo/imu_integration.h"
#include <memory>
#include <utility>
#include "gpu_svo/feature.h"
#include "gpu_svo/point.h"
#include "gpu_svo/svoVisionFactor.h"
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Pose3.h>
#include <vikit/params_helper.h>
#include <gpu_svo/config.h>

Imu_Integration::Imu_Integration(Sophus::SE3& SE_init){
    graphPtr=std::make_shared<gtsam::NonlinearFactorGraph>();
    valuesPtr=std::make_shared<gtsam::Values>();
    parameterPtr = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD();// TODO check the IMU frame
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
    //optimizerParamPtr->evaluateNonlinearError=true;
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
        preintegratedPtr->integrateMeasurement(Eigen::Vector3d(imu[0],imu[1],imu[2]),Eigen::Vector3d(imu[3],imu[4],imu[5]),0.005);
    }else{
        preintegratedPtr->integrateMeasurement(Eigen::Vector3d(imu[0],imu[1],imu[2]),Eigen::Vector3d(imu[3],imu[4],imu[5]),std::chrono::duration<double>(in-t_1).count());
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
    new_frame->T_f_w_=Sophus::SE3(statePtr->R(),statePtr->t());
    new_frame->Cov_=optimizer.marginalCovariance(P(imu_factor_id-1));
    graphPtr->resize(0);
    valuesPtr->clear();
    valuesPtr->insert(P(imu_factor_id-1),statePtr->pose());
    valuesPtr->insert(V(imu_factor_id-1),statePtr->v());
    valuesPtr->insert(B(imu_factor_id-1),*imu_biasPtr);
    graphPtr->addPrior(P(imu_factor_id-1),statePtr->pose(),optimizer.marginalCovariance(P(imu_factor_id-1)));
    graphPtr->addPrior(V(imu_factor_id-1),statePtr->v(),optimizer.marginalCovariance(V(imu_factor_id-1)));
    graphPtr->addPrior(B(imu_factor_id-1),*imu_biasPtr,optimizer.marginalCovariance(B(imu_factor_id-1)));
    return true;
}
bool Imu_Integration::reset(gtsam::LevenbergMarquardtOptimizer& optimizer,boost::shared_ptr<svo::Frame>& new_frame){
    gtsam::Values result = optimizer.optimize();
    // Overwrite the beginning of the preintegration for the next step.
    statePtr=std::make_shared<gtsam::NavState>(result.at<gtsam::Pose3>(P(imu_factor_id)), result.at<gtsam::Vector3>(V(imu_factor_id)));
    imu_biasPtr=std::make_shared<gtsam::imuBias::ConstantBias>(result.at<gtsam::imuBias::ConstantBias>(B(imu_factor_id)));
    preintegratedPtr->resetIntegrationAndSetBias(*imu_biasPtr);
    ++imu_factor_id;
    syn=false;
    gtsam::Marginals marginals(*graphPtr,result);
    new_frame->T_f_w_=Sophus::SE3(statePtr->R(),statePtr->t());
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
bool Imu_Integration::predict(boost::shared_ptr<svo::Frame>& new_frame,std::size_t& num_obs,const double reproj_thresh){
    usleep(5000);
    syn=true;
    if(imu_factor_id<1){
        statePtr=std::make_shared<gtsam::NavState>(gtsam::Pose3(new_frame->T_f_w_.matrix()),
                                                   gtsam::Vector3(1e-12,1e-12,1e-12));
        valuesPtr->insert(P(0),statePtr->pose());
        valuesPtr->insert(V(0),statePtr->v());
        valuesPtr->insert(B(0),*imu_biasPtr);
        graphPtr->addPrior(P(0),statePtr->pose(),gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9).finished()));
        graphPtr->addPrior(V(0),statePtr->v(),gtsam::noiseModel::Isotropic::Sigma(3, 1e-4));
        graphPtr->addPrior(B(0),*imu_biasPtr,gtsam::noiseModel::Isotropic::Sigma(6, 1e-2));
    }else{
        if(imu_n<1)return false;
        auto preint_imu_combined =dynamic_cast<const gtsam::PreintegratedCombinedMeasurements&>(*preintegratedPtr);
        valuesPtr->insert(P(imu_factor_id), gtsam::Pose3(new_frame->T_f_w_.matrix()));
        valuesPtr->insert(V(imu_factor_id), statePtr->v());
        valuesPtr->insert(B(imu_factor_id), *imu_biasPtr);
        gtsam::CombinedImuFactor imu_factor(P(imu_factor_id - 1), V(imu_factor_id - 1), P(imu_factor_id), V(imu_factor_id), B(imu_factor_id - 1), B(imu_factor_id),
                                            preint_imu_combined);
        graphPtr->emplace_shared<gtsam::CombinedImuFactor>(imu_factor);
        imu_n=0;
    }

    num_obs=0;
    for(auto&& f:new_frame->fts_){
       if(f->point==NULL)continue;
        graphPtr->emplace_shared<svo::VisionFactor>(f->f, f->point->pos_,f->level, ProjectNoisePtr,P(imu_factor_id));
        ++num_obs;
    }
    // Now optimize.
    //gtsam::ISAM2 optimizer(*optimizerParamPtr);
    //optimizer.update(*graphPtr, *valuesPtr);
    // Each call to iSAM2 update(*) performs one iteration of the iterative
    // nonlinear solver. If accuracy is desired at the expense of time,
    // update(*) can be called additional times to perform multiple optimizer
    // iterations every step.
    //optimizer.update();
   // optimizer.print();
    gtsam::LevenbergMarquardtOptimizer optimizer(*graphPtr, *valuesPtr);

    reset(optimizer,new_frame);
    // Remove Measurements with too large reprojection error
    double reproj_thresh_scaled = reproj_thresh / new_frame->cam_->errorMultiplier2();
    for(auto&& f:new_frame->fts_)
    {
        if(f->point == NULL)
            continue;
        Eigen::Vector2d e = (vk::project2d(f->f) - vk::project2d(new_frame->T_f_w_ * f->point->pos_))*1.0 / (1<<f->level);
        if(e.norm() > reproj_thresh_scaled)
        {
            // we don't need to delete a reference in the point since it was not created yet
            f->point = NULL;
            --num_obs;
        }
    }
    return true;
}