//
// Created by gufe on 4/9/21.
//

#ifndef SVO_IMU_INTEGRATION_H
#define SVO_IMU_INTEGRATION_H
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <sophus/se3.h>
#include <gpu_svo/frame.h>

using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::P;  // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::L;  // Landmark

class Imu_Integration{
public:
    Imu_Integration(Sophus::SE3& SE_init);
    ~Imu_Integration();
    bool reset(gtsam::ISAM2& optimizer,boost::shared_ptr<svo::Frame>& new_frame);
    bool reset(gtsam::LevenbergMarquardtOptimizer& optimizer,boost::shared_ptr<svo::Frame>& new_frame);
    bool update(double* imu= nullptr);
    bool predict(boost::shared_ptr<svo::Frame>&,
                 boost::shared_ptr<svo::Frame>& last_frame ,std::size_t&,const double reproj_thresh);
    bool init(boost::shared_ptr<svo::Frame>&);
    Sophus::SE3 preintegrate_predict();
private:
    std::shared_ptr<gtsam::NavState> statePtr;
    std::shared_ptr<gtsam::imuBias::ConstantBias> imu_biasPtr;
    std::shared_ptr<gtsam::Values> valuesPtr;
    std::shared_ptr<gtsam::NonlinearFactorGraph> graphPtr;
    boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> parameterPtr;
    std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> preintegratedPtr;
    boost::shared_ptr<gtsam::noiseModel::Isotropic> ProjectNoisePtr;
    std::chrono::steady_clock::time_point t_1;
    std::uint32_t imu_factor_id=0;
    uint imu_n=0;
    std::shared_ptr<gtsam::ISAM2Params> optimizerParamPtr;
    gtsam::NavState Integ_predict;
    bool syn=false;
};
#endif //SVO_IMU_INTEGRATION_H
