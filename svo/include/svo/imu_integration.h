//
// Created by gufe on 4/9/21.
//

#ifndef SVO_IMU_INTEGRATION_H
#define SVO_IMU_INTEGRATION_H
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/inference/Symbol.h>
#include <sophus/se3.h>
#include <svo/frame.h>

using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::P;  // Pose3 (x,y,z,r,p,y)

class Imu_Integration{
public:
    Imu_Integration(Sophus::SE3& SE_init);
    ~Imu_Integration();
    bool reset();
    bool update(float* imu= nullptr);
    bool predict(boost::shared_ptr<svo::Frame>&,std::size_t&);
private:
    std::shared_ptr<gtsam::NavState> statePtr;
    std::shared_ptr<gtsam::imuBias::ConstantBias> imu_biasPtr;
    std::unique_ptr<gtsam::Values> valuesPtr;
    std::unique_ptr<gtsam::NonlinearFactorGraph> graphPtr;
    boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> parameterPtr;
    std::shared_ptr<gtsam::PreintegrationType> preintegratedPtr;
    std::uint32_t t_1=1e-18;
    std::uint32_t imu_factor_id=0;
};
#endif //SVO_IMU_INTEGRATION_H
