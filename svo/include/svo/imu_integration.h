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
#include <sophus/se3.h>
#include <svo/frame.h>

using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::P;  // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::F;  // Feature

//https://stackoverflow.com/questions/24881799/get-index-in-c11-foreach-loop
// Wrapper class
template <typename T>
class enumerate_impl
{
public:
    // The return value of the operator* of the iterator, this
    // is what you will get inside of the for loop
    struct item
    {
        size_t index;
        typename T::value_type & item;
    };
    typedef item value_type;

    // Custom iterator with minimal interface
    struct iterator
    {
        iterator(typename T::iterator _it, size_t counter=0) :
                it(_it), counter(counter)
        {}

        iterator operator++()
        {
            return iterator(++it, ++counter);
        }

        bool operator!=(iterator other)
        {
            return it != other.it;
        }

        typename T::iterator::value_type item()
        {
            return *it;
        }

        value_type operator*()
        {
            return value_type{counter, *it};
        }

        size_t index()
        {
            return counter;
        }

    private:
        typename T::iterator it;
        size_t counter;
    };

    enumerate_impl(T & t) : container(t) {}

    iterator begin()
    {
        return iterator(container.begin());
    }

    iterator end()
    {
        return iterator(container.end());
    }

private:
    T & container;
};

// A templated free function allows you to create the wrapper class
// conveniently
template <typename T>
enumerate_impl<T> enumerate(T & t)
{
    return enumerate_impl<T>(t);
}

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
    boost::shared_ptr<gtsam::noiseModel::Isotropic> ProjectNoisePtr;
    boost::shared_ptr<gtsam::Cal3_S2> cameraPtr;
    std::uint32_t t_1=1e-18;
    std::uint32_t imu_factor_id=0;
};
#endif //SVO_IMU_INTEGRATION_H
