//
// Created by root on 4/13/21.
//

#ifndef SVO_SVOVISIONFACTOR_H
#define SVO_SVOVISIONFACTOR_H
#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <vikit/robust_cost.h>
#include <vikit/math_utils.h>
#include <vio/frame.h>

namespace vio {

    class VisionFactor: public gtsam::NoiseModelFactor1<gtsam::Pose3> {

    private:
        // measurement information
        gtsam::Point3 feature;   ///<Unit-bearing vector of the feature.
        gtsam::Point3 landmark;   ///<Unit-bearing vector of the feature.
        boost::optional<gtsam::Point3> body_P_sensor_; ///< TODO The pose of the sensor in the body frame
        int level;            ///< Image pyramid level where feature was extracted.

    public:
        /// shorthand for a smart pointer to a factor
        typedef boost::shared_ptr<VisionFactor> shared_ptr;

        /**
        * Constructor
        * @param feature Unit-bearing vector of the feature.
        * @param model is the standard deviation
        * @param pointKey is the index of the landmark
        * @param poseKey is the index of the camera pose
        * @param landmark 3D position of the landmark
        * @param body_P_sensor is the transform from body to sensor frame (default identity)
        * @param level Image pyramid level where feature was extracted.
        */
        VisionFactor(const gtsam::Point3& feature, const gtsam::Point3& landmark,const int level, const gtsam::SharedNoiseModel& Hp,
                      gtsam::Key poseKey,boost::optional<gtsam::Point3> body_P_sensor = boost::none) :
                gtsam::NoiseModelFactor1<gtsam::Pose3>(Hp, poseKey), feature(feature), landmark(landmark),body_P_sensor_(body_P_sensor),level(level){}
        virtual ~VisionFactor(){}

        /// error function
        /// @param p    the Estimated pose in Pose3
        /// @param H    the optional Jacobian matrix, which use boost optional and has default null pointer
        gtsam::Vector evaluateError(const gtsam::Pose3& p, boost::optional<gtsam::Matrix&> Hp = boost::none) const {
            // note that use boost optional like a pointer
            // only calculate jacobian matrix when non-null pointer exists
            gtsam::Matrix26 J;
            gtsam::Point3 point(p*landmark);
            svo::Frame::jacobian_xyz2uv(point, J);
            Vector2d e = (vk::project2d(feature) - vk::project2d(point))*1.0 / (1<<level);
            if(Hp)(*Hp) = (gtsam::Matrix(2,6)<< J * 1.0 / (1<<level)).finished();
            return e;
        }
        // The second is a 'clone' function that allows the factor to be copied. Under most
        // circumstances, the following code that employs the default copy constructor should
        // work fine.
        virtual gtsam::NonlinearFactor::shared_ptr clone() const {
            return boost::static_pointer_cast<gtsam::NonlinearFactor>(
                    gtsam::NonlinearFactor::shared_ptr(new VisionFactor(*this))); }

    };

} // namespace gtsamexamples
#endif //SVO_SVOVISIONFACTOR_H
