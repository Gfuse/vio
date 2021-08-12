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
#include <vio/robust_cost.h>
#include <vio/math_utils.h>
#include <vio/frame.h>

namespace vio {

    class VisionFactor: public gtsam::NoiseModelFactor1<gtsam::Pose3> {

    private:
        // measurement information
        gtsam::Pose3 pose;   ///<Estimated pose in svo after GaussNewton optimization.

    public:
        /// shorthand for a smart pointer to a factor
        typedef boost::shared_ptr<VisionFactor> shared_ptr;

        /**
        * Constructor
        * @param pose Camera pose.
        * @param pointKey is the index of the landmark
        */
        VisionFactor(const gtsam::Pose3& pose,
                     const gtsam::SharedNoiseModel& H,
                      gtsam::Key poseKey) :
                gtsam::NoiseModelFactor1<gtsam::Pose3>(H, poseKey), pose(pose){}
        virtual ~VisionFactor(){}

        /// error function
        /// @param p    the Estimated pose in Pose3
        /// @param H    the optional Jacobian matrix, which use boost optional and has default null pointer
        gtsam::Vector evaluateError(const gtsam::Pose3& p
                                    ,boost::optional<gtsam::Matrix&> H1 = boost::none) const {
            // note that use boost optional like a pointer
            // only calculate jacobian matrix when non-null pointer exists
            Eigen::Matrix<double,3,3> E_r=0.5*(p.rotation().matrix().transpose()*pose.rotation().matrix()-pose.rotation().matrix().transpose()*p.rotation().matrix());
            Vector3d e_t=p.translation()-pose.translation();
            if(H1)(*H1) = (gtsam::Matrix(3,6)<< 1,0,0,0,0,0,
                                                      0,0,1,0,0,0,
                                                      0,0,0,0,1,0).finished();
            return (gtsam::Matrix(3,1)<<e_t.x(),e_t.z(),-E_r(0,2)).finished();
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
