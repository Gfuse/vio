//
// Created by root on 4/13/21.
//

#ifndef SVO_VisionFactorLandMark_H
#define SVO_VisionFactorLandMark_H
#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <vikit/robust_cost.h>
#include <vikit/math_utils.h>
#include <gpu_svo/frame.h>

namespace svo {

    class VisionFactorLandMark: public gtsam::NoiseModelFactor3<gtsam::Pose3,gtsam::Pose3,gtsam::Point3> {

    private:
        // measurement information
        gtsam::Point3 feature_new;   ///<Unit-bearing vector of the feature.
        gtsam::Point3 feature_ref;   ///<Unit-bearing vector of the feature.
        int level;

    public:
        /// shorthand for a smart pointer to a factor
        typedef boost::shared_ptr<VisionFactorLandMark> shared_ptr;

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
        VisionFactorLandMark(const gtsam::Point3& feature_new,const gtsam::Point3& feature_ref, int  level,
                     const gtsam::SharedNoiseModel& H,
                      gtsam::Key poseKey_ref, gtsam::Key poseKey_new,gtsam::Key landmarkKey ) :
                gtsam::NoiseModelFactor3<gtsam::Pose3,gtsam::Pose3,gtsam::Point3>(H, poseKey_ref,poseKey_new,landmarkKey), feature_new(feature_new),
                feature_ref(feature_ref),level(level){}
        virtual ~VisionFactorLandMark(){}

        /// error function
        /// @param p_new    the Estimated pose in Pose3
        /// @param P_ref    the Estimated point of landmark
        /// @param l    the Estimated point of landmark
        /// @param H    the optional Jacobian matrix, which use boost optional and has default null pointer
        gtsam::Vector evaluateError(const gtsam::Pose3& p_ref,const gtsam::Pose3& p_new, const gtsam::Point3& l,
                                    boost::optional<gtsam::Matrix&> H1 = boost::none,boost::optional<gtsam::Matrix&> H2 = boost::none,
        boost::optional<gtsam::Matrix&> H3 = boost::none) const {
            // note that use boost optional like a pointer
            // only calculate jacobian matrix when non-null pointer exists
            gtsam::Matrix26 J,J1;
            gtsam::Matrix23 J2,J3;

            gtsam::Point3 point(p_ref*l);
            svo::Frame::jacobian_xyz2uv(point, J);
            svo::Frame::jacobian_l2uv(point,J2);

            gtsam::Point3 point2(p_new*l);
            svo::Frame::jacobian_xyz2uv(point2, J1);
            svo::Frame::jacobian_l2uv(point2,J3);
            Vector2d e1=vk::project2d(feature_ref) - vk::project2d(point)*1.0 / (1<<level);
            Vector2d e2=vk::project2d(feature_new) - vk::project2d(point2)*1.0 / (1<<level);
            Vector4d e=(gtsam::Matrix(4,1)<< e1,e2).finished();
            //e*=1.0 / (1<<level);
            if(H1)(*H1) = (gtsam::Matrix(4,6)<< J,gtsam::Matrix(2,6).setZero()).finished()*1.0 / (1<<level);
            if(H2)(*H2) = (gtsam::Matrix(4,6)<< gtsam::Matrix(2,6).setZero(),J1).finished()*1.0 / (1<<level);
            if(H3)(*H3) = (gtsam::Matrix(4,3)<< J2,J3).finished()*1.0 / (1<<level);
            return e;
        }
        // The second is a 'clone' function that allows the factor to be copied. Under most
        // circumstances, the following code that employs the default copy constructor should
        // work fine.
        virtual gtsam::NonlinearFactor::shared_ptr clone() const {
            return boost::static_pointer_cast<gtsam::NonlinearFactor>(
                    gtsam::NonlinearFactor::shared_ptr(new VisionFactorLandMark(*this))); }

    };

} // namespace gtsamexamples
#endif //SVO_SVOVISIONFACTOR_H
