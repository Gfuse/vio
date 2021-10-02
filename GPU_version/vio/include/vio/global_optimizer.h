// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SVO_DEPTH_FILTER_H_
#define SVO_DEPTH_FILTER_H_

#include <queue>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <vio/global.h>
#include <vio/feature_detection.h>
#include <vio/matcher.h>
#include <vio/map.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/stuff/sampler.h>
#include <memory>

G2O_USE_OPTIMIZATION_LIBRARY(cholmod)


namespace vio {

class BA_Glob
{
public:

  BA_Glob(Map& map);

  virtual ~BA_Glob();

  /// Start this thread when seed updating should be in a parallel thread.
  void startThread();

  /// Stop the parallel thread that is running.
  void stopThread();

  void new_key_frame(){
      boost::unique_lock< boost::mutex > lk( mtx_);
      new_keyframe_=true;
      cond_.notify_one();
#if VIO_DEBUG
      fprintf(log_,"[%s] New key frame \n",
              vio::time_in_HH_MM_SS_MMM().c_str());
#endif
  }
 boost::mutex ba_mux_;
protected:
  bool new_keyframe_=false;
  boost::condition_variable cond_;
  boost::mutex mtx_;
  boost::thread* thread_;
  Map& map_;

#if VIO_DEBUG
  FILE* log_=nullptr;
#endif
        /// Temporary container to hold the g2o edge with reference to frame and point.
        struct EdgeContainerSE3{
            g2o::EdgeProjectXYZ2UV*     edge;
            std::shared_ptr<Frame>          frame;
            std::shared_ptr<Feature>        feature;
        };

/// Create a g2o vertice from a keyframe object.
        g2o::VertexSE3Expmap* createG2oFrameSE3(
                FramePtr kf,
                size_t id,
                bool fixed);
    /// Creates a g2o vertice from a mappoint object.
        g2o::VertexSBAPointXYZ* createG2oPoint(
                Vector3d pos,
                size_t id,
                bool fixed);
  /// Creates a g2o edge between a g2o keyframe and mappoint vertice with the provided measurement.
        g2o::EdgeProjectXYZ2UV* createG2oEdgeSE3(
                g2o::VertexSE3Expmap* v_kf,
                g2o::VertexSBAPointXYZ* v_mp,
                const Vector2d& f_up,
                bool robust_kernel,
                double huber_width,
                double weight = 1);
  /// A thread that is continuously optimizing the map.
  /// Global bundle adjustment.
  void updateLoop();
};

} // namespace vio

#endif // SVO_DEPTH_FILTER_H_
