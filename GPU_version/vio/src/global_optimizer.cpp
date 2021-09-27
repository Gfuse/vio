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

#include <algorithm>
#include <vio/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <vio/global.h>
#include <vio/global_optimizer.h>
#include <vio/frame.h>
#include <vio/point.h>
#include <vio/feature.h>
#include <vio/config.h>
#if VIO_DEBUG
#include <sys/types.h>
#include <sys/stat.h>
#endif
namespace vio {


    BA_Glob::BA_Glob(Map& map) :map_(map),thread_(NULL)
    {
#if VIO_DEBUG
        log_ =fopen((std::string(PROJECT_DIR)+"/depth_filter_log.txt").c_str(),"w+");
        assert(log_);
        chmod((std::string(PROJECT_DIR)+"/depth_filter_log.txt").c_str(), ACCESSPERMS);
#endif
    }

    BA_Glob::~BA_Glob()
    {
        stopThread();
    }

    void BA_Glob::startThread()
    {
        thread_ = new boost::thread(&BA_Glob::updateLoop, this);
    }

    void BA_Glob::stopThread()
    {

        if(thread_ != NULL)
        {
            thread_->interrupt();
            usleep(5000);
            thread_->join();
            thread_ = NULL;
        }
#if VIO_DEBUG
        fclose(log_);
#endif
    }
    void BA_Glob::updateLoop()
    {
        while(!boost::this_thread::interruption_requested())
        {
            boost::unique_lock< boost::mutex > lk( mtx_);
            while(map_.keyframes_.empty() || new_keyframe_ == false)
                cond_.wait(lk);
#if VIO_DEBUG
            fprintf(log_,"[%s] BA loop \n",
                    vio::time_in_HH_MM_SS_MMM().c_str());
#endif
            new_keyframe_=false;
            // init g2o
            setupG2o();
            list<EdgeContainerSE3> edges;
            list< pair<FramePtr,std::shared_ptr<Feature>> > incorrect_edges;
            // Go through all Keyframes
            size_t v_id = 0;
            point_mut_.lock();
            for(list<FramePtr>::iterator it_kf = map_.keyframes_.begin();
                it_kf != map_.keyframes_.end(); ++it_kf)
            {
                // New Keyframe Vertex
                std::shared_ptr<g2o::VertexSE3Expmap> v_kf = createG2oFrameSE3(*it_kf, v_id++, false);
                (*it_kf)->v_kf_ = v_kf;
                optimizer_->addVertex(v_kf.get());
                for(auto&& it_ftr:(*it_kf)->fts_)
                {
                    // for each keyframe add edges to all observed mapoints
                    std::shared_ptr<Point> mp = it_ftr->point;
                    if(mp == NULL)
                        continue;
                    std::shared_ptr<g2o::VertexPointXYZ> v_mp = mp->v_pt_;
                    if(v_mp == NULL)
                    {
                        // mappoint-vertex doesn't exist yet. create a new one:
                        v_mp = createG2oPoint(mp->pos_, v_id++, false);
                        mp->v_pt_ = v_mp;
                        optimizer_->addVertex(v_mp.get());
                    }

                    // Due to merging of mappoints it is possible that references in kfs suddenly
                    // have a very large reprojection error which may result in distorted results.
                    Vector2d error = vk::project2d(it_ftr->f) - vk::project2d((*it_kf)->w2f(mp->pos_));
                    if(error.squaredNorm() > pow(2,Config::poseOptimThresh()/(*it_kf)->cam_->errorMultiplier2()))
                        incorrect_edges.push_back(pair<FramePtr,std::shared_ptr<Feature>>(*it_kf, it_ftr));
                    else
                    {
                        std::shared_ptr<g2o::EdgeProjectXYZ2UV> e = createG2oEdgeSE3(v_kf, v_mp, vk::project2d(it_ftr->f),
                                                                         true,
                                                                         Config::poseOptimThresh()/(*it_kf)->cam_->errorMultiplier2()*Config::lobaRobustHuberWidth());
                        EdgeContainerSE3 edge;
                        edge.frame=(*it_kf);
                        edge.edge=e;
                        edge.feature=it_ftr;
                        edges.push_back(edge);
                        optimizer_->addEdge(e.get());
                    }
                }
            }

            // Optimization
            double init_error=0.0, final_error=0.0;
            if(Config::lobaNumIter() > 0)
                runSparseBAOptimizer( Config::lobaNumIter(), init_error, final_error);

            // Update Keyframe and MapPoint Positions
            for(list<FramePtr>::iterator it_kf = map_.keyframes_.begin();
                it_kf != map_.keyframes_.end(); ++it_kf)
            {
                (*it_kf)->T_f_w_ = SE3( (*it_kf)->v_kf_->estimate().rotation(),
                                        (*it_kf)->v_kf_->estimate().translation());
                (*it_kf)->v_kf_ = NULL;
                for(Features::iterator it_ftr=(*it_kf)->fts_.begin(); it_ftr!=(*it_kf)->fts_.end(); ++it_ftr)
                {
                    std::shared_ptr<Point> mp = (*it_ftr)->point;
                    if(mp == NULL)
                        continue;
                    if(mp->v_pt_ == NULL)
                        continue;       // mp was updated before
                    mp->pos_ = mp->v_pt_->estimate();
                    mp->v_pt_ = NULL;
                }
            }

            // Remove Measurements with too large reprojection error
            for(list< pair<FramePtr,std::shared_ptr<Feature>> >::iterator it=incorrect_edges.begin();
                it!=incorrect_edges.end(); ++it)
                map_.removePtFrameRef(it->first, it->second);

            for(list<EdgeContainerSE3>::iterator it = edges.begin(); it != edges.end(); ++it)
            {
                if(it->edge->chi2() > 0.001)
                {
                    map_.removePtFrameRef(it->frame, it->feature);
                }
            }
            Levenberg_.reset();
            point_mut_.unlock();
        }
    }

   void BA_Glob::setupG2o()
   {
       // TODO: What's happening with all this HEAP stuff? Memory Leak?
       optimizer_=std::make_shared<g2o::SparseOptimizer>();
       optimizer_->setVerbose(false);
       using LinearSolver = g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>;
       Levenberg_ = std::make_shared<g2o::OptimizationAlgorithmLevenberg>(
               g2o::make_unique<g2o::BlockSolver_6_3>(g2o::make_unique<LinearSolver>()));

       Levenberg_->setMaxTrialsAfterFailure(5);
       optimizer_->setAlgorithm(Levenberg_.get());

       // setup camera
       std::shared_ptr<g2o::CameraParameters> cam_params = std::make_shared<g2o::CameraParameters>(1.0, Vector2d(0.,0.), 0.);
       cam_params->setId(0);
       if (!optimizer_->addParameter(cam_params.get())) {
           assert(false);
       }
   }

   void
   BA_Glob::runSparseBAOptimizer(unsigned int num_iter,
                        double& init_error, double& final_error)
   {
       optimizer_->initializeOptimization();
       optimizer_->computeActiveErrors();
       init_error = optimizer_->activeChi2();
       optimizer_->optimize(num_iter);
       final_error = optimizer_->activeChi2();
   }

   std::shared_ptr<g2o::VertexSE3Expmap>
   BA_Glob::createG2oFrameSE3(FramePtr frame, size_t id, bool fixed)
   {
       std::shared_ptr<g2o::VertexSE3Expmap> v=std::make_shared<g2o::VertexSE3Expmap>();
       v->setId(id);
       v->setFixed(fixed);
       v->setEstimate(g2o::SE3Quat(frame->T_f_w_.se3().unit_quaternion(), frame->T_f_w_.se3().translation()));
       return v;
   }

   std::shared_ptr<g2o::VertexPointXYZ>
   BA_Glob::createG2oPoint(Vector3d pos,
                  size_t id,
                  bool fixed)
   {
       std::shared_ptr<g2o::VertexPointXYZ> v =std::make_shared<g2o::VertexPointXYZ>();
       v->setId(id);
       v->setFixed(fixed);
       v->setEstimate(pos);
       return v;
   }

   std::shared_ptr<g2o::EdgeProjectXYZ2UV> BA_Glob::createG2oEdgeSE3( std::shared_ptr<g2o::VertexSE3Expmap> v_frame,
                     std::shared_ptr<g2o::VertexPointXYZ> v_point,
                     const Vector2d& f_up,
                     bool robust_kernel,
                     double huber_width,
                     double weight)
   {
       std::shared_ptr<g2o::EdgeProjectXYZ2UV> e=std::make_shared<g2o::EdgeProjectXYZ2UV>();
       e->setVertex(0, (g2o::OptimizableGraph::Vertex*)(v_point).get());
       e->setVertex(1, (g2o::OptimizableGraph::Vertex*)(v_frame).get());
       e->setMeasurement(f_up);
       e->information() = weight * Eigen::Matrix2d::Identity(2,2);

       e->setRobustKernel(new g2o::RobustKernelHuber());
       e->setParameterId(0, 0); //old: e->setId(v_point->id());
       return e;
   }

} // namespace vio
