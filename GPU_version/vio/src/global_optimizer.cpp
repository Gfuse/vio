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
            fprintf(log_,"[%s] BA loop run \n",
                    vio::time_in_HH_MM_SS_MMM().c_str());
#endif
            new_keyframe_=false;
            g2o::SparseOptimizer optimizer;
            optimizer.setVerbose(false);

            g2o::BlockSolver_6_3::LinearSolverType * linearSolver=new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
            g2o::BlockSolver_6_3 * solver_ptr= new g2o::BlockSolver_6_3(linearSolver);

            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

            optimizer.setAlgorithm(solver);

            // setup camera
            g2o::CameraParameters* cam_params = new g2o::CameraParameters(1.0, Vector2d(0.,0.), 0.);
            cam_params->setId(0);
            if (!optimizer.addParameter(cam_params)) {
                assert(false);
            }
            // init g2o
            list<EdgeContainerSE3> edges;
            list< pair<FramePtr,std::shared_ptr<Feature>> > incorrect_edges;
            ba_mux_.lock();
            // Go through all Keyframes
            size_t v_id = 0;
            for(list<FramePtr>::iterator it_kf = map_.keyframes_.begin();
                it_kf != map_.keyframes_.end(); ++it_kf)
            {
                // New Keyframe Vertex
               g2o::VertexSE3Expmap* v_kf = createG2oFrameSE3(*it_kf, v_id++, false);
                (*it_kf)->v_kf_ = v_kf;
                optimizer.addVertex(v_kf);
                for(auto&& it_ftr:(*it_kf)->fts_)
                {
                    if(it_ftr->point==NULL)continue;
                    // for each keyframe add edges to all observed mapoints
                    std::shared_ptr<Point> mp = it_ftr->point;
                    g2o::VertexSBAPointXYZ* v_mp = mp->v_pt_;
                    if(v_mp == NULL)
                    {
                        // mappoint-vertex doesn't exist yet. create a new one:
                        v_mp = createG2oPoint(mp->pos_, v_id++, true);
                        mp->v_pt_ = v_mp;
                        optimizer.addVertex(v_mp);
                    }

                    // Due to merging of mappoints it is possible that references in kfs suddenly
                    // have a very large reprojection error which may result in distorted results.
                    Vector2d error = vk::project2d(it_ftr->f) - vk::project2d((*it_kf)->w2f(mp->pos_));
                    if(error.squaredNorm() > pow(2,Config::poseOptimThresh()/(*it_kf)->cam_->errorMultiplier2()))
                        incorrect_edges.push_back(pair<FramePtr,std::shared_ptr<Feature>>(*it_kf, it_ftr));
                    else
                    {
                        g2o::EdgeProjectXYZ2UV* e = createG2oEdgeSE3(v_kf, v_mp, vk::project2d(it_ftr->f),
                                                                         true,
                                                                         Config::poseOptimThresh()/(*it_kf)->cam_->errorMultiplier2()*Config::lobaRobustHuberWidth());
                        EdgeContainerSE3 edge;
                        edge.frame=(*it_kf);
                        edge.edge=e;
                        edge.feature=it_ftr;
                        edges.push_back(edge);
                        optimizer.addEdge(e);
                    }
                }
            }
            if(optimizer.vertices().size()<1)continue;
            // Optimization
            optimizer.initializeOptimization();
            optimizer.computeActiveErrors();
#if VIO_DEBUG
            fprintf(log_,"[%s] init error: %f \n",
                    vio::time_in_HH_MM_SS_MMM().c_str(),optimizer.activeChi2());
#endif
            if(optimizer.optimize(vio::Config::lobaNumIter())<1)continue;
#if VIO_DEBUG
            fprintf(log_,"[%s] end error: %f \n",
                    vio::time_in_HH_MM_SS_MMM().c_str(),optimizer.activeChi2());
#endif
            // Update Keyframe and MapPoint Positions
            for(list<FramePtr>::iterator it_kf = map_.keyframes_.begin();
                it_kf != map_.keyframes_.end();++it_kf)
            {
                (*it_kf)->T_f_w_ = SE2_5(SE3((*it_kf)->v_kf_->estimate().rotation(),
                                        (*it_kf)->v_kf_->estimate().translation()));
                (*it_kf)->v_kf_ = NULL;
                for(Features::iterator it_ftr=(*it_kf)->fts_.begin(); it_ftr!=(*it_kf)->fts_.end(); ++it_ftr)
                {
                    if((*it_ftr)->point == NULL)
                        continue;
                    std::shared_ptr<Point> mp = (*it_ftr)->point;
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
            ba_mux_.unlock();
        }
    }

   g2o::VertexSE3Expmap*
   BA_Glob::createG2oFrameSE3(FramePtr frame, size_t id, bool fixed)
   {
       g2o::VertexSE3Expmap* v= new g2o::VertexSE3Expmap();
       v->setId(id);
       v->setFixed(false);
       v->setEstimate(g2o::SE3Quat(frame->T_f_w_.se3().unit_quaternion(), frame->T_f_w_.se3().translation()));
       return v;
   }

   g2o::VertexSBAPointXYZ*
   BA_Glob::createG2oPoint(Vector3d pos,
                  size_t id,
                  bool fixed)
   {
       g2o::VertexSBAPointXYZ* v =new g2o::VertexSBAPointXYZ();
       v->setId(id);
       v->setFixed(fixed);
       v->setMarginalized(true);
       v->setEstimate(pos);
       return v;

   }

   g2o::EdgeProjectXYZ2UV* BA_Glob::createG2oEdgeSE3( g2o::VertexSE3Expmap* v_frame,
                     g2o::VertexSBAPointXYZ* v_point,
                     const Vector2d& f_up,
                     bool robust_kernel,
                     double huber_width,
                     double weight)
   {
       g2o::EdgeProjectXYZ2UV* e= new g2o::EdgeProjectXYZ2UV;
       e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
       e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
       e->setMeasurement(f_up);
       e->information() = weight*Eigen::Matrix2d::Identity();
       g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
       rk->setDelta(huber_width);
       e->setRobustKernel(rk);
       e->setParameterId(0, 0); //old: e->setId(v_point->id());
       return e;
   }

} // namespace vio
