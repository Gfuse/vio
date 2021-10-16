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
#include <g2o/solvers/structure_only/structure_only_solver.h>
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

            ba_mux_.lock();
            // Go through all Keyframes
            v_id_ = 0;
            for(auto it_kf=map_.keyframes_.begin();it_kf!=map_.keyframes_.end();++it_kf)
            {
                // New Keyframe Vertex
                (*it_kf)->v_kf_ = createG2oFrameSE3(*it_kf);
                for(auto& it_ftr:(*it_kf)->fts_)
                {
                    if(it_ftr->point==NULL)continue;
                    if(it_ftr->point->type_ != vio::Point::TYPE_GOOD)continue;
                    // for each keyframe add edges to all observed mapoints
                    if(it_ftr->point->pos_.hasNaN())continue;
                    if(it_ftr->point->pos_.norm()==0.)continue;
                    if(it_ftr->point->v_pt_.first == NULL)
                    {
                        // mappoint-vertex doesn't exist yet. create a new one:
                        it_ftr->point->v_pt_.first = createG2oPoint(it_ftr->point->pos_);
                    }
                    it_ftr->point->v_pt_.second.push_back(createG2oEdgeSE3((*it_kf)->v_kf_, it_ftr->point->v_pt_.first, vk::project2d(it_ftr->f),
                                                                         true,
                                                                         Config::poseOptimThresh()/(*it_kf)->cam_->errorMultiplier2()*Config::lobaRobustHuberWidth()));
                }
            }
            ba_mux_.unlock();
            g2o::SparseOptimizer optimizer;
            optimizer.setVerbose(false);

            /*g2o::BlockSolver_6_3::LinearSolverType * linearSolver=new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();*/
            g2o::BlockSolver_6_3::LinearSolverType * linearSolver=new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
            g2o::BlockSolver_6_3 * solver_ptr= new g2o::BlockSolver_6_3(linearSolver);

            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

            optimizer.setAlgorithm(solver);
            double* pam=map_.keyframes_.back()->cam_->params();
            // setup camera
            g2o::CameraParameters* cam_params = new g2o::CameraParameters(1.0, Vector2d(0.,0.), 0.);
            cam_params->setId(0);
            if (!optimizer.addParameter(cam_params)) {
                assert(false && "Camera initialization in BA");
            }
            // init g2o
            g2o::OptimizableGraph::VertexContainer points;
            auto end=map_.keyframes_.end();
            auto end_1=end--;
            for(auto it=map_.keyframes_.begin(); it!=map_.keyframes_.end();++it){
                optimizer.addVertex((*it)->v_kf_);
                for(auto& p:(*it)->fts_){
                    if(p->point == NULL)continue;
                    if(p->point->v_pt_.first == NULL || p->point->v_pt_.second.empty())continue;
                    if(p->point->v_pt_.second.size()>1){
                        optimizer.addVertex(p->point->v_pt_.first);///TODO
                        points.push_back(p->point->v_pt_.first);
                        for(auto&& e:p->point->v_pt_.second)optimizer.addEdge(e);
                        p->point->v_pt_.second.clear();
                    }
                }
                if(it == end_1-- || it ==end || it==map_.keyframes_.end()) (*it)->v_kf_->setFixed(false);
                if((*it)->v_kf_->dimension()==3)points.push_back((*it)->v_kf_);
            }
            // Optimization
            if(points.empty() || optimizer.vertices().empty()){
                optimizer.clear();
                continue;
            }
            optimizer.initializeOptimization();
            optimizer.computeActiveErrors();
            g2o::StructureOnlySolver<3> structure_only_ba;
            structure_only_ba.calc(points, vio::Config::lobaNumIter());
#if VIO_DEBUG
            fprintf(log_,"[%s] init error: %f \n",
                    vio::time_in_HH_MM_SS_MMM().c_str(),optimizer.activeChi2());
#endif
            if(optimizer.optimize(vio::Config::lobaNumIter())<1){
                optimizer.clear();
                continue;
            }
#if VIO_DEBUG
            fprintf(log_,"[%s] end error: %f \n",
                    vio::time_in_HH_MM_SS_MMM().c_str(),optimizer.activeChi2());
#endif
            ba_mux_.lock();
            // Update Keyframe and MapPoint Positions
            for(list<FramePtr>::iterator it_kf = map_.keyframes_.begin();
                it_kf != map_.keyframes_.end();++it_kf)
            {
                (*it_kf)->T_f_w_ = SE2_5(SE3((*it_kf)->v_kf_->estimate().rotation().toRotationMatrix(),
                                        (*it_kf)->v_kf_->estimate().translation()));
                (*it_kf)->v_kf_ = NULL;
                for(Features::iterator it_ftr=(*it_kf)->fts_.begin(); it_ftr!=(*it_kf)->fts_.end(); ++it_ftr)
                {
                    if((*it_ftr)->point == NULL)
                        continue;
                    if((*it_ftr)->point->v_pt_.first == NULL)
                        continue;       // mp was updated before
                    (*it_ftr)->point->pos_ = (*it_ftr)->point->v_pt_.first->estimate();
                    (*it_ftr)->point->v_pt_ = std::make_pair<g2o::VertexSBAPointXYZ*,std::vector<g2o::EdgeProjectXYZ2UV*>>(NULL,
                            std::vector<g2o::EdgeProjectXYZ2UV*>());
                }
            }
            ba_mux_.unlock();
            optimizer.clear();
        }
    }

   g2o::VertexSE3Expmap*
   BA_Glob::createG2oFrameSE3(FramePtr frame)
   {
       g2o::VertexSE3Expmap* v= new g2o::VertexSE3Expmap();
       ++v_id_;
       v->setId(v_id_);
       v->setFixed(true);
       v->setEstimate(g2o::SE3Quat(frame->se3().unit_quaternion(), frame->se3().translation()));
       return v;
   }

   g2o::VertexSBAPointXYZ*
   BA_Glob::createG2oPoint(Vector3d pos)
   {
       ++v_id_;
       g2o::VertexSBAPointXYZ* v =new g2o::VertexSBAPointXYZ();
       v->setId(v_id_);
       v->setFixed(false);
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
