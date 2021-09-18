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
#include <vio/math_utils.h>
#include <vio/abstract_camera.h>
#include <vio/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <vio/global.h>
#include <vio/depth_filter.h>
#include <vio/frame.h>
#include <vio/point.h>
#include <vio/feature.h>
#include <vio/matcher.h>
#include <vio/config.h>
#include <vio/feature_detection.h>
#if VIO_DEBUG
#include <sys/types.h>
#include <sys/stat.h>
#endif
namespace vio {

    int Seed::batch_counter = 0;
    int Seed::seed_counter = 0;

    Seed::Seed(std::shared_ptr<Feature> ftr, float depth_mean, float depth_min) :
            batch_id(batch_counter),
            id(seed_counter++),
            ftr(ftr),
            a(10),
            b(10),
            mu(1.0/depth_mean),
            z_range(1.0/depth_min),
            sigma2(z_range*z_range/36)
    {}

    DepthFilter::DepthFilter(feature_detection::DetectorPtr feature_detector, callback_t seed_converged_cb) :
            feature_detector_(feature_detector),
            seed_converged_cb_(seed_converged_cb),
            seeds_updating_halt_(false),
            thread_(NULL),
            new_keyframe_set_(false),
            new_keyframe_min_depth_(0.0),
            new_keyframe_mean_depth_(0.0)
    {
#if VIO_DEBUG
        log_ =fopen((std::string(PROJECT_DIR)+"/depth_filter_log.txt").c_str(),"w+");
        assert(log_);
        chmod((std::string(PROJECT_DIR)+"/depth_filter_log.txt").c_str(), ACCESSPERMS);
#endif
    }

    DepthFilter::~DepthFilter()
    {
        stopThread();
#if VIO_DEBUG
        fprintf(log_,"[%s] DepthFilter destructed.\n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif
    }

    void DepthFilter::startThread()
    {
        thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
#if VIO_DEBUG
        fprintf(log_,"[%s] init depth filter\n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif
    }

    void DepthFilter::stopThread()
    {
#if VIO_DEBUG
        fprintf(log_,"[%s] DepthFilter stop thread invoked.\n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif
        if(thread_ != NULL)
        {
#if VIO_DEBUG
            fprintf(log_,"[%s] DepthFilter interrupt and join thread \n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif
            seeds_updating_halt_ = true;
            thread_->interrupt();
            usleep(5000);
            thread_->join();
            thread_ = NULL;
        }
#if VIO_DEBUG
        fclose(log_);
#endif
    }

    void DepthFilter::addFrame(FramePtr frame)
    {
#if VIO_DEBUG
        fprintf(log_,"[%s] add new frame\n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif
        if(thread_ != NULL)
        {
            {
                if(frame_queue_.empty())lock_t lock(frame_queue_mut_);
                if(frame_queue_.size() > 10)
                    frame_queue_.pop();
                frame_queue_.push(frame);
            }
            seeds_updating_halt_ = false;
            frame_queue_cond_.notify_one();
        }else{
            updateSeeds(frame);
        }

    }

    void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min)
    {
        new_keyframe_min_depth_ = depth_min;
        new_keyframe_mean_depth_ = depth_mean;
#if VIO_DEBUG
        fprintf(log_,"[%s] add key frame\n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif

        if(thread_ != NULL)
        {
            new_keyframe_ = frame;
            new_keyframe_set_ = true;
            seeds_updating_halt_ = true;
            frame_queue_cond_.notify_one();
        }else{
            initializeSeeds(frame);
        }
    }

    void DepthFilter::initializeSeeds(FramePtr frame)
    {
        Features new_features;
#if VIO_DEBUG
        fprintf(log_,"[%s] init seeds\n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif
        feature_detector_->setExistingFeatures(frame->fts_);
        feature_detection_mut_.lock();
        feature_detector_->detect(frame, frame->img_pyr_,
                                  Config::triangMinCornerScore(), new_features);

        feature_detection_mut_.unlock();
        // initialize a seed for every new feature
        seeds_updating_halt_ = true;
        seeds_mut_.lock();
        ++Seed::batch_counter;
        for(auto&& ftr:new_features)seeds_.push_back(make_shared<Seed>(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));

#if VIO_DEBUG
        fprintf(log_,"[%s] DepthFilter: Initialized %d new seeds\n",vio::time_in_HH_MM_SS_MMM().c_str(),new_features.size());
#endif
        seeds_updating_halt_ = false;
        seeds_mut_.unlock();
    }

    void DepthFilter::removeKeyframe(FramePtr frame)
    {
        seeds_updating_halt_ = true;
#if VIO_DEBUG
        fprintf(log_,"[%s] remove key frame\n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif
        seeds_mut_.lock();
        std::list<std::shared_ptr<Seed>>::iterator it=seeds_.begin();
        size_t n_removed = 0;
        while(it!=seeds_.end())
        {
            if((*it)->ftr->frame->id_ == frame->id_)
            {
                it = seeds_.erase(it);
                ++n_removed;
            }
            else
                ++it;
        }
        seeds_updating_halt_ = false;
        seeds_mut_.unlock();
    }

    void DepthFilter::reset()
    {
        seeds_updating_halt_ = true;
        {
            seeds_mut_.lock();
            seeds_.clear();
            seeds_mut_.unlock();
        }
        while(!frame_queue_.empty())
            frame_queue_.pop();
        seeds_updating_halt_ = false;

#if VIO_DEBUG
        fprintf(log_,"[%s] DepthFilter: RESET.\n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif
    }

    void DepthFilter::updateSeedsLoop()
    {
        while(!boost::this_thread::interruption_requested())
        {
            FramePtr frame;
            {
                lock_t lock(frame_queue_mut_);
                while(frame_queue_.empty() && new_keyframe_set_ == false)
                    frame_queue_cond_.wait(lock);
                lock.unlock();
                if(new_keyframe_set_)
                {
                    new_keyframe_set_ = false;
                    clearFrameQueue();
                    frame = new_keyframe_;
                }
                else
                {
                    frame = frame_queue_.back();
                    frame_queue_.pop();
                }
            }
            updateSeeds(frame);
            if(frame->isKeyframe())
                initializeSeeds(frame);
        }
    }

    void DepthFilter::updateSeeds(FramePtr frame)
    {
        // update only a limited number of seeds, because we don't have time to do it
        // for all the seeds in every frame!
        size_t n_updates=0, n_failed_matches=0;
        lock_t lock(seeds_mut_);
        std::list<std::shared_ptr<Seed>>::iterator it=seeds_.begin();
#if VIO_DEBUG
        fprintf(log_,"[%s] update seed\n",vio::time_in_HH_MM_SS_MMM().c_str());
#endif
        const double focal_length = frame->cam_->errorMultiplier2();
        double px_noise = 1.0;
        double px_error_angle = atan2(px_noise,(2.0*focal_length))*2.0; // law of chord (sehnensatz)
        while( it!=seeds_.end())
        {
            // set this value true when seeds updating should be interrupted
            if(seeds_updating_halt_)
                return;

            if((*it)->ftr->frame == NULL){
                it = seeds_.erase(it);
                continue;
            }
            // check if point is visible in the current image
            SE2 T(frame->T_f_w_.so2()*(*it)->ftr->frame->T_f_w_.inverse().so2(),frame->T_f_w_.se2().translation()+(*it)->ftr->frame->T_f_w_.inverse().translation());
            ///TODO add 15 degrees roll orientation
            Quaterniond q;
            q = AngleAxisd(-0.122173, Vector3d::UnitX())
                * AngleAxisd(atan2(T.so2().unit_complex().imag(),T.so2().unit_complex().real()), Vector3d::UnitY())
                * AngleAxisd(0.0, Vector3d::UnitZ());
            SE3 T_cur_ref(q.toRotationMatrix(),Vector3d(T.translation()(0), 0.0,T.translation()(1)));
            const Vector3d xyz_f(T_cur_ref*(1.0/(*it)->mu * (*it)->ftr->f) );
#if VIO_DEBUG
            fprintf(log_,"[%s]  If point is visible? %f, %f, %f\n",vio::time_in_HH_MM_SS_MMM().c_str(),xyz_f.x(),xyz_f.y(),xyz_f.z());
#endif

            if(!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>())) {
                ++it; // point does not project in image
                continue;
            }

            float z_min = (*it)->mu + sqrt((*it)->sigma2);
            float z_max = max((*it)->mu - sqrt((*it)->sigma2), 0.00000001f);
#if VIO_DEBUG
            fprintf(log_,"[%s]  Z inverse min: %f, Z inverse max: %f\n",vio::time_in_HH_MM_SS_MMM().c_str(),z_min,z_max);
#endif

            double z;
            if(!matcher_.findEpipolarMatchDirect(
                    *(*it)->ftr->frame, *frame, *(*it)->ftr, 1.0/(*it)->mu, 1.0/z_min, 1.0/z_max, z))
            {
                (*it)->b++; // increase outlier probability when no match was found
                ++it;
                ++n_failed_matches;
                continue;
            }
            double tau = computeTau(T_cur_ref.inverse(), (*it)->ftr->f, z, px_error_angle);
            double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

            // update the estimate
            if(!updateSeed(1.0/z, tau_inverse*tau_inverse, *it)){
                ++it;
                ++n_failed_matches;
                continue;
            }
            ++n_updates;

            if(frame->isKeyframe())
            {
                // The feature detector should not initialize new seeds close to this location
                feature_detector_->setGridOccpuancy(matcher_.px_cur_);
            }

            // if the seed has converged, we initialize a new candidate point and remove the seed
            if(sqrt((*it)->sigma2) < (*it)->z_range/options_.seed_convergence_sigma2_thresh)
            {
                assert((*it)->ftr->point == NULL); // TODO this should not happen anymore
                Vector3d xyz_world((*it)->ftr->frame->getSE3Inv() * ((*it)->ftr->f/(*it)->mu));
                std::shared_ptr<Point> point = std::make_shared<Point>(xyz_world, (*it)->ftr);
                (*it)->ftr->point = point;
                seed_converged_cb_(point, (*it)->sigma2); // put in candidate list
                it = seeds_.erase(it);
            }
            else if(isnan(z_min))
            {
                it = seeds_.erase(it);
            }
            else
                ++it;
        }
    }

    void DepthFilter::clearFrameQueue()
    {
        while(!frame_queue_.empty())
            frame_queue_.pop();
    }

    void DepthFilter::getSeedsCopy(const FramePtr& frame, std::list<std::shared_ptr<Seed>>& seeds)
    {
        lock_t lock(seeds_mut_);
        for(std::list<std::shared_ptr<Seed>>::iterator it=seeds_.begin(); it!=seeds_.end(); ++it)
        {
            if ((*it)->ftr->frame->id_ == frame->id_)
                seeds.push_back(*it);
        }
    }
    bool DepthFilter::updateSeed(const float x, const float tau2, std::shared_ptr<Seed> seed)
    {
        float norm_scale = sqrt(seed->sigma2 + tau2);
        if(std::isnan(norm_scale))
            return false;
        boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
        float s2 = 1./(1./seed->sigma2 + 1./tau2);
        float m = s2*(seed->mu/seed->sigma2 + x/tau2);
        float C1 = seed->a/(seed->a+seed->b) * boost::math::pdf(nd, x);
        float C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range;
        float normalization_constant = C1 + C2;
        C1 /= normalization_constant;
        C2 /= normalization_constant;
        float f = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
        float e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
                  + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));

        // update parameters
        float mu_new = C1*m+C2*seed->mu;
        seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;
        seed->mu = mu_new;
        seed->a = (e-f)/(f-e/f);
        seed->b = seed->a*(1.0f-f)/f;
        return true;
    }

    double DepthFilter::computeTau(
            const SE3& T_ref_cur,
            const Vector3d& f,
            const double z,
            const double px_error_angle)
    {
        Vector3d t(T_ref_cur.translation());
        Vector3d a = f*z-t;
        double t_norm = t.norm();
        double beta_plus = acos(a.dot(-t)/(t_norm*a.norm())) + px_error_angle;
        return (t_norm*sin(beta_plus)/sin(PI-acos(f.dot(t)/t_norm)-beta_plus) - z); // ( triangle angles sum to PI and law of sines ) tau
    }

} // namespace vio
