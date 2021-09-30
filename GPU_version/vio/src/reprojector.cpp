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
#include <stdexcept>
#include <vio/reprojector.h>
#include <vio/frame.h>
#include <vio/point.h>
#include <vio/feature.h>
#include <vio/map.h>
#include <vio/config.h>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <vio/abstract_camera.h>
#include <vio/math_utils.h>
#include <vio/timer.h>
#include <fstream>
#include <vio/feature_detection.h>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <vio/for_it.hpp>

namespace vio {

    Reprojector::Reprojector(vk::AbstractCamera *cam, Map& map) :
            map_(map) {
        initializeGrid(cam);
    }

    Reprojector::~Reprojector() {
        std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell *c) { delete c; });
    }

    void Reprojector::initializeGrid(vk::AbstractCamera *cam) {
        grid_.cell_size = Config::gridSize();
        grid_.grid_n_cols = ceil(static_cast<double>(cam->width()) / grid_.cell_size);
        grid_.grid_n_rows = ceil(static_cast<double>(cam->height()) / grid_.cell_size);
        grid_.cells.resize(grid_.grid_n_cols * grid_.grid_n_rows);
        std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell *&c) { c = new Cell; });
        grid_.cell_order.resize(grid_.cells.size());
        for (size_t i = 0; i < grid_.cells.size(); ++i)
            grid_.cell_order[i] = i;
        random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end()); // maybe we should do it at every iteration!
    }

    void Reprojector::resetGrid() {
        n_matches_ = 0;
        n_trials_ = 0;
        std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell *c) { c->clear(); });
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO - second reimplementation
    void Reprojector::reprojectMap2(
            FramePtr frame,
            FramePtr last_frame,
            std::vector<std::pair<FramePtr, std::size_t> > &overlap_kfs,
            opencl* gpu_fast_,
            FILE* log_) {
        if(frame->id_<1)return;
        resetGrid();
        Features keypoints;
        cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING2,false);
        feature_detection::FastDetector detector(
                frame->img().cols, frame->img().rows, Config::gridSize(), gpu_fast_,Config::nPyrLevels());
        detector.detect(frame, frame->img_pyr_, Config::triangMinCornerScore(), keypoints);
        std::vector<cv::KeyPoint> keypoints_cur;
        list<std::shared_ptr<Feature>>::iterator it_cur=keypoints.begin();
        cv::Mat cur_des=cv::Mat(keypoints.size(),64,CV_8UC1);
//        std::cerr<<"84"<<std::endl;
        //cv::Mat maskup = cv::Mat(keypoints.size(),64,CV_8UC1);
        //cv::Mat maskdown = cv::Mat(keypoints.size(),64,CV_8UC1);
        cv::Mat maskup = cv::Mat_<uchar>(1, keypoints.size());
        cv::Mat maskdown = cv::Mat_<uchar>(1, keypoints.size());
//        std::cerr<<"88"<<std::endl;
////        uint8_t one[64] = {255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
////                           255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
////                           255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
////                           255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
////        uint8_t zero[64] = {0};
//        std::cerr<<"94"<<std::endl;

        for (int i=0;i<keypoints.size() && it_cur !=keypoints.end();++i) {
            keypoints_cur.push_back(cv::KeyPoint((*it_cur)->px.x(), (*it_cur)->px.y(), 7.f, (*it_cur)->score));
            memcpy(cur_des.data+(i*64),(*it_cur)->descriptor,sizeof(uint8_t)*64);
            if ((*it_cur)->px.y() < frame->img().rows/2) {
                maskup.at<uchar>(0, i) = 1;
                maskdown.at<uchar>(0, i) = 0;
                //memcpy(maskup.data + (i * 64), one, sizeof(uint8_t) * 64);
                //memcpy(maskdown.data + (i * 64), zero, sizeof(uint8_t) * 64);
            }
            else{
                maskup.at<uchar>(0, i) = 0;
                maskdown.at<uchar>(0, i) = 1;
                //memcpy(maskup.data + (i * 64), zero, sizeof(uint8_t) * 64);
                //memcpy(maskdown.data + (i * 64), one, sizeof(uint8_t) * 64);
            }
            ++it_cur;
        }
//        std::cerr<<"114"<<std::endl;
        list<pair<FramePtr, double> > close_kfs;
        map_.getCloseKeyframes(frame, close_kfs);
        if (!last_frame->fts_.empty())
            close_kfs.push_back(pair<FramePtr, double>(last_frame, (frame->T_f_w_.se2().translation() -
                                                                    last_frame->T_f_w_.se2().translation()).norm()));
        close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                       boost::bind(&std::pair<FramePtr, double>::second, _2));
        overlap_kfs.reserve(options_.max_n_kfs);
        std::vector<int> added_keypoints;
        for (auto &&it_frame:_for(close_kfs)) {
            if (it_frame.index > options_.max_n_kfs)continue;
            overlap_kfs.push_back(pair<FramePtr, size_t>(it_frame.item.first, 0));
            list<std::shared_ptr<Feature>>::iterator it_ref=it_frame.item.first->fts_.begin();
            for (int i=0;i<it_frame.item.first->fts_.size() && it_ref !=it_frame.item.first->fts_.end();++i) {
                std::vector<cv::KeyPoint> keypoints_kfs;
                std::vector<std::vector<cv::DMatch>>  matches;
                keypoints_kfs.push_back(cv::KeyPoint((*it_ref)->px.x(), (*it_ref)->px.y(), 7.f,(*it_ref)->score));


//                std::cerr<<"135"<<std::endl;
                cv::Mat ref_des=cv::Mat(1,64,CV_8UC1,(*it_ref)->descriptor);
//                std::cerr<<"136"<<std::endl;
                //std::cerr<<maskup.size<<" "<<maskdown.size<<" "<<cur_des.size<<std::endl;
//                cv::Mat mup, mdown;
//                cv::cvtColor(maskup, mup, cv::COLOR_GRAY2RGB);
//                cv::imwrite("/root/Projects/ROS_p_33_dev/src/p_33_vio/GPU_version/vio/maskup.png", mup);
//                cv::cvtColor(maskdown, mdown, cv::COLOR_GRAY2RGB);
//                cv::imwrite("/root/Projects/ROS_p_33_dev/src/p_33_vio/GPU_version/vio/maskdown.png", mdown);
//                std::cerr<<"137"<<std::endl;
                if ((*it_ref)->px.y() < frame->img().rows/2) {
                    std::cerr<<"if"<<std::endl;
                    matcher->knnMatch(ref_des, cur_des, matches, 1, cv::InputArray(maskup));
                }
                else{
                    std::cerr<<"else"<<std::endl;
                    matcher->knnMatch(ref_des, cur_des, matches, 1, cv::InputArray(maskdown));
                }
//                std::cerr<<"144"<<std::endl;
                //matcher->knnMatch(ref_des, cur_des, matches, 1);




                for(auto&& match:matches.back()){
                    it_cur=keypoints.begin();
                    std::advance(it_cur,match.trainIdx);
                    if(!(*it_cur))continue;
                    if ((*it_ref)->point == NULL){
                        const int k = static_cast<int>((*it_cur)->px.y() / grid_.cell_size) *
                                      grid_.grid_n_cols
                                      + static_cast<int>((*it_cur)->px.x() / grid_.cell_size);
                        if(k>grid_.cells.size()-1)continue;
                        assert(grid_.cells.at(k) != nullptr);
                        Vector2d px((int) (*it_cur)->px.x(),
                                    (int) (*it_cur)->px.y());
                        SE3 T_ref_cur=it_frame.item.first->se3().inverse()*frame->se3();
                        Vector3d pos=vk::triangulateFeatureNonLin(T_ref_cur.rotation_matrix(),T_ref_cur.translation(),
                                                                        frame->c2f(px),(*it_ref)->f);
                        if(pos.z()<0.0)continue;
                        std::shared_ptr<Point> new_point=std::make_shared<Point>(it_frame.item.first->se3()*pos,(*it_ref));
                        frame->addFeature(std::make_shared<Feature>(frame,
                                                                    new_point,
                                                                    px,(*it_cur)->level,(*it_cur)->score,(*it_cur)->descriptor));
                        new_point->addFrameRef(frame->fts_.back());
                        added_keypoints.push_back(match.trainIdx);
                        frame->fts_.back()->point->last_frame_overlap_id_=it_frame.item.first->id_;
                        grid_.cells.at(k)->push_back(Candidate(frame->fts_.back()->point, px));
                        overlap_kfs.back().second++;
                    }else{
                        (*it_ref)->point->last_frame_overlap_id_ = frame->id_;
                        const int k = static_cast<int>((*it_cur)->px.y()/ grid_.cell_size) *
                                      grid_.grid_n_cols
                                      + static_cast<int>((*it_cur)->px.x() / grid_.cell_size);
                        if(k>grid_.cells.size()-1)continue;
                        assert(grid_.cells.at(k) != nullptr);
                        Vector2d px((int) (*it_cur)->px.x(),
                                    (int) (*it_cur)->px.y());
                        frame->addFeature(std::make_shared<Feature>(frame,
                                                                    (*it_ref)->point,
                                                                    px,(*it_cur)->level,(*it_cur)->score,(*it_cur)->descriptor));
                        added_keypoints.push_back(match.trainIdx);
                        grid_.cells.at(k)->push_back(Candidate((*it_ref)->point, px));
                        overlap_kfs.back().second++;
                    }
                }
/*                cv::Mat imgMatch,key_point_ref,key_point_cur;
                cv::drawMatches( it_frame.item.first->img_pyr_[0], keypoints_kfs, frame->img_pyr_[0], keypoints_cur,matches, imgMatch);
                cv::imshow("img",imgMatch);
                cv::waitKey();*/
                ++it_ref;
            }
        }
        {
            for (auto&& cell : grid_.cells) {
                if (reprojectCell(*cell, frame, log_))
                    ++n_matches_;
                if (n_matches_ > (size_t) Config::maxFts())
                    break;
            }
        }
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool Reprojector::pointQualityComparator(Candidate &lhs, Candidate &rhs) {
        if (lhs.pt->type_ > rhs.pt->type_)
            return true;
        return false;
    }

    bool Reprojector::reprojectCell(Cell &cell, FramePtr frame, FILE* log_) {
        if(cell.size()<1){
            return false;
        }
        cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));
        Cell::iterator it=cell.begin();
        while(it!=cell.end())
        {
            ++n_trials_;
            if( it->pt ==NULL){
                it = cell.erase(it);
                continue;
            }
            if(it->pt->type_ == Point::TYPE_DELETED)
            {
                it = cell.erase(it);
                continue;
            }
            bool res = matcher_.findMatchDirect(*it->pt, *frame, it->px);
            if(!res)
            {
                it->pt->n_failed_reproj_++;
                if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15)it->pt.reset();
                it = cell.erase(it);
                continue;
            }

            it->pt->n_succeeded_reproj_++;
            if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10)
                it->pt->type_ = Point::TYPE_GOOD;

            cell.erase(it);

            // Maximum one point per cell.
            return true;
        }

        return false;
    }

}