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

    Reprojector::Reprojector(vk::AbstractCamera *cam, Map &map) :
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
            opencl* gpu_fast_) {
        if(frame->id_<1)return;
        resetGrid();
        cv::Mat descriptors_cur;
        Features keypoints_cur;
        cv::Ptr<cv::xfeatures2d::FREAK> extractor = cv::xfeatures2d::FREAK::create(true, true, 22.0f, 4);
        cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        feature_detection_mut_.lock();
        feature_detection::FastDetector detector(
                frame->img().cols, frame->img().rows, Config::gridSize(), gpu_fast_,Config::nPyrLevels());
        detector.detect(frame, frame->img_pyr_, Config::triangMinCornerScore(), keypoints_cur,&descriptors_cur);
        feature_detection_mut_.unlock();
        list<pair<FramePtr, double> > close_kfs;
        map_.getCloseKeyframes(frame, close_kfs);
        if (!last_frame->fts_.empty())
            close_kfs.push_back(pair<FramePtr, double>(last_frame, (frame->T_f_w_.se2().translation() -
                                                                    last_frame->T_f_w_.se2().translation()).norm()));
        close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                       boost::bind(&std::pair<FramePtr, double>::second, _2));
        overlap_kfs.reserve(options_.max_n_kfs);
        boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
        for (auto &&it_frame:_for(close_kfs)) {
            if (it_frame.index > options_.max_n_kfs)continue;
            std::vector<cv::KeyPoint> keypoints_kfs;
            std::vector<cv::DMatch> matches;
            cv::Mat descriptors_ref;
            overlap_kfs.push_back(pair<FramePtr, size_t>(it_frame.item.first, 0));
            for (auto &&it_ftr:it_frame.item.first->fts_) {
                keypoints_kfs.push_back(cv::KeyPoint(it_ftr->px.x(), it_ftr->px.y(), 7));
            }
            extractor->compute(it_frame.item.first->img(), keypoints_kfs, descriptors_ref);
            matcher->match(descriptors_ref, descriptors_cur, matches);
            for (auto &&f: _for(it_frame.item.first->fts_)) {
                for (auto &&match:matches) {
                    if (match.queryIdx != f.index)continue;
                    if(keypoints_cur.size() < match.trainIdx)continue;
                    list<std::shared_ptr<Feature>>::iterator point=keypoints_cur.begin();
                    std::advance(point,match.trainIdx);
                    if(!(*point))continue;
                    if (f.item->point == NULL){
                        const int k = static_cast<int>((*point)->px.y() / grid_.cell_size) *
                                      grid_.grid_n_cols
                                      + static_cast<int>((*point)->px.x() / grid_.cell_size);
                        if(k>grid_.cells.size()-1)continue;
                        assert(grid_.cells.at(k) != nullptr);
                        Vector2d px((int) (*point)->px.x(),
                                    (int) (*point)->px.y());
                        SE3 T_ref_cur=it_frame.item.first->getSE3Inv()*frame->se3();
                        Vector3d new_point=vk::triangulateFeatureNonLin(T_ref_cur.rotation_matrix(),T_ref_cur.translation(),
                                                                        frame->c2f(px),f.item->f);
                        if(new_point.z()<0.0)break;
                        frame->fts_.push_back(std::make_shared<Feature>(it_frame.item.first,
                                                                        std::make_shared<Point>(frame->se3()*new_point,f.item),
                                                                                px,f.item->f,(*point)->level));
                        frame->fts_.back()->point->last_frame_overlap_id_=it_frame.item.first->id_;
                        grid_.cells.at(k)->push_back(Candidate(frame->fts_.back()->point, px));
                        overlap_kfs.back().second++;
                        break;
                    }else{
                        if (f.item->point->last_frame_overlap_id_ == frame->id_)continue;
                        f.item->point->last_frame_overlap_id_ = frame->id_;
                        const int k = static_cast<int>((*point)->px.y()/ grid_.cell_size) *
                                      grid_.grid_n_cols
                                      + static_cast<int>((*point)->px.x() / grid_.cell_size);
                        if(k>grid_.cells.size()-1)continue;
                        assert(grid_.cells.at(k) != nullptr);
                        Vector2d px((int) (*point)->px.x(),
                                    (int) (*point)->px.y());
                        grid_.cells.at(k)->push_back(Candidate(f.item->point, px));
                        overlap_kfs.back().second++;
                        break;
                    }
                }
            }
        }
        {
            for (auto&& cell : grid_.cells) {
                if (reprojectCell(*cell, frame))
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

    bool Reprojector::reprojectCell(Cell &cell, FramePtr frame) {
        if(cell.size()<1)return false;
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

            syn_=true;

            if(!matcher_.findMatchDirect(*it->pt, *frame, it->px))
            {
                it->pt->n_failed_reproj_++;
                if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15)
                    map_.safeDeletePoint(it->pt);
                if(it->pt->type_ == Point::TYPE_CANDIDATE  && it->pt->n_failed_reproj_ > 30)
                    map_.point_candidates_.deleteCandidatePoint(it->pt);
                it = cell.erase(it);
               // ++it;
                continue;
            }
            syn_=false;
            it->pt->n_succeeded_reproj_++;
            if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10)
                it->pt->type_ = Point::TYPE_GOOD;

            std::shared_ptr<Feature> new_feature = std::make_shared<Feature>(frame, it->px, matcher_.search_level_);
            frame->addFeature(new_feature);
            // Here we add a reference in the feature to the 3D point, the other way
            // round is only done if this frame is selected as keyframe.
            new_feature->point = it->pt;
            if(matcher_.ref_ftr_== nullptr){
                cell.erase(it);
                return true;
            }
            assert(matcher_.ref_ftr_!= nullptr);
            if(matcher_.ref_ftr_->type == Feature::EDGELET)
            {
                new_feature->type = Feature::EDGELET;
                new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
                new_feature->grad.normalize();
            }
            // If the keyframe is selected and we reproject the rest, we don't have to
            // check this point anymore.
            cell.erase(it);

            // Maximum one point per cell.
            return true;
        }
        return false;
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO original implementation
    bool Reprojector::reprojectPoint(FramePtr frame, std::shared_ptr<Point> point) {
        Vector2d px(frame->w2c(point->pos_));
        if (frame->cam_->isInFrame(px.cast<int>(), 8)) // 8px is the patch size in the matcher
        {
            const int k = static_cast<int>(px[1] / grid_.cell_size) * grid_.grid_n_cols
                          + static_cast<int>(px[0] / grid_.cell_size);
            grid_.cells.at(k)->push_back(Candidate(point, px));
            return true;
        }
        return false;
    }
}