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

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <vio/for_it.hpp>

namespace vio {

Reprojector::Reprojector(vk::AbstractCamera* cam, Map& map) :
    map_(map)
{
  initializeGrid(cam);
}

Reprojector::~Reprojector()
{
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ delete c; });
}

void Reprojector::initializeGrid(vk::AbstractCamera* cam)
{
  grid_.cell_size = Config::gridSize();
  grid_.grid_n_cols = ceil(static_cast<double>(cam->width())/grid_.cell_size);
  grid_.grid_n_rows = ceil(static_cast<double>(cam->height())/grid_.cell_size);
  grid_.cells.resize(grid_.grid_n_cols*grid_.grid_n_rows);
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell*& c){ c = new Cell; });
  grid_.cell_order.resize(grid_.cells.size());
  for(size_t i=0; i<grid_.cells.size(); ++i)
    grid_.cell_order[i] = i;
  random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end()); // maybe we should do it at every iteration!
}

void Reprojector::resetGrid()
{
  n_matches_ = 0;
  n_trials_ = 0;
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ c->clear(); });
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO - Original implementation
void Reprojector::reprojectMap(
    FramePtr frame,
    std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs)
{
    resetGrid();

        list< pair<FramePtr,double> > close_kfs;
        map_.getCloseKeyframes(frame, close_kfs);

        close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                       boost::bind(&std::pair<FramePtr, double>::second, _2));

        size_t n = 0;
        overlap_kfs.reserve(options_.max_n_kfs);
        for(auto it_frame=close_kfs.begin(), ite_frame=close_kfs.end();
            it_frame!=ite_frame && n<options_.max_n_kfs; ++it_frame, ++n)
        {
            FramePtr ref_frame = it_frame->first;
            overlap_kfs.push_back(pair<FramePtr,size_t>(ref_frame,0));

            for(auto it_ftr=ref_frame->fts_.begin(), ite_ftr=ref_frame->fts_.end();
                it_ftr!=ite_ftr; ++it_ftr)
            {
                if((*it_ftr)->point == NULL)
                    continue;
                if((*it_ftr)->point->last_frame_overlap_id_ == frame->id_)
                    continue;
                (*it_ftr)->point->last_frame_overlap_id_= frame->id_;
                if(reprojectPoint(frame, (*it_ftr)->point)){
                    overlap_kfs.back().second++;
                }

            }
        }
        {
            boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
            auto it=map_.point_candidates_.candidates_.begin();
            while(it!=map_.point_candidates_.candidates_.end())
            {
                if(!reprojectPoint(frame, it->first))
                {
                    it->first->n_failed_reproj_ += 3;
                    if(it->first->n_failed_reproj_ > 30)
                    {
                        map_.point_candidates_.deleteCandidate(*it);
                        it = map_.point_candidates_.candidates_.erase(it);
                        continue;
                    }
                }
                ++it;
            }
        }
        for(size_t i=0; i<grid_.cells.size(); ++i)
        {
            if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame))
                ++n_matches_;
            if(n_matches_ > (size_t) Config::maxFts())
                break;
        }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO - First reimplementation
    void Reprojector::reprojectMap1(
            FramePtr frame,
            FramePtr framel,
            std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs)
    {
        resetGrid();

        std::vector<cv::KeyPoint> keypoints_ref;
        cv::KeyPoint * kp = new cv::KeyPoint;
        for(auto it_ftr=framel->fts_.begin(), ite_ftr=framel->fts_.end();
            it_ftr!=ite_ftr; ++it_ftr)
        {
            kp->pt.x = (*it_ftr)->px.x();
            kp->pt.y = (*it_ftr)->px.y();
            keypoints_ref.push_back(*kp);
        }

        overlap_kfs.reserve(1);

        overlap_kfs.push_back(pair<FramePtr,size_t>(frame,0));

        cv::Mat descriptors_ref, descriptors_cur;
        std::vector<cv::KeyPoint> keypoints_cur;
        cv::FAST(frame->img(),keypoints_cur,50);
        std::vector<cv::DMatch> matches;
        cv::Ptr<cv::xfeatures2d::FREAK> extractor = cv::xfeatures2d::FREAK::create(true, true, 50.0f, 4);
        extractor->compute(framel->img(), keypoints_ref, descriptors_ref);
        extractor->compute(frame->img(), keypoints_cur, descriptors_cur);
        cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        matcher->match(descriptors_ref,descriptors_cur,matches);
        //std::cerr<<"matches size : "<<matches.size()<<std::endl;

        for(int i=0; i < matches.size(); i++)
        {
            auto fts = framel->fts_.begin();
            auto fts2 = fts;
            for(int f = 0; f < matches[i].queryIdx; f++)
                fts2++;
            if(reprojectPoint1(frame, (*fts2)->point, &keypoints_cur[i])){
                overlap_kfs.back().second++;
            }
            keypoints_cur[i].class_id = 0;

        }
        for(int i=0; i < keypoints_cur.size(); i++)
        {
            if(keypoints_cur[i].class_id == 0)
                continue;
            if(reprojectPoint1(frame, nullptr, &keypoints_cur[i]))
                overlap_kfs.back().second++;
        }
        {
            boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
            auto it=map_.point_candidates_.candidates_.begin();
            while(it!=map_.point_candidates_.candidates_.end())
            {
                if(!reprojectPoint(frame, it->first))
                {
                    it->first->n_failed_reproj_ += 3;
                    if(it->first->n_failed_reproj_ > 30)
                    {
                        map_.point_candidates_.deleteCandidate(*it);
                        it = map_.point_candidates_.candidates_.erase(it);
                        continue;
                    }
                }
                ++it;
            }
        }
        for(size_t i=0; i<grid_.cells.size(); ++i)
        {
            if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame))
                ++n_matches_;
            if(n_matches_ > (size_t) Config::maxFts())
                break;
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO - second reimplementation
    void Reprojector::reprojectMap2(
            FramePtr frame,
            FramePtr last_frame,
            std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs)
    {
        resetGrid();
        cv::Mat descriptors_cur;
        std::vector<cv::KeyPoint> keypoints_cur;
        cv::Ptr<cv::xfeatures2d::FREAK> extractor = cv::xfeatures2d::FREAK::create(true, true, 22.0f, 4);
        cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        cv::FAST(frame->img(),keypoints_cur,50);
        extractor->compute(frame->img(), keypoints_cur, descriptors_cur);
        list< pair<FramePtr,double> > close_kfs;
        map_.getCloseKeyframes(frame, close_kfs);
        if(!last_frame->fts_.empty())close_kfs.push_back(pair<FramePtr,double>(last_frame, (frame->T_f_w_.se2().translation()-last_frame->T_f_w_.se2().translation()).norm()));
        close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                       boost::bind(&std::pair<FramePtr, double>::second, _2));
        overlap_kfs.reserve(options_.max_n_kfs);
        for(auto&& it_frame:_for(close_kfs)) {
            if(it_frame.index>options_.max_n_kfs)continue;
            std::vector<cv::KeyPoint> keypoints_kfs;
            std::vector<cv::DMatch> matches;
            cv::Mat descriptors_ref;
            overlap_kfs.push_back(pair<FramePtr,size_t>(it_frame.item.first,0));
            for(auto&& it_ftr:it_frame.item.first->fts_)
            {
                keypoints_kfs.push_back(cv::KeyPoint(it_ftr->px.x(),it_ftr->px.y(),1));
            }
            extractor->compute(it_frame.item.first->img(), keypoints_kfs, descriptors_ref);
            matcher->match(descriptors_ref, descriptors_cur, matches);
            for(auto&& f: _for(it_frame.item.first->fts_)){
                if(f.item->point==NULL)f.item->point=new Point(it_frame.item.first->cam_->cam2world(f.item->px.x(),f.item->px.y()));
                if(f.item->point->last_frame_overlap_id_ == frame->id_)continue;
                f.item->point->last_frame_overlap_id_ = frame->id_;
                for(auto&& match:matches){
                    if(match.queryIdx!=f.index)continue;
                    assert(keypoints_cur.size()>match.trainIdx);
                    const int k = static_cast<int>(keypoints_cur.at(match.trainIdx).pt.y / grid_.cell_size) * grid_.grid_n_cols
                                  + static_cast<int>(keypoints_cur.at(match.trainIdx).pt.x / grid_.cell_size);
                    assert(grid_.cells.at(k)!= nullptr);
                    Vector2d px((int)keypoints_cur.at(match.trainIdx).pt.x,(int)keypoints_cur.at(match.trainIdx).pt.y);
                    grid_.cells.at(k)->push_back(Candidate(f.item->point,px));
                    overlap_kfs.back().second++;
                    break;
                }
            }
        }
        for(size_t i=0; i<grid_.cells.size(); ++i)
        {
            if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame))
                ++n_matches_;
            if(n_matches_ > (size_t) Config::maxFts())
                break;
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool Reprojector::pointQualityComparator(Candidate& lhs, Candidate& rhs)
{
  if(lhs.pt->type_ > rhs.pt->type_)
    return true;
  return false;
}

bool Reprojector::reprojectCell(Cell& cell, FramePtr frame)
{
    cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));
  Cell::iterator it=cell.begin();
  while(it!=cell.end())
  {
    ++n_trials_;
    if(it->pt->type_ == Point::TYPE_DELETED)
    {
      it = cell.erase(it);
      continue;
    }/*
    bool found_match = true;
    if(options_.find_match_direct)
      found_match = matcher_.findMatchDirect(*it->pt, *frame, it->px);
    if(!found_match)
    {
      it->pt->n_failed_reproj_++;
      if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15)
        map_.safeDeletePoint(it->pt);
      if(it->pt->type_ == Point::TYPE_CANDIDATE  && it->pt->n_failed_reproj_ > 30)
        map_.point_candidates_.deleteCandidatePoint(it->pt);
      it = cell.erase(it);
      continue;
    }*/
    it->pt->n_succeeded_reproj_++;
    if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10)
      it->pt->type_ = Point::TYPE_GOOD;

    Feature* new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);
    frame->addFeature(new_feature);

    // Here we add a reference in the feature to the 3D point, the other way
    // round is only done if this frame is selected as keyframe.
    new_feature->point = it->pt;

    if(matcher_.ref_ftr_->type == Feature::EDGELET)
    {
      new_feature->type = Feature::EDGELET;
      new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
      new_feature->grad.normalize();
    }

    // If the keyframe is selected and we reproject the rest, we don't have to
    // check this point anymore.
    it = cell.erase(it);

    // Maximum one point per cell.
    return true;
  }
  return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO original implementation
bool Reprojector::reprojectPoint(FramePtr frame, Point* point)
{
  Vector2d px(frame->w2c(point->pos_));
  if(frame->cam_->isInFrame(px.cast<int>(), 8)) // 8px is the patch size in the matcher
  {
    const int k = static_cast<int>(px[1]/grid_.cell_size)*grid_.grid_n_cols
                + static_cast<int>(px[0]/grid_.cell_size);
    grid_.cells.at(k)->push_back(Candidate(point, px));
    return true;
  }
  return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO first reimplementation
    bool Reprojector::reprojectPoint1(FramePtr frame, Point* point, cv::KeyPoint* px_cv)
    {
        if(point != nullptr) {
            Vector2d px((int)px_cv->pt.x, (int)px_cv->pt.y);
            if (frame->cam_->isInFrame(px.cast<int>(), 8)) // 8px is the patch size in the matcher
            {
                const int k = static_cast<int>(px[1] / grid_.cell_size) * grid_.grid_n_cols
                              + static_cast<int>(px[0] / grid_.cell_size);
                if(k<grid_.cells.size()-1){
                    assert(grid_.cells.at(k)!= nullptr);
                    assert(point!=NULL);
                    grid_.cells.at(k)->push_back(Candidate(point, px));
                    return true;
                }
            }
            return false;
        }
        return false;/*else{
            Vector2d px(x, y);
            if (frame->cam_->isInFrame(px.cast<int>(), 8)) // 8px is the patch size in the matcher
            {
                const int k = static_cast<int>(px[1] / grid_.cell_size) * grid_.grid_n_cols
                              + static_cast<int>(px[0] / grid_.cell_size);
                frame->fts_.push_back(Fea)
                grid_.cells.at(k)->push_back(Candidate(vio::Point(Vector3d(frame->cam_->cam2world(x, y))), px));
                return true;
            }
            return false;
        }*/
    }

} // namespace vio
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////