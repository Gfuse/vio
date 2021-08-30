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

namespace vio {

Reprojector::Reprojector(vk::AbstractCamera* cam, Map& map/*, opencl* kernel*/) :
    map_(map)//, gpu_freak(kernel)
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

//void Reprojector::reprojectMap(
//    FramePtr frame,
//    std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs)
//{
//  resetGrid();
//
//  // Identify those Keyframes which share a common field of view.
////  SVO_START_TIMER("reproject_kfs");
//  list< pair<FramePtr,double> > close_kfs;
//  map_.getCloseKeyframes(frame, close_kfs);
//
//  // Sort KFs with overlap according to their closeness
//  close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
//                 boost::bind(&std::pair<FramePtr, double>::second, _2));
//
//  // Reproject all mappoints of the closest N kfs with overlap. We only store
//  // in which grid cell the points fall.
//  size_t n = 0;
//  overlap_kfs.reserve(options_.max_n_kfs);
//  for(auto it_frame=close_kfs.begin(), ite_frame=close_kfs.end();
//      it_frame!=ite_frame && n<options_.max_n_kfs; ++it_frame, ++n)
//  {
//    FramePtr ref_frame = it_frame->first;
//    overlap_kfs.push_back(pair<FramePtr,size_t>(ref_frame,0));
//
//    // Try to reproject each mappoint that the other KF observes
//    for(auto it_ftr=ref_frame->fts_.begin(), ite_ftr=ref_frame->fts_.end();
//        it_ftr!=ite_ftr; ++it_ftr)
//    {
//      // check if the feature has a mappoint assigned
//      if((*it_ftr)->point == NULL)
//        continue;
//
//      // make sure we project a point only once
//      if((*it_ftr)->point->last_projected_kf_id_ == frame->id_)
//        continue;
//      (*it_ftr)->point->last_projected_kf_id_ = frame->id_;
//      if(reprojectPoint(frame, (*it_ftr)->point)){
//          overlap_kfs.back().second++;
//      }
//
//    }
//  }
//  //SVO_STOP_TIMER("reproject_kfs");
//
//  // Now project all point candidates
// // SVO_START_TIMER("reproject_candidates");
//  {
//    boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
//    auto it=map_.point_candidates_.candidates_.begin();
//    while(it!=map_.point_candidates_.candidates_.end())
//    {
//      if(!reprojectPoint(frame, it->first))
//      {
//        it->first->n_failed_reproj_ += 3;
//        if(it->first->n_failed_reproj_ > 30)
//        {
//          map_.point_candidates_.deleteCandidate(*it);
//          it = map_.point_candidates_.candidates_.erase(it);
//          continue;
//        }
//      }
//      ++it;
//    }
//  } // unlock the mutex when out of scope
//  //SVO_STOP_TIMER("reproject_candidates");
//  // Now we go through each grid cell and select one point to match.
//  // At the end, we should have at maximum one reprojected point per cell.
//  //SVO_START_TIMER("feature_align");
//
////  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////  std::vector<cl_float4> kps; // keypoints
////  cl_float4  data;
////  int frame_number = 0;
////  for(auto it_frame = overlap_kfs.begin(), ite_frame = overlap_kfs.end(); it_frame != ite_frame; it_frame++){
////      for(auto it_ftr = it_frame->first->fts_.begin(), ite_ftr = it_frame->first->fts_.end(); it_ftr != ite_ftr; it_ftr++){
////          data.w = frame_number;
////          data.x = (*it_ftr)->px.x();
////          data.y = (*it_ftr)->px.y();
////          data.z = 0.0;
////          kps.push_back(data);
////      }
////      frame_number++;
////  }
//
//
////  cv::Mat imgbegin = overlap_kfs.begin()->first->img().clone();
////  cv::imwrite("/root/Projects/ROS/src/p_33_vio/GPU_version/img_overlap_kfs_begin.png",imgbegin);
////  cv::Mat imgbend = overlap_kfs.end()->first->img().clone();
////  cv::imwrite("/root/Projects/ROS/src/p_33_vio/GPU_version/img_overlap_kfs_end.png",imgbend);
//
////    ofstream fileref;
////    fileref.open("/root/Projects/ROS/src/p_33_vio/GPU_version/dataref.txt");
////  //cl_float4 keypoints[kps.size()];
////  //size_t counter = 0;
////  for(auto it = kps.begin(), ite = kps.end(); it != ite; it++) {
////      cl_float4  ftr;
////      ftr.w = it->w;
////      ftr.x = it->x;
////      ftr.y = it->y;
////      ftr.z = it->z;
////      //keypoints[counter++] = ftr;
////      fileref<<ftr.x<<","<<ftr.y<<","<<ftr.z<<std::endl;
////  }
////    fileref.close();
//
//
////  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////  ofstream filecur;
////  filecur.open("/root/Projects/ROS/src/p_33_vio/GPU_version/datacur.txt");
////  //cl_float3 keypoints[overlap_kfs.back().second];
////  size_t counter = 0;
////  for(size_t i = 0; i < grid_.cells.size(); ++i)
////  {
////      Cell* cell = grid_.cells.at(grid_.cell_order[i]);
////      for(auto it = cell->begin(), ite = cell->end(); it != ite; it++, counter++)
////      {
////          //keypoints[counter].x = it->px.x(); //x
////          //keypoints[counter].y = it->px.y(); //y
////          //keypoints[counter].z = 0.0; //angle
////          filecur<<it->px.x()<<","<<it->px.y()<<","<<0.0<<std::endl;
////      }
////  }
////  filecur.close();
//
//
////  opencl_initializer();
////  cv::Mat img = overlap_kfs.back().first->img();
////  cv::Mat integral_img=cv::Mat(cv::Size(701, 501),CV_8UC1);
////  gpu_freak->reload_buf(0,0,img);
////  gpu_freak->run(0);
////  gpu_freak->read(0,1,1,&integral_img);
//
////  // FREAK preprocessing
////  static const float FREAK_LOG2 = 0.693147180559945;
////  static const int FREAK_NB_ORIENTATION = 256;
////  static const int NB_ORIENPAIRS = 45;
////  static const int FREAK_NB_POINTS = 43;
////  static const int FREAK_SMALLEST_KP_SIZE = 7; // smallest size of keypoints
////  static const int NB_PAIRS = 512;
////  static const int NB_SCALES = 64;
////
////  bool orientationNormalized = true;
////  bool scaleNormalized = true;
////  float patternScale = 22.0f;
////  int nOctaves = 4;
////
////  bool extAll = false; // true if all pairs need to be extracted for pairs selection
////
////  cl_float3 patternLookup[NB_SCALES * FREAK_NB_ORIENTATION * FREAK_NB_POINTS]; // look-up table for the pattern points (position+sigma of all points at all scales and orientation)
////  int patternSizes[NB_SCALES]; // size of the pattern at a specific scale (used to check if a point is within image boundaries)
////  cl_uchar2 descriptionPairs[NB_PAIRS];
////  cl_int4 orientationPairs[NB_ORIENPAIRS];
////
////  // default pairs
////  static const int FREAK_DEF_PAIRS[NB_PAIRS] = {
////    404,431,818,511,181,52,311,874,774,543,719,230,417,205,11,
////    560,149,265,39,306,165,857,250,8,61,15,55,717,44,412,
////    592,134,761,695,660,782,625,487,549,516,271,665,762,392,178,
////    796,773,31,672,845,548,794,677,654,241,831,225,238,849,83,
////    691,484,826,707,122,517,583,731,328,339,571,475,394,472,580,
////    381,137,93,380,327,619,729,808,218,213,459,141,806,341,95,
////    382,568,124,750,193,749,706,843,79,199,317,329,768,198,100,
////    466,613,78,562,783,689,136,838,94,142,164,679,219,419,366,
////    418,423,77,89,523,259,683,312,555,20,470,684,123,458,453,833,
////    72,113,253,108,313,25,153,648,411,607,618,128,305,232,301,84,
////    56,264,371,46,407,360,38,99,176,710,114,578,66,372,653,
////    129,359,424,159,821,10,323,393,5,340,891,9,790,47,0,175,346,
////    236,26,172,147,574,561,32,294,429,724,755,398,787,288,299,
////    769,565,767,722,757,224,465,723,498,467,235,127,802,446,233,
////    544,482,800,318,16,532,801,441,554,173,60,530,713,469,30,
////    212,630,899,170,266,799,88,49,512,399,23,500,107,524,90,
////    194,143,135,192,206,345,148,71,119,101,563,870,158,254,214,
////    276,464,332,725,188,385,24,476,40,231,620,171,258,67,109,
////    844,244,187,388,701,690,50,7,850,479,48,522,22,154,12,659,
////    736,655,577,737,830,811,174,21,237,335,353,234,53,270,62,
////    182,45,177,245,812,673,355,556,612,166,204,54,248,365,226,
////    242,452,700,685,573,14,842,481,468,781,564,416,179,405,35,
////    819,608,624,367,98,643,448,2,460,676,440,240,130,146,184,
////    185,430,65,807,377,82,121,708,239,310,138,596,730,575,477,
////    851,797,247,27,85,586,307,779,326,494,856,324,827,96,748,
////    13,397,125,688,702,92,293,716,277,140,112,4,80,855,839,1,
////    413,347,584,493,289,696,19,751,379,76,73,115,6,590,183,734,
////    197,483,217,344,330,400,186,243,587,220,780,200,793,246,824,
////    41,735,579,81,703,322,760,720,139,480,490,91,814,813,163,
////    152,488,763,263,425,410,576,120,319,668,150,160,302,491,515,
////    260,145,428,97,251,395,272,252,18,106,358,854,485,144,550,
////    131,133,378,68,102,104,58,361,275,209,697,582,338,742,589,
////    325,408,229,28,304,191,189,110,126,486,211,547,533,70,215,
////    670,249,36,581,389,605,331,518,442,822
////  };
////
////  float scaleStep = pow(2.0, (float)(nOctaves)/NB_SCALES ); // 2 ^ ( (nOctaves-1) /nbScales)
////
////  const int nn[8] = {6, 6, 6, 6, 6, 6, 6, 1}; // number of points on each concentric circle (from outer to inner)
////  const float bigR = (float)(2.0 / 3.0); // bigger radius
////  const float smallR = (float)(2.0 / 24.0); // smaller radius
////  const float unitSpace = (float)((bigR - smallR) / 21.0 ); // define spaces between concentric circles (from center to outer: 1,2,3,4,5,6)
////  // radii of the concentric cirles (from outer to inner)
////  const float radius[8] = {bigR                 , bigR -  6 * unitSpace, bigR - 11 * unitSpace, bigR - 15 * unitSpace,
////                           bigR - 18 * unitSpace, bigR - 20 * unitSpace, smallR               , 0.0};
////  // sigma of pattern points (each group of 6 points on a concentric cirle has the same sigma)
////  const float sigma[8] = {float(radius[0] / 2.0), float(radius[1] / 2.0), float(radius[2] / 2.0), float(radius[3] / 2.0),
////                          float(radius[4] / 2.0), float(radius[5] / 2.0), float(radius[6] / 2.0), float(radius[6] / 2.0)};
////
////  for( int scaleIdx=0; scaleIdx < NB_SCALES; ++scaleIdx )
////  {
////      patternSizes[scaleIdx] = 0; // proper initialization
////      float scalingFactor = pow(scaleStep,scaleIdx); //scale of the pattern, scaleStep ^ scaleIdx
////
////      for( int orientationIdx = 0; orientationIdx < FREAK_NB_ORIENTATION; ++orientationIdx )
////      {
////          int pointIdx = 0;
////
////          for( size_t i = 0; i < 8; ++i )
////          {
////              for( int k = 0 ; k < nn[i]; ++k )
////              {
////                  float alpha = (float)(k) * 2 * (float)3.141592653589793238462643383279502884197169399375 / (float)(nn[i])
////                              + ((float)3.141592653589793238462643383279502884197169399375 / nn[i] * ( i % 2 )/*beta*/)
////                              + ((float)(orientationIdx * 2 * (float)3.141592653589793238462643383279502884197169399375 / (float)(FREAK_NB_ORIENTATION)) /* theta*/);
////
////                  // add the point to the look-up table
////                  cl_float3* point = &patternLookup[scaleIdx * FREAK_NB_ORIENTATION * FREAK_NB_POINTS + orientationIdx * FREAK_NB_POINTS + pointIdx];
////                  point->x = (float)(radius[i] * cos(alpha) * scalingFactor * patternScale); //x
////                  point->y = (float)(radius[i] * sin(alpha) * scalingFactor * patternScale); //y
////                  point->z = (float)(sigma[i] * scalingFactor * patternScale); //sigma
////
////                  // adapt the sizeList if necessary
////                  const int sizeMax = (int)(ceil((radius[i] + sigma[i]) * scalingFactor * patternScale)) + 1;
////                  if( patternSizes[scaleIdx] < sizeMax )
////                      patternSizes[scaleIdx] = sizeMax;
////
////                  ++pointIdx;
////              }
////          }
////      }
////  }
////
////  // build the list of orientation pairs
////  orientationPairs[0].w=0;   orientationPairs[0].x=3;   orientationPairs[1].w=1;   orientationPairs[1].x=4;    orientationPairs[2].w=2;   orientationPairs[2].x=5;
////  orientationPairs[3].w=0;   orientationPairs[3].x=2;   orientationPairs[4].w=1;   orientationPairs[4].x=3;    orientationPairs[5].w=2;   orientationPairs[5].x=4;
////  orientationPairs[6].w=3;   orientationPairs[6].x=5;   orientationPairs[7].w=4;   orientationPairs[7].x=0;    orientationPairs[8].w=5;   orientationPairs[8].x=1;
////
////  orientationPairs[9].w=6;   orientationPairs[9].x=9;   orientationPairs[10].w=7;  orientationPairs[10].x=10;  orientationPairs[11].w=8;  orientationPairs[11].x=11;
////  orientationPairs[12].w=6;  orientationPairs[12].x=8;  orientationPairs[13].w=7;  orientationPairs[13].x=9;   orientationPairs[14].w=8;  orientationPairs[14].x=10;
////  orientationPairs[15].w=9;  orientationPairs[15].x=11; orientationPairs[16].w=10; orientationPairs[16].x=6;   orientationPairs[17].w=11; orientationPairs[17].x=7;
////
////  orientationPairs[18].w=12; orientationPairs[18].x=15; orientationPairs[19].w=13; orientationPairs[19].x=16;  orientationPairs[20].w=14; orientationPairs[20].x=17;
////  orientationPairs[21].w=12; orientationPairs[21].x=14; orientationPairs[22].w=13; orientationPairs[22].x=15;  orientationPairs[23].w=14; orientationPairs[23].x=16;
////  orientationPairs[24].w=15; orientationPairs[24].x=17; orientationPairs[25].w=16; orientationPairs[25].x=12;  orientationPairs[26].w=17; orientationPairs[26].x=13;
////
////  orientationPairs[27].w=18; orientationPairs[27].x=21; orientationPairs[28].w=19; orientationPairs[28].x=22;  orientationPairs[29].w=20; orientationPairs[29].x=23;
////  orientationPairs[30].w=18; orientationPairs[30].x=20; orientationPairs[31].w=19; orientationPairs[31].x=21;  orientationPairs[32].w=20; orientationPairs[32].x=22;
////  orientationPairs[33].w=21; orientationPairs[33].x=23; orientationPairs[34].w=22; orientationPairs[34].x=18;  orientationPairs[35].w=23; orientationPairs[35].x=19;
////
////  orientationPairs[36].w=24; orientationPairs[36].x=27; orientationPairs[37].w=25; orientationPairs[37].x=28;  orientationPairs[38].w=26; orientationPairs[38].x=29;
////  orientationPairs[39].w=30; orientationPairs[39].x=33; orientationPairs[40].w=31; orientationPairs[40].x=34;  orientationPairs[41].w=32; orientationPairs[41].x=35;
////  orientationPairs[42].w=36; orientationPairs[42].x=39; orientationPairs[43].w=37; orientationPairs[43].x=40;  orientationPairs[44].w=38; orientationPairs[44].x=41;
////
////  for( unsigned m = NB_ORIENPAIRS; m--; )
////  {
////      float dx = patternLookup[orientationPairs[m].w].x - patternLookup[orientationPairs[m].x].x;
////      float dy = patternLookup[orientationPairs[m].w].y - patternLookup[orientationPairs[m].x].y;
////      float norm_sq = (dx * dx + dy * dy);
////      orientationPairs[m].y = round((dx / (norm_sq)) * 4096.0); //weight_dx
////      orientationPairs[m].z = round((dy / (norm_sq)) * 4096.0); //weight_dy
////  }
////
////  // build the list of description pairs
////  cl_uchar2 allPairs[(int)((FREAK_NB_POINTS * (FREAK_NB_POINTS + 1)) / 2)];
////  for( int i = 1, idx = 0; i < FREAK_NB_POINTS; ++i )
////  {
////      // (generate all the pairs)
////      for( int j = 0; j < i; ++j )
////      {
////          allPairs[idx++] = (cl_uchar2){(uchar)i, (uchar)j};
////      }
////  }
////  for( int i = 0; i < NB_PAIRS; ++i )
////      descriptionPairs[i] = allPairs[FREAK_DEF_PAIRS[i]];
////
////  int kpScaleIdx[overlap_kfs.back().second];
//
//
////  gpu_freak->reload_buf(1,0,img);
////  gpu_freak->reload_buf(1,1,integral_img);
////  gpu_freak->reload_buf(1,2, keypoints);
////  gpu_freak->reload_buf(1,4,patternSizes);
////  gpu_freak->reload_buf(1,5,orientationPairs);
////  gpu_freak->reload_buf(1,6,patternLookup);
////  gpu_freak->reload_buf(1,7,descriptionPairs);
////  gpu_freak->write_buf(1,8,nOctaves);
////  gpu_freak->reload_buf(1,9,kpScaleIdx);
////  gpu_freak->run(1);
////  cl_uchar descriptor[frame_number*300*64]={0};
////  gpu_freak->read(1,3,frame_number*300*64,descriptor);
//
//
//  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  for(size_t i=0; i<grid_.cells.size(); ++i)
//  {
//    // we prefer good quality points over unkown quality (more likely to match)
//    // and unknown quality over candidates (position not optimized)
//    if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame))
//      ++n_matches_;
//    if(n_matches_ > (size_t) Config::maxFts())
//      break;
//  }
//// SVO_STOP_TIMER("feature_align");
//}

    void Reprojector::reprojectMap(
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

        for(int i; i < matches.size(); i++)
        {
            auto fts = framel->fts_.begin();
            auto fts2 = fts;
            for(int f = 0; f < matches[i].queryIdx; f++)
                fts2++;
            if(reprojectPoint(frame, (*fts2)->point, keypoints_cur[i].pt.x, keypoints_cur[i].pt.y)){
                overlap_kfs.back().second++;
            }
            keypoints_cur[i].class_id = 0;

        }
        for(int i; i < keypoints_cur.size(); i++)
        {
            if(keypoints_cur[i].class_id == 0)
                continue;
            if(reprojectPoint(frame, nullptr, keypoints_cur[i].pt.x, keypoints_cur[i].pt.y))
                overlap_kfs.back().second++;
        }
        //SVO_STOP_TIMER("reproject_kfs");

        // Now project all point candidates
        // SVO_START_TIMER("reproject_candidates");
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
        } // unlock the mutex when out of scope
        //SVO_STOP_TIMER("reproject_candidates");
        // Now we go through each grid cell and select one point to match.
        // At the end, we should have at maximum one reprojected point per cell.
        //SVO_START_TIMER("feature_align");
        for(size_t i=0; i<grid_.cells.size(); ++i)
        {
            // we prefer good quality points over unkown quality (more likely to match)
            // and unknown quality over candidates (position not optimized)
            if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame))
                ++n_matches_;
            if(n_matches_ > (size_t) Config::maxFts())
                break;
        }
// SVO_STOP_TIMER("feature_align");
    }

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
    //std::cerr<<"n_trials_ : "<<n_trials_<<"\t"<<it->pt->type_<<"\t"<<it->pt->n_failed_reproj_<<"\t"<<it->pt->n_succeeded_reproj_<<std::endl;
    if(it->pt->type_ == Point::TYPE_DELETED)
    {
      it = cell.erase(it);
      continue;
    }
    /*bool found_match = true;
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

bool Reprojector::reprojectPoint(FramePtr frame, Point* point)
{
  Vector2d px(frame->w2c(point->pos_));
  if(frame->cam_->isInFrame(px.cast<int>(), 3)) // 8px is the patch size in the matcher
  {
    const int k = static_cast<int>(px[1]/grid_.cell_size)*grid_.grid_n_cols
                + static_cast<int>(px[0]/grid_.cell_size);
    grid_.cells.at(k)->push_back(Candidate(point, px));
    return true;
  }
  return false;
}

    bool Reprojector::reprojectPoint(FramePtr frame, Point* point, float x, float y)
    {
        if(point) {
            Vector2d px(x, y);
            if (frame->cam_->isInFrame(px.cast<int>(), 3)) // 8px is the patch size in the matcher
            {
                const int k = static_cast<int>(px[1] / grid_.cell_size) * grid_.grid_n_cols
                              + static_cast<int>(px[0] / grid_.cell_size);
                grid_.cells.at(k)->push_back(Candidate(point, px));
                return true;
            }
            return false;
        }
        else{
            Vector2d px(x, y);
            if (frame->cam_->isInFrame(px.cast<int>(), 3)) // 8px is the patch size in the matcher
            {
                const int k = static_cast<int>(px[1] / grid_.cell_size) * grid_.grid_n_cols
                              + static_cast<int>(px[0] / grid_.cell_size);
                Point* pnt = new Point(Vector3d(frame->cam_->cam2world(x, y)));
                pnt->type_ = Point::TYPE_UNKNOWN;
                grid_.cells.at(k)->push_back(Candidate(pnt, px));
                delete pnt;
                return true;
            }
            return false;
        }
    }

} // namespace vio
