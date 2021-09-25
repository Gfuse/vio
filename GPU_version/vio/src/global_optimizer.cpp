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


    BA_Glob::BA_Glob(Map& map) :map_(map),seeds_updating_halt_(false),thread_(NULL)
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

    void BA_Glob::updateLoop()
    {
        while(!boost::this_thread::interruption_requested())
        {
            usleep(20000);
        }
    }

} // namespace vio
