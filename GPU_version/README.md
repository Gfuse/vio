SVO
===

This code implements a semi-direct monocular visual odometry pipeline which has been integrated with GTSAM to infused IMU as well. The feature extraction part is implemented in OpenCL to take advantage of GPU.


Paper svo: http://rpg.ifi.uzh.ch/docs/ICRA14_Forster.pdf
Paper Svo+GTSAM: https://arxiv.org/abs/1512.02363


#### Disclaimer

SVO has been tested under ROS Groovy, Hydro and Indigo with Ubuntu 12.04, 13.04 and 14.04. This is research code, any fitness for a particular purpose is disclaimed.


#### Licence

The source code is released under a GPLv3 licence. A closed-source professional edition is available for commercial purposes. In this case, please contact the authors for further info.


#### Citing

If you use SVO in an academic context, please cite the following publication:

    @inproceedings{Forster2014ICRA,
      author = {Forster, Christian and Pizzoli, Matia and Scaramuzza, Davide},
      title = {{SVO}: Fast Semi-Direct Monocular Visual Odometry},
      booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
      year = {2014}
    }
    
    
#### Documentation
OpenCL documentation

The API is documented here: http://uzh-rpg.github.io/rpg_svo/doc/

The GTSAM documentation  https://gtsam.org/tutorials/

#### Instructions
Please be sure that you have the OpenCL driver installed

First install gtsam library => https://github.com/borglab/gtsam

Then follow the SVO instalation

See the Wiki for more instructions. https://github.com/uzh-rpg/rpg_svo/wiki and https://gtsam.org/get_started/


## Setting

You can edit the algorithm parameters in the follwoing file, please look at the config file to underestand the meaning of the parameters (CPU_version/svo/include/svo/config.h)

CPU_version/vio_svo/param/vo_fast.yaml

# the GPU_version

Dose not have dependency with GTSAM, the dependencies are:
1. OpenCV 3
2. Eigen
3. Sophus     https://github.com/strasdat/Sophus.git
4. Boost
5. OpenCL

if make fails in the build of Sophus with the error lvalue required as left operand of assignment in so2.cpp, see this issue: https://github.com/uzh-rpg/rpg_svo/issues/237




