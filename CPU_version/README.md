SVO
===

This code implements a semi-direct monocular visual odometry pipeline which has been integrated with GTSAM to infused IMU as well. 


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

The API is documented here: http://uzh-rpg.github.io/rpg_svo/doc/
The GTSAM documentation  https://gtsam.org/tutorials/

#### Instructions

See the Wiki for more instructions. https://github.com/uzh-rpg/rpg_svo/wiki and https://gtsam.org/get_started/

#### Contributing

You are very welcome to contribute to SVO by opening a pull request via Github.
I try to follow the ROS C++ style guide http://wiki.ros.org/CppStyleGuide
