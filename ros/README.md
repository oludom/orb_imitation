
# install instructions

please install mavros for your ros distro as described here: 
https://github.com/mavlink/mavros/blob/master/mavros/README.md#installation

afterwards install dependencies with:   
wstool init src  
wstool merge -t src/ packages.rosinstall  
wstool update -t src/ -j8

and build with:  
catkin build
