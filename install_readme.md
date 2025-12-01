In case libero dependency install fails because of egl_probe, manually install it from source:


git clone https://github.com/StanfordVL/egl_probe.git
cd egl_probe/egl_probe
Add "cmake_minimum_required(VERSION 3.5)" at top of CMakeLists.txt
pip install -e .
