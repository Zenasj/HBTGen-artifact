import torch

# apt update
# apt dist-upgrade
# apt install libnuma-dev
# reboot
# wget -q -O - https://repo.radeon.com/rocm/apt/4.0.1/rocm.gpg.key | apt-key add -
# echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/4.0.1/ xenial main' | tee /etc/apt/sources.list.d/rocm.list
# apt update
# apt install rocm-dkms 
# reboot
# apt install rocm-libs
# reboot
# echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/rocprofiler/bin:/opt/rocm/opencl/bin' | tee -a /etc/profile.d/rocm.sh
# usermod -a -G video <user>
# pip3 install torch -f https://download.pytorch.org/whl/rocm4.0.1/torch_stable.html
# pip3 install ninja
# pip3 install 'git+https://github.com/pytorch/vision.git@v0.9.1'
# reboot