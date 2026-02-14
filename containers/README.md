# LRZ Cluster

To run hlrp on the lrz cluster you need a enroot container based on nvidia ngc. You can build the dockerfile in this package locally, push it to docker hub and then execute import and create on the cluster with your user specific data to get the .sqsh file.

## Lerobot in python3.12 install on workstation
sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
conda install -c conda-forge cmake ninja
sudo apt update
sudo apt install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev
pip install --no-build-isolation egl_probe hf-egl-probe -> failed
conda install -c conda-forge "cmake<4" ninja
pip install --no-build-isolation egl_probe 
pip install -e ".[libero]"
pip uninstall torch torchvision torchaudio
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
