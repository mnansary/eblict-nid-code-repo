#!/bin/sh
 conda install cudatoolkit=11.2 cudnn=8.2 -c conda-forge 
conda install paddlepaddle-gpu==2.3.0  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/  
pip uninstall protobuf
pip install --no-binary protobuf protobuf==3.18.0
pip install opencv-python==4.6.0.66
pip install shapely==1.8.2
pip install pyclipper==1.3.0.post3
pip install scikit-image==0.19.3
pip install imgaug==0.4.0
pip install lmdb==1.3.0
pip install tqdm==4.64.0
pip install attrdict==2.0.1
pip install git+https://github.com/mnansary/PaddleOCR.git --verbose
pip install torch==1.11.0 torchvision==0.12.0  --extra-index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime-gpu==1.11
pip install termcolor==1.1.0
pip install gdown==4.5.1
python weights/download.py
python setup_check.py
echo succeeded
