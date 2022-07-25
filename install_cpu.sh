#!/bin/sh
pip install paddlepaddle==2.3.0
pip uninstall protobuf -y
pip install --no-binary protobuf protobuf
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
pip install onnxruntime==1.11.1
pip install termcolor==1.1.0
pip install gdown==4.5.1
python weights/download.py
echo succeeded
