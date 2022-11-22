#!/bin/sh
conda install cudatoolkit 
conda install cudnn
python -m pip install paddlepaddle-gpu==2.3.0.post111 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
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
pip install bnunicodenormalizer
sudo chmod -R 777 weights/
sudo chmod -R 777 tests/
sudo touch logs.log
sudo chmod 777 logs.log
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
python weights/download.py
python setup_check.py 
echo succeeded
