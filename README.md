# DFNN

The DFNN (Deep Fourier neutral network) we propose is a neural network that facilitates defocus inference within a single shot for a light sheet system. In this repository, codes used for training the network and defocus inference, some testing data, the trained models and corresponding licenses are uploaded.

## Contents
- [Enviroment](https://github.com/ZJUOPTKuangLab/DFNN/edit/main/README.md#enviroment)
- [Dataset](https://github.com/ZJUOPTKuangLab/DFNN/edit/main/README.md#Dataset)
- [Usage](https://github.com/ZJUOPTKuangLab/DFNN/edit/main/README.md#Usage)
- [License](https://github.com/ZJUOPTKuangLab/DFNN/edit/main/README.md#License)

## Enviroment
-   Windows 10
-   CUDA 11.2
-   Python 3.7.10
-   Tensorflow 1.14.0
-   Keras 2.2.5
-   GPU GeForce RTX 2080Ti

## Dataset



## Usage
### Test 
Files named predict_3grad.py and predict_2grad.py are programming for testing three gradients or two gradients mutil-focal data respectively. We provide mitochodra illuminated by 2 gradients pattern and F-actin illuminated by 3 gradients pattern  data, and placed in ./dataset/test,
### Training

You can train DFNN network by downloading our open source dataset and running  train_DFNN.py

## License

This repository is released under the MIT License (refer to the LICENSE file for details).
