# DFNN

The DFNN (Deep Fourier neutral network) we propose is a neural network that facilitates defocus inference within a single shot for a light sheet system. In this repository, codes used for training the network and defocus inference, some testing data, the trained models and the corresponding licenses are uploaded.

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

Files named predict_3grad.py and predict_2grad.py are programmed for the defocus inference of single-shot images illuminated by multi-depth lattice patterns with three gradients and two gradients, respectively. In addition, data of mitochondria illuminated by the pattern of 2 gradients with Dz 0f 0.3 um (G2#0.3um) and F-actins illuminated by the pattern G3#0.5um are provided in folder ./dataset/test.

### Training

The DFNN can be trained by the file <train_DFNN.py> with our open-source datasets.

### License

This repository is released under the MIT License (refer to the LICENSE file for details).
