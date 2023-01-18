import argparse
import glob
import os
from time import *

import cv2
import imageio
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
from keras import optimizers
from keras.layers import Conv2D, Activation, UpSampling2D, Lambda, Dropout, MaxPooling2D, multiply, add, Conv2DTranspose
from keras.layers import Input, add, multiply, Lambda, BatchNormalization, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import AveragePooling2D, Conv2D
from keras.layers.core import Dense, Activation, Flatten
from keras.models import *
from keras.models import Model
from numpy import fft

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
#     # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
       print(e)

def prctile_norm(x, min_prc=0, max_prc=100):
    # y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
    y = x / np.percentile(x, max_prc)
    y[y > 1] = 1
    y[y < 0] = 0
    return y

def prctile_norm_img(x, min_prc=0, max_prc=100):
    datamin = np.percentile(x, min_prc)
    datamax = np.percentile(x, max_prc)
    y=(x-datamin)/(datamax-datamin)

    # y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
    # y = x / np.percentile(x, max_prc)
    y[y > 1] = 1
    y[y < 0] = 0
    return y

def cal_xs(img, xs):
    data = np.array(img[:, int(128)])
    data = prctile_norm(data)
    real_xs = xs / data[int(69)]
    return real_xs

def initial():
    im = Input(shape=(8, 1, 3))
    out = Input(shape=(8, 1, 3))
    x0 = Conv2D(8, kernel_size=1, padding='same')(im)
    x0 = BatchNormalization()(x0)
    x0 = Activation('relu')(x0)
    y0 = Flatten(input_shape=(1, 1))(x0)
    y0 = Dropout(0.5)(y0)
    y1 = Dense(16)(y0)
    y1 = Activation('relu')(y1)
    y2 = Dense(16)(y1)
    y2 = Activation('relu')(y2)
    output = Dense(1)(y2)
    global g
    g= Model(inputs=[im, out], outputs=output)
    optimizer = optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999)
    g.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    g.load_weights("trained_models/grad3_0.5um_50power/weights.best.h5")

def find_center(img):
    pic_size=256
    center_find = np.sum(img,axis=0)
    temp_center = np.where(center_find == np.max(center_find))
    # print(len(temp_center))
    temp_center = temp_center[0]
    if len(temp_center) > 1:
        temp_center=np.mean(temp_center)

    # print(temp_center)
    if temp_center < 129:
        temp_center = (pic_size / 2) + 1
    if temp_center > 384:
        temp_center = 512 - (pic_size / 2)
    return temp_center

def cut_fft(img, depth):
    # print(img.shape)
    # img = prctile_norm_img(img)
    center=find_center(img[int(255 * depth + 34-1):int(255 * depth + 256 + 34-1), :])
    # center=256
    im = img[int(255 * depth + 34-1):int(255 * depth + 256 + 34-1), int(center - 128-1):int(center + 128-1)]
    fft_im = fft.fftshift(fft.fft2(fft.fftshift(im)))
    im_fft = np.log2(np.abs(fft_im) + 1)
    im_fft = im_fft / np.max(im_fft)
    im_fft=np.array(im_fft*256).astype(np.int)
    return im_fft

def predict(onfocus, out):
    cur_img = []
    # xs = [0.85, 1, 0.85]
    xs = [0.73, 1, 0.93]
    # xs=[0.975,1,0.885]
    rz=[]
    for j in range(0, 3):
        img = cut_fft(out, j)
        path_on = cut_fft(onfocus, j)
        a = cal_xs(path_on, xs[j])
        img = img[:, 128]
        img = prctile_norm(img)
        rz.append(img[69]*0.95)
        img = img[65:73] * a
        img = img[:, np.newaxis]
        cur_img.append(img)
    img = np.array(cur_img).transpose((1, 2, 0))
    img = img[np.newaxis, :, :, :]
    pr1 = np.squeeze(g.predict([img, img]))
    return pr1,rz


# creat model&load weights
initial()


# select processing mode
# 0: predict single image defocus amount
# 1: process focal-centered image sequence
process_single_img=0
if process_single_img:
    data_path = 'data/test/Microtubules/single/cell11G3Dz0p5umOffP0p4um'
    data_path_on = 'data/test/Microtubules/single/cell11G3Dz0p5umON(1)'
    all_img = []
    onfocus = imageio.imread(data_path_on + '.tif').astype(np.float)
    all_img.append(onfocus)
    out = imageio.imread(data_path + '.tif').astype(np.float)
    all_img.append(out)
    all_img = prctile_norm_img(all_img)
    onfocus = all_img[0, :, :]
    out = all_img[1, :, :]
    val, rz = predict(onfocus, out)
    print(-val)
else:
    depth=[]
    rz1=[]
    rz2=[]
    rz3=[]
    data_path='data/test/Microtubules/stack/cell11G3Dz0p5um_ch0_stack0001_488nm_0004525msec_0007559766msecAbs00'
    star=36
    end=66
    for i in range(star, end):
        all_img = []
        onfocus = imageio.imread(data_path + str(51) + '.tif').astype(np.float)
        all_img.append(onfocus)
        out = imageio.imread(data_path + str(i) + '.tif').astype(np.float)
        all_img.append(out)
        all_img = prctile_norm_img(all_img)
        onfocus = all_img[0, :, :]
        out = all_img[1, :, :]
        val,rz= predict(onfocus, out)
        depth.append(val)
    d=np.linspace(star, end, end-star)
    real=(d-51)*0.05
    # print(max(abs(depth-real)))
    fig, ax = plt.subplots()
    ax.plot(real,real,label='real')
    ax.scatter(real,depth,s=60, c='hotpink', marker='.',label='predict')
    plt.xlabel('True defocus(um)')
    plt.ylabel('Predicted defocus(um)')
    plt.legend()
    plt.show()

