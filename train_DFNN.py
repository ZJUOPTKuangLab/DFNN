import argparse
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import datetime
from keras.callbacks import TensorBoard
import glob
import os
from utils.data_loader import data_loader_focus, prctile_norm
import tensorflow.compat.v1 as tf
from model import focus_net, fca_net, fca_classify
import cv2
import keras
from tensorflow.compat.v1 import ConfigProto
from lr_controller import ReduceLROnPlateau
import imageio
import keras.backend as K
from keras.models import *
from keras.layers import Conv2D, Activation, UpSampling2D, Lambda, Dropout, MaxPooling2D, multiply, add, Conv2DTranspose
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Input, add, multiply, Lambda, BatchNormalization, MaxPooling2D
from keras.layers.convolutional import AveragePooling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import LeakyReLU
from model.common import fft2d, fftshift2d, gelu, pixel_shiffle, conv_block2d, global_average_pooling2d
from keras.layers.advanced_activations import LeakyReLU
import math
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--gpu_memory_fraction", type=float, default=0.2)
parser.add_argument("--mixed_precision_training", type=int, default=1)
parser.add_argument("--norm_flag", type=int, default=1)
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--validate_interval", type=int, default=500)
parser.add_argument("--validate_num", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--pic_dep", type=float, default=0.5)
parser.add_argument("--start_lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--load_weights", type=int, default=0)
parser.add_argument("--optimizer_name", type=str, default="adam")
parser.add_argument("--patch_height", type=int, default=8)
parser.add_argument("--patch_width", type=int, default=1)
parser.add_argument("--input_channels", type=int, default=3)
parser.add_argument("--data_dir", type=str, default="dataset/train/depth3_0.5um")
parser.add_argument("--save_weights_dir", type=str, default="trained_models/")
parser.add_argument("--model_name", type=str, default="MT_10000epoch")
args = parser.parse_args()
gpu_id = str(args.gpu_id)
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision_training = str(args.mixed_precision_training)
data_dir = args.data_dir
save_weights_dir = args.save_weights_dir
validate_interval = args.validate_interval
batch_size = args.batch_size
dep=args.pic_dep
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
patch_height = args.patch_height
patch_width = args.patch_width
input_channels = args.input_channels
norm_flag = args.norm_flag
validate_num = args.validate_num
iterations = args.iterations
load_weights = args.load_weights
optimizer_name = args.optimizer_name
model_name = args.model_name
scale_factor = 1
case=input_channels

# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = mixed_precision_training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
data_name = data_dir.split('/')[-1]

def stand(data):
    me=data.mean(axis=0)
    st=data.std(axis=0)
    std_data=(data-me)/st
    return std_data


if input_channels == 1:
    save_weights_name = model_name + data_name
    cur_data_loader = data_loader_focus
    train_images_path = data_dir + '/train/'
    validate_images_path = data_dir + '/test/'
else:
    save_weights_name = model_name + data_name
    cur_data_loader = data_loader_focus
    train_images_path = data_dir + '/train_FFT/'
    validate_images_path = data_dir + '/test_FFT/'
    validate_images_path = 'G:/Yezitong/20221207/Autofocus/80Power/test_FFT/'
save_weights_path = save_weights_dir + '/' + save_weights_name + '/'

if not os.path.exists(save_weights_path):
    os.mkdir(save_weights_path)


def loss_mse(y_true, y_pred):
    mae_para = 0
    mse_para = 1
    ssim_para = 1
    # nomolization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))
    mse_loss = mse_para * K.mean(K.square(y - x))
    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    return mse_loss + ssim_loss


def read(x, stan, psf):
    out = tf.nn.conv2d(stan, psf, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
    out = tf.math.divide(tf.math.subtract(out, tf.reduce_min(out)),
                         tf.math.subtract(tf.reduce_max(out), tf.reduce_min(out)))
    return out



def loss(im, out):
    def loss_psf(y_true, y_pred):
        ssim_loss = (1 - K.mean(tf.image.ssim(out, im, 1)))
        loss = K.mean(K.square(out - im)) * 0 + K.mean(K.abs(y_true - y_pred)) + ssim_loss *0
        return loss
    return loss_psf


# --------------------------------------------------------------------------------
#                              define DFNN model
# --------------------------------------------------------------------------------
im = Input(shape=(patch_height, patch_width, input_channels))
out = Input(shape=(patch_height, patch_width, input_channels))
x0 = Conv2D(8, kernel_size=1, padding='same')(im)
x0 = BatchNormalization()(x0)
x0 = Activation('relu')(x0)
y0 = Flatten(input_shape=(1, 1))(x0)
y0=Dropout(0.5)(y0)
y1 = Dense(16)(y0)
y1 = Activation('relu')(y1)
y2 = Dense(16)(y1)
y2 = Activation('relu')(y2)
output = Dense(1)(y2)
g = Model(inputs=[im, out], outputs=output)
#optimizer select
optimizer_g = optimizers.Adam(lr=start_lr, beta_1=0.9, beta_2=0.999)
g.compile(loss='mse', optimizer='rmsprop')
lr_controller = ReduceLROnPlateau(model=g, factor=lr_decay_factor, patience=10, mode='min', min_delta=1e-4,
                                  cooldown=0, min_lr=start_lr * 0.1, verbose=1)

# --------------------------------------------------------------------------------
#                                 about Tensorboard
# --------------------------------------------------------------------------------
log_path = save_weights_path + 'graph'
if not os.path.exists(log_path):
    os.mkdir(log_path)
callback = TensorBoard(log_path)
callback.set_model(g)
train_names = 'training_loss'
val_names = ['val_MSE', 'val_SSIM', 'val_PSNR', 'val_NRMSE']


def write_log(callback, names, logs, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = logs
    summary_value.tag = names
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

validate_mse = [np.Inf]
# --------------------------------------------------------------------------------
#                             Sample and validate
# --------------------------------------------------------------------------------
def Validate(iter):
    validate_path = glob.glob(validate_images_path + '*')
    validate_path.sort()
    if validate_num < validate_path.__len__():
        validate_path = validate_path[0:validate_num]
    mses, nrmses, psnrs, ssims = [], [], [], []
    for i in range(0, validate_num):
        [img, gt, cell] = cur_data_loader(validate_images_path, 27,5,
                                              patch_width, 1,dep,case, norm_flag=norm_flag)
        output = np.squeeze(g.predict([img, img]))
        mses.append(abs(output - gt))
    g.save_weights(save_weights_path + 'weights'+str(iter)+'.latest.h5')
    if min(validate_mse) > np.mean(mses):
        g.save_weights(save_weights_path + 'weights.best.h5')
    validate_mse.append(np.mean(mses))
    curlr = lr_controller.on_epoch_end(iter, np.mean(nrmses))
    write_log(callback, val_names[0], np.mean(mses), iter)
    write_log(callback, 'lr', curlr, iter)
    print("ok")


# --------------------------------------------------------------------------------
#                                    training
# --------------------------------------------------------------------------------
start_time = datetime.datetime.now()
loss_record = []
validate_nrmse = [np.Inf]
lr_controller.on_train_begin()
images_path = glob.glob(train_images_path + '/*')

for it in range(iterations):
    input_g, gt_g, cell = cur_data_loader(train_images_path, 1, 93, patch_width,
                                                   batch_size,dep,case, norm_flag=norm_flag)
    print(input_g.shape)
    loss_generator = g.train_on_batch([input_g,input_g], gt_g)
    loss_record.append(loss_generator)
    elapsed_time = datetime.datetime.now() - start_time
    print("%d epoch: time: %s, g_loss = %s" % (it + 1, elapsed_time, loss_generator))
    if (it + 1) % validate_interval == 0:
        Validate(it + 1)
        write_log(callback, train_names, np.mean(loss_record), it + 1)
        loss_record = []
