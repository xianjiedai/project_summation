# UNet Implementation with Keras
import os
import numpy as np

from tensorflow.keras.optimizers import *
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers import Dropout, BatchNormalization, Activation

from keras.layers import concatenate
from keras.losses import binary_crossentropy
#from metric_loss import f1


# Convolution Block for UNet
def conv_block(inputs, n_filter, k_size=3, activation='relu'):
    x = Conv2D(n_filter, kernel_size=k_size, padding="same", kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(n_filter, kernel_size=k_size, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x


# Dropout Block
def dropout(x, d_out=True, d_rate=0.5):
    if d_out is True:
        x = Dropout(d_rate)(x)
    return x


# Standard UNet Model
def unet(size=(None, None, 3), n_filter=64, activation='relu', d_out=True, d_rate=0.5,
         loss=binary_crossentropy, lr_rate=1e-4, model_path=None):
    """
    Arguments:
        size: size of input images
        n_filter: number of filters of the 1st layer
        activation: activation function to use
        d_out: flag of dropout layer
        d_rate: dropout rate
        loss: loss function to use (default: binary_crossentropy)
        lr_rate: learning rate of Adam Optimizer (default: 1e-4)
        model_path: load pretrained weights if exists

    Return:
        UNet model to train
    """

    inputs = Input(shape=size)

    # Convolution
    conv1 = conv_block(inputs, n_filter, k_size=3, activation=activation)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, n_filter * 2, k_size=3, activation=activation)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, n_filter * 4, k_size=3, activation=activation)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, n_filter * 8, k_size=3, activation=activation)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottom & Deconvolution
    conv5 = conv_block(pool4, n_filter * 16, k_size=3, activation=activation)
    dconv1 = Conv2DTranspose(n_filter * 8, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv5)
    merge1 = concatenate([conv4, dconv1], axis=3)
    merge1 = dropout(merge1, d_out, d_rate)
    conv6 = conv_block(merge1, n_filter * 8, k_size=3, activation=activation)

    dconv2 = Conv2DTranspose(n_filter * 4, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv6)
    merge2 = concatenate([conv3, dconv2], axis=3)
    merge2 = dropout(merge2, d_out, d_rate)
    conv7 = conv_block(merge2, n_filter * 4, k_size=3, activation=activation)

    dconv3 = Conv2DTranspose(n_filter * 2, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv7)
    merge3 = concatenate([conv2, dconv3], axis=3)
    merge3 = dropout(merge3, d_out, d_rate)
    conv8 = conv_block(merge3, n_filter * 2, k_size=3, activation=activation)

    dconv4 = Conv2DTranspose(n_filter, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv8)
    merge4 = concatenate([conv1, dconv4], axis=3)
    merge4 = dropout(merge4, d_out, d_rate)
    conv9 = conv_block(merge4, n_filter, k_size=3, activation=activation)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=lr_rate), loss=loss, metrics=[f1, 'accuracy'])
    model.summary()

    # Load previous model weights if exist
    if model_path:
        model.load_weights(filepath=model_path)

    return model
