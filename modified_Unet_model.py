# -*- coding:utf-8 -*-
import tensorflow as tf

def modified_Unet(input_shape=(512, 512, 3), classes=3):

    #
    h = inputs = tf.keras.Input(input_shape)

    #
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    h = tf.keras.layers.SpatialDropout2D(0.1)(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    block_1 = h

    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    h = tf.keras.layers.SpatialDropout2D(0.1)(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    block_2 = h

    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    h = tf.keras.layers.SpatialDropout2D(0.1)(h)
    
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)

    block_3 = h

    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    h = tf.keras.layers.SpatialDropout2D(0.1)(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)

    block_4 = h

    h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h)

    h = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    h = tf.keras.layers.SpatialDropout2D(0.1)(h)

    h = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=2, strides=2)(h)

    h = tf.concat([h, block_4], -1)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    h = tf.keras.layers.SpatialDropout2D(0.1)(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=2, strides=2)(h)
    h = tf.concat([h, block_3], -1)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    h = tf.keras.layers.SpatialDropout2D(0.1)(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2)(h)
    h = tf.concat([h, block_2], -1)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    h = tf.keras.layers.SpatialDropout2D(0.1)(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2)(h)
    h = tf.concat([h, block_1], -1)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)
    h = tf.keras.layers.SpatialDropout2D(0.1)(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ELU()(h)

    h = tf.keras.layers.Conv2D(filters=classes, kernel_size=1)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)
