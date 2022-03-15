# -*- coding:utf-8 -*-
import tensorflow as tf

def residual_block(input):

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=1, use_bias=False)(input)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]])
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=1, use_bias=False)(input)
    h = tf.keras.layers.BatchNormalization()(h)
    
    return h + input

def Depth_wise_Unet(input_shape=(512, 512, 3)):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False, name="conv1")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False, name="conv2")(h)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU(max_value=6.)(conv2)
    
    h = tf.keras.layers.MaxPool2D((2,2), strides=2, name="maxpool1")(conv2)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, use_bias=False, name="conv3")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, use_bias=False, name="conv4")(h)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.ReLU(max_value=6.)(conv4)

    h = tf.keras.layers.MaxPool2D((2,2), strides=2, name="maxpool2")(conv4)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, use_bias=False, name="conv5")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, use_bias=False, name="conv6")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv7 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, use_bias=False, name="conv7")(h)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.ReLU(max_value=6.)(conv7)

    h = tf.keras.layers.MaxPool2D((2,2), strides=2, name="maxpool3")(conv7)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=False, name="conv8")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=False, name="conv9")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv10 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=False, name="conv10")(h)
    conv10 = tf.keras.layers.BatchNormalization()(conv10)
    conv10 = tf.keras.layers.ReLU(max_value=6.)(conv10)

    h = tf.keras.layers.MaxPool2D((2,2), strides=2, name="maxpool4")(conv10)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=False, name="conv11")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=False, name="conv12")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv13 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=False, name="conv13")(h)
    conv13 = tf.keras.layers.BatchNormalization()(conv13)
    conv13 = tf.keras.layers.ReLU(max_value=6.)(conv13)

    h = tf.keras.layers.MaxPool2D((2,2), strides=2, name="maxpool5")(conv13)

    for i in range(4):
        h = residual_block(h)

    h = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.concat([h, conv13], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=1, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.concat([h, conv10], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=1, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.concat([h, conv7], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=1, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.concat([h, conv4], -1)
    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)
    h = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.concat([h, conv2], -1)

    object_h = h[:, :, :, 64:96]
    object_h = tf.pad(object_h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    object_h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, use_bias=False)(object_h)
    object_h = tf.keras.layers.BatchNormalization()(object_h)
    object_h = tf.keras.layers.ReLU(max_value=6.)(object_h)
    object_h = tf.pad(object_h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    object_h = tf.keras.layers.Conv2D(filters=1, kernel_size=3)(object_h)

    crop_h = h[:, :, :, 0:32] * tf.nn.sigmoid(object_h)
    crop_h = tf.pad(crop_h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    crop_h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, use_bias=False)(crop_h)
    crop_h = tf.keras.layers.BatchNormalization()(crop_h)
    crop_h = tf.keras.layers.ReLU(max_value=6.)(crop_h) * object_h
    crop_h = tf.pad(crop_h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    crop_h = tf.keras.layers.Conv2D(filters=1, kernel_size=3)(crop_h)

    weed_h = h[:, :, :, 32:64] * tf.nn.sigmoid(object_h)
    weed_h = tf.pad(weed_h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    weed_h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, use_bias=False)(weed_h)
    weed_h = tf.keras.layers.BatchNormalization()(weed_h)
    weed_h = tf.keras.layers.ReLU(max_value=6.)(weed_h) * object_h
    weed_h = tf.pad(weed_h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    weed_h = tf.keras.layers.Conv2D(filters=1, kernel_size=3)(weed_h)

    final_output = tf.concat([crop_h, weed_h, object_h], -1)

    return tf.keras.Model(inputs=inputs, outputs=final_output)

model = Depth_wise_Unet()
model.summary()
m = tf.keras.applications.MobileNetV2
