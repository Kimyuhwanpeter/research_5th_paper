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

def Depth_wise_Unet(input_shape=(512, 512, 3), pretrained=True):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, use_bias=True, name="conv1")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, use_bias=True, name="conv2")(h)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU(max_value=6.)(conv2)
    
    h = tf.keras.layers.MaxPool2D((2,2), strides=2, name="maxpool1")(conv2)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, use_bias=True, name="conv3")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, use_bias=True, name="conv4")(h)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.ReLU(max_value=6.)(conv4)

    h = tf.keras.layers.MaxPool2D((2,2), strides=2, name="maxpool2")(conv4)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, use_bias=True, name="conv5")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, use_bias=True, name="conv6")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv7 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, use_bias=True, name="conv7")(h)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.ReLU(max_value=6.)(conv7)

    h = tf.keras.layers.MaxPool2D((2,2), strides=2, name="maxpool3")(conv7)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=True, name="conv8")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=True, name="conv9")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv10 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=True, name="conv10")(h)
    conv10 = tf.keras.layers.BatchNormalization()(conv10)
    conv10 = tf.keras.layers.ReLU(max_value=6.)(conv10)

    h = tf.keras.layers.MaxPool2D((2,2), strides=2, name="maxpool4")(conv10)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=True, name="conv11")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=True, name="conv12")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU(max_value=6.)(h)

    h = tf.pad(h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    conv13 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=True, name="conv13")(h)
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

    crop_weed_h = h[:, :, :, 0:64] * tf.nn.sigmoid(object_h)
    crop_weed_h = tf.pad(crop_weed_h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    crop_weed_h = tf.keras.layers.Conv2D(filters=16, kernel_size=3, use_bias=False)(crop_weed_h)
    crop_weed_h = tf.keras.layers.BatchNormalization()(crop_weed_h)
    crop_weed_h = tf.keras.layers.ReLU(max_value=6.)(crop_weed_h)
    crop_weed_h = tf.pad(crop_weed_h, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    crop_weed_h = tf.keras.layers.Conv2D(filters=1, kernel_size=3)(crop_weed_h)

    final_output = tf.concat([crop_weed_h, object_h], -1)

    model = tf.keras.Model(inputs=inputs, outputs=final_output)

    backbone = tf.keras.applications.VGG16(input_shape=(224, 224, 3))

    if pretrained:
        model.get_layer("conv1").set_weights(backbone.get_layer("block1_conv1").get_weights())
        model.get_layer("conv2").set_weights(backbone.get_layer("block1_conv2").get_weights())
        model.get_layer("conv3").set_weights(backbone.get_layer("block2_conv1").get_weights())
        model.get_layer("conv4").set_weights(backbone.get_layer("block2_conv2").get_weights())
        model.get_layer("conv5").set_weights(backbone.get_layer("block3_conv1").get_weights())
        model.get_layer("conv6").set_weights(backbone.get_layer("block3_conv2").get_weights())
        model.get_layer("conv7").set_weights(backbone.get_layer("block3_conv3").get_weights())
        model.get_layer("conv8").set_weights(backbone.get_layer("block4_conv1").get_weights())
        model.get_layer("conv9").set_weights(backbone.get_layer("block4_conv2").get_weights())
        model.get_layer("conv10").set_weights(backbone.get_layer("block4_conv3").get_weights())
        model.get_layer("conv11").set_weights(backbone.get_layer("block5_conv1").get_weights())
        model.get_layer("conv12").set_weights(backbone.get_layer("block5_conv2").get_weights())
        model.get_layer("conv13").set_weights(backbone.get_layer("block5_conv3").get_weights())
        print("================================================================================")
        print("** The encoder part is used based on the pretrained VGG-16 (imagenet dataset).**")
        print("================================================================================")
    else:
        print("================================================================================")
        print("** The encoder part is used based on the VGG-16 (Train from scretch).**")
        print("================================================================================")

    return tf.keras.Model(inputs=inputs, outputs=final_output)
