# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import get_submodules_from_kwargs
from _common_blocks import Conv2dBn
from backbones_factory import Backbones
from _utils import freeze_model, filter_keras_submodules

import tensorflow as tf
# https://github.com/qubvel/segmentation_models
def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    #concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    concat_axis = 3

    def wrapper(input_tensor, skip=None):
        x = tf.keras.layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = tf.keras.layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    # concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    concat_axis = bn_axis = 3

    def layer(input_tensor, skip=None):

        x = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = tf.keras.layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = tf.keras.layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer

def DecoderTransposeX2Block_object(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose_object'.format(stage)
    bn_name = 'decoder_stage{}a_bn_object'.format(stage)
    relu_name = 'decoder_stage{}a_relu_object'.format(stage)
    conv_block_name = 'decoder_stage{}b_object'.format(stage)
    concat_name = 'decoder_stage{}_concat_object'.format(stage)

    # concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    concat_axis = bn_axis = 3

    def layer(input_tensor, skip=None):

        x = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = tf.keras.layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = tf.keras.layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer

# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], tf.keras.layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)
    x_ = x
    decoder_filters_ = (512, 256, 128, 64, 32)
    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = DecoderTransposeX2Block_object(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x_ = DecoderTransposeX2Block(decoder_filters_[i], stage=i, use_batchnorm=use_batchnorm)(x_, skip)
    x_ = x_ * tf.reduce_mean(tf.nn.sigmoid(x), -1, keepdims=True)
    # model head (define number of output classes)
    x = tf.keras.layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform'
    )(x)

    x_ = tf.keras.layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform'
    )(x_)

    x = tf.concat([x_, x], -1, name="final_conv")


    # create keras model instance
    model = tf.keras.Model(input_, x)

    return model

def fix_Unet(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        **kwargs
):

    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )


    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model

# from model_profiler import model_profiler

# model = Unet(input_shape=(512, 512, 3))
# model.summary()
# model_pro = model_profiler(model, 8)
# print(model_pro)
