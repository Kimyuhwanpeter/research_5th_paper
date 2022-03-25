# -*- coding:utf-8 -*-
from CED_model import *

import tensorflow as tf
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size_1": 448,
                           
                           "img_size_2": 896,

                           "tr_txt": "/content/train.txt",
                           
                           "tr_lab_path": "/content/raw_aug_gray_mask",
                           
                           "tr_img_path": "/content/low_light2",
                           
                           "pre_checkpoint": False,

                           "train": True,
                           
                           "pre_checkpoint_path": "",
                           
                           "batch_size": 4,
                           
                           "epochs": 40,
                           
                           "lr": 0.0001,
                           
                           "save_checkpoint_M1": "/content/drive/MyDrive/5th_paper/related_work/CED_Net/checkpoint_BoniRob/m1/m1.ckpt",
                           
                           "save_checkpoint_M2": "/content/drive/MyDrive/5th_paper/related_work/CED_Net/checkpoint_BoniRob/m2/m2.ckpt",
                           
                           "save_checkpoint_M3": "/content/drive/MyDrive/5th_paper/related_work/CED_Net/checkpoint_BoniRob/m3/m3.ckpt",
                           
                           "save_checkpoint_M4": "/content/drive/MyDrive/5th_paper/related_work/CED_Net/checkpoint_BoniRob/m4/m4.ckpt",
                           
                           "te_txt": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/test.txt",
                           
                           "te_lab_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_aug_gray_mask",
                           
                           "te_img_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/low_light2",})

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def Upsample_Conc(result,C):  
    D = tf.keras.layers.UpSampling2D(size = (2,2))(result)
    x = tf.keras.layers.concatenate([D, C],axis=-1)
    return x

def tr_func_C1(img_path, lab_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, 3)
    img_C1 = tf.image.resize(img, [FLAGS.img_size_1, FLAGS.img_size_1])

    lab = tf.io.read_file(lab_path)
    lab = tf.image.decode_png(lab, 3)
    lab_C1 = tf.image.resize(lab, [FLAGS.img_size_1, FLAGS.img_size_1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab_C1 = tf.image.convert_image_dtype(lab_C1, tf.uint8)

    return img_C1, lab_C1

def tr_func_C2(img_path, lab_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, 3)
    img_C2 = tf.image.resize(img, [FLAGS.img_size_2, FLAGS.img_size_2])
    img_C1 = tf.image.resize(img, [FLAGS.img_size_1, FLAGS.img_size_1])

    lab = tf.io.read_file(lab_path)
    lab = tf.image.decode_png(lab, 3)
    lab_C2 = tf.image.resize(lab, [FLAGS.img_size_2, FLAGS.img_size_2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab_C2 = tf.image.convert_image_dtype(lab_C2, tf.uint8)

    return img_C2, img_C1, lab_C2

def true_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    #y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

@tf.function
def cal_loss(m, batch_images, batch_labels):

    with tf.GradientTape() as tape:

        batch_labels = tf.reshape(batch_labels, [-1,])

        logits = m(batch_images, True)
        logits = tf.reshape(logits, [-1,])

        loss = true_dice_loss(batch_labels, logits)
       
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def main():

    m1 = MYMODEL1() # weed
    m1.summary()
    m2 = MYMODEL2() # weed
    m2.summary()
    m3 = MYMODEL3() # crop
    m3.summary()
    m4 = MYMODEL4() # crop
    m4.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint()
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    if FLAGS.train:
        
        img_path = np.loadtxt(FLAGS.tr_txt, dtype="<U200", skiprows=0, usecols=0)
        lab_path = np.loadtxt(FLAGS.tr_txt, dtype="<U200", skiprows=0, usecols=0)
        img_path = [FLAGS.tr_img_path + "/" + data for data in img_path]
        lab_path = [FLAGS.tr_lab_path + "/" + data for data in lab_path]

        count = 0
        for epoch in range(FLAGS.epochs):   # C1

            tr_gener = tf.data.Dataset.from_tensor_slices((img_path, lab_path))
            tr_gener = tr_gener.shuffle(len(img_path))
            tr_gener = tr_gener.map(tr_func_C1)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(img_path) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)
                
                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == 255, 0, batch_labels)
                batch_labels = np.where(batch_labels == 128, 1, batch_labels)

                loss = cal_loss(m1, batch_images, batch_labels)

                if count % 10 == 0:
                    print("M1 Loss = {}, [{}/{}]".format(loss, step + 1, tr_idx))

                count += 1

        ckpt = tf.train.Checkpoint(m1=m1, optim=optim)
        ckpt.save(FLAGS.save_checkpoint_M1)

        count = 0
        for epoch in range(FLAGS.epochs):
            tr_gener = tf.data.Dataset.from_tensor_slices((img_path, lab_path))
            tr_gener = tr_gener.shuffle(len(img_path))
            tr_gener = tr_gener.map(tr_func_C2)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(img_path) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, img_C1, batch_labels = next(tr_iter)
                result1 = m1.predict(img_C1)
                batch_images = Upsample_Conc(result1, batch_images)

                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == 255, 0, batch_labels)
                batch_labels = np.where(batch_labels == 128, 1, batch_labels)

                loss = cal_loss(m2, batch_images, batch_labels)

                if count % 10 == 0:
                    print("M2 Loss = {}, [{}/{}]".format(loss, step + 1, tr_idx))

                count += 1

        ckpt = tf.train.Checkpoint(m2=m2, optim=optim)
        ckpt.save(FLAGS.save_checkpoint_M2)

        count = 0
        for epoch in range(FLAGS.epochs):   # C1

            tr_gener = tf.data.Dataset.from_tensor_slices((img_path, lab_path))
            tr_gener = tr_gener.shuffle(len(img_path))
            tr_gener = tr_gener.map(tr_func_C1)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(img_path) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)
                
                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == 255, 1, batch_labels)
                batch_labels = np.where(batch_labels == 128, 0, batch_labels)

                loss = cal_loss(m3, batch_images, batch_labels)

                if count % 10 == 0:
                    print("M3 Loss = {}, [{}/{}]".format(loss, step + 1, tr_idx))

                count += 1

        ckpt = tf.train.Checkpoint(m3=m3, optim=optim)
        ckpt.save(FLAGS.save_checkpoint_M3)

        count = 0
        for epoch in range(FLAGS.epochs):
            tr_gener = tf.data.Dataset.from_tensor_slices((img_path, lab_path))
            tr_gener = tr_gener.shuffle(len(img_path))
            tr_gener = tr_gener.map(tr_func_C2)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(img_path) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, img_C1, batch_labels = next(tr_iter)
                result1 = m3.predict(img_C1)
                batch_images = Upsample_Conc(result1, batch_images)

                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == 255, 1, batch_labels)
                batch_labels = np.where(batch_labels == 128, 0, batch_labels)

                loss = cal_loss(m4, batch_images, batch_labels)

                if count % 10 == 0:
                    print("M4 Loss = {}, [{}/{}]".format(loss, step + 1, tr_idx))

                count += 1

        ckpt = tf.train.Checkpoint(m2=m2, optim=optim)
        ckpt.save(FLAGS.save_checkpoint_M4)

    else:
        ckpt1 = tf.train.Checkpoint(m1=m1, optim=optim)
        ckpt_manager1 = tf.train.CheckpointManager(ckpt1, FLAGS.save_checkpoint_M1, 5)
        ckpt1.restore(ckpt_manager1.latest_checkpoint)
        print("Restored the M1.....")

        ckpt2 = tf.train.Checkpoint(m2=m2, optim=optim)
        ckpt_manager2 = tf.train.CheckpointManager(ckpt2, FLAGS.save_checkpoint_M2, 5)
        ckpt2.restore(ckpt_manager2.latest_checkpoint)
        print("Restored the M2.....")

        ckpt3 = tf.train.Checkpoint(m3=m3, optim=optim)
        ckpt_manager3 = tf.train.CheckpointManager(ckpt3, FLAGS.save_checkpoint_M3, 5)
        ckpt3.restore(ckpt_manager3.latest_checkpoint)
        print("Restored the M3.....")

        ckpt4 = tf.train.Checkpoint(m4=m4, optim=optim)
        ckpt_manager4 = tf.train.CheckpointManager(ckpt3, FLAGS.save_checkpoint_M4, 5)
        ckpt4.restore(ckpt_manager4.latest_checkpoint)
        print("Restored the M4.....")

        img_path = np.loadtxt(FLAGS.te_txt, dtype="<U200", skiprows=0, usecols=0)
        lab_path = np.loadtxt(FLAGS.te_txt, dtype="<U200", skiprows=0, usecols=0)
        img_path = [FLAGS.te_img_path + "/" + data for data in img_path]
        lab_path = [FLAGS.te_lab_path + "/" + data for data in lab_path]

        te_gener = tf.data.Dataset.from_tensor_slices((img_path, lab_path))
        te_gener = te_gener.shuffle(len(img_path))
        te_gener = te_gener.map(tr_func_C2)
        te_gener = te_gener.batch(1)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        te_iter = iter(te_gener)
        for i in range(len(img_path)):
            batch_images, img_C1, batch_labels = next(te_iter)
            result1 = m1.predict(img_C1)
            C_predict = Upsample_Conc(result1, batch_images)
            result_W = m2.predict(C_predict).numpy()

            result2 = m3.predict(img_C1)
            C_predict2 = Upsample_Conc(result2, batch_images)
            result_C = m4.predict(C_predict2).numpy()

            re = np.zeros([FLAGS.img_size_2, FLAGS.img_size_2, 2])
            C = result_C[i, :, :, 0]
            W = result_W[i, :, :, 0]

            c = np.zeros([896,896])
            w = np.zeros([896,896])
            w[np.where(W>=0.5)] = 1
            c[np.where(C>=0.5)] = 1
            c[np.where(w==1)] = 0   # 여기는 테스트를 해봐야할것같음

if __name__ == "__main__":
    main()
