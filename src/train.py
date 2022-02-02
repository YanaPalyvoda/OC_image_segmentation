import functions as f
import CustomDataGenerator as gn
import numpy as np
import pylab as plt
from glob import glob
import os
import pickle as pkl
from numpy.lib import stride_tricks
import time
import tensorflow as tf
from skimage import feature
from sklearn import metrics
from sklearn.model_selection import train_test_split
from segmentation_models.metrics import IOUScore
from segmentation_models.losses import JaccardLoss
from warnings import warn
from config import *


# Check for a GPU
if not tf.test.gpu_device_name():
    warn.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def read_data(data_dir):
    print ('[INFO] Reading image data.')
    #train data
    train_batch = glob(os.path.join(data_dir+'/train/images/', '*.png'))    
    train_label_batch = glob(os.path.join(data_dir +'/train/masks/', '*.png'))
    train_batch.sort()
    train_label_batch.sort()
    print(len(train_batch),train_batch[0])
    print(len(train_label_batch),train_label_batch[0])
    #validation data
    val_batch = glob(os.path.join(data_dir+'/val/images/', '*.png'))
    val_label_batch = glob(os.path.join(data_dir +'/val/masks/', '*.png'))
    val_batch.sort()
    val_label_batch.sort()
    train_set = f.get_tuple_image_mask(train_batch,train_label_batch)
    valid_set = f.get_tuple_image_mask(val_batch,val_label_batch)
    return train_set, valid_set

def get_labels_data():
    labels = f.get_label_info()
    label_values, colors_replacements = f.get_labels_mapping(labels)
    return label_values, colors_replacements


def create_generators(train_set, valid_set,label_values, colors_replacements,image_size,batch_size):
    train_generator = gn.CustomDataGenerator(images_and_labels=train_set,batch_size = batch_size, label_values=label_values,target_size=(image_size,2*image_size), colors_replacements=colors_replacements,data_augmentation=True)
    val_generator = gn.CustomDataGenerator(images_and_labels=valid_set,batch_size = batch_size, label_values=label_values, target_size=(image_size,2*image_size),colors_replacements=colors_replacements)
    return train_generator, val_generator

def get_model_Unet_compile(image_size):
    model_Unet = f.Unet(image_size,2*image_size, 64)
    loss = JaccardLoss()
    score = IOUScore()
    model_Unet.compile(optimizer="adam", loss=loss ,metrics=[score])
    return model_Unet

def train_model(train_generator_unet, val_generator_unet, model,epochs):
        train_steps_unet = train_generator_unet.__len__()
        val_steps_unet = val_generator_unet.__len__()
        print ('[INFO] Start training U-Net model....')
        results_Unet = model.fit(train_generator_unet,steps_per_epoch=train_steps_unet ,epochs=epochs,validation_data=val_generator_unet,validation_steps=val_steps_unet)
        print ('[INFO] Fin training U-Net model.')
        return model

def test_model(val_generator, model):
    scores = model.evaluate(val_generator, verbose=0,return_dict=True)
    print ('--------------------------------')
    print ('[RESULTS] Loss: %.2f' %scores.get('loss'))
    print ('[RESULTS] Intersection-Over-Union: %.2f' %(scores.get('iou_score')*100))
    print ('--------------------------------')

def run():
    # Set variables from config.py file
    image_size = IMAGE_SIZE
    epochs = EPOCHS
    bath_size = BATCH_SIZE
    data_dir = DATA_DIR
    datastore = DATASTORE
    model_path = MODEL_PATH
    start = time.time()
    train_set, valid_set = read_data(data_dir)
    label_values, colors_replacements = get_labels_data()
    train_generator, val_generator = create_generators(train_set, valid_set,
                                                       label_values,
                                                       colors_replacements,
                                                       image_size,bath_size)
    model_comp = get_model_Unet_compile(image_size)
    model_train = train_model(train_generator,val_generator,model_comp,epochs)
    test_model(val_generator, model_train)
    os.makedirs('../outputs', exist_ok=True)
    model_train.save(model_path)
    print ('Processing time:',time.time()-start)
    return model_train

if __name__ == '__main__':
    run()
