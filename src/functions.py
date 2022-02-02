#import librairies
import random
import numpy as np
import os.path
import os
from glob import glob
from sklearn.utils import shuffle
import random
import time
import shutil
import tensorflow as tf
import scipy.misc
from matplotlib import pyplot as plt
from tensorflow import keras
#from tensorflow.python.keras.engine import keras_tensor
#import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical,Sequence
#from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, Sequential
from warnings import warn
from distutils.version import LooseVersion
#%env SM_FRAMEWORK=tf.keras
from segmentation_models.metrics import IOUScore,FScore
import albumentations as A
from functionsLabels import *




# function definition
def get_tuple_image_mask(images,masks):
    #return pair of path data image, mask
    return tuple(zip(images, masks))

# FCN implementation
def layer_conv_block_FCN(x, nfilters, nb_convblocks = 2, activation = "relu",block_name = 'block',size=3, padding='same', initializer="he_normal"):
    for i in range(nb_convblocks):
        x = Conv2D(filters=nfilters, kernel_size=(size, size), 
                   activation = activation,name=block_name + '_'+str(i),
                   padding=padding, kernel_initializer=initializer)(x)
    
    #x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name=block_name + '_pool')(x)
    return x

def FCN_model(n_classes=8, input_height=128,input_width=256):

    inputs = Input((input_height,input_width, 3))
    f1 = layer_conv_block_FCN(inputs,nfilters = 64,block_name = 'block1')

    f2 = layer_conv_block_FCN(f1, nfilters = 128, block_name = 'block2')

    f3 = layer_conv_block_FCN(f2, nfilters = 256, nb_convblocks = 3,
                              block_name = 'block3')
                             
    pred3= Conv2DTranspose( n_classes , kernel_size=(16,16),strides=(16,16),
                           use_bias=False)(f3)

    f4 = layer_conv_block_FCN(f3, nfilters = 512, nb_convblocks = 3,
                              block_name = 'block4')

    pred4= Conv2DTranspose( n_classes , kernel_size=(32,32), strides=(32,32),
                           use_bias=False )(f4)

    f5 = layer_conv_block_FCN(f4, nfilters = 512, nb_convblocks = 3,
                              block_name = 'block5')

    pred5= Conv2DTranspose( n_classes, kernel_size=(64,64),strides=(64,64),
                           use_bias=False )(f5)


    o = Add(name="add")([pred3, pred4, pred5])

    o = MaxPooling2D((2, 2), strides=(2, 2), name='final')(o)
    o = (Conv2D(n_classes, ( 1 , 1 ),kernel_initializer='he_normal'))(o)
    #o = Conv2DTranspose( n_classes , kernel_size=(32,32) ,  strides=(32,32) , use_bias=False ,  data_format=IMAGE_ORDERING )(o)
    o = (Activation('softmax'))(o)

    model = Model([inputs] , [o] )
    return model

# Unet implementation
def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y

def Unet(h, w, filters):
# down
    input_layer = Input(shape=(h, w, 3), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.5)(conv5)
# up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
    output_layer = Conv2D(filters=8, kernel_size=(1, 1), activation='softmax')(deconv9)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model



def display_loss_and_metrics(epochs, history, metrics):
  N = np.arange(0, epochs)
  for mtr in metrics:
    plt.figure()
    plt.plot(N, history[mtr], label=mtr)
    plt.plot(N, history["val_"+mtr], label="val_"+mtr)
    plt.title("Training and validate  "+mtr)
    plt.xlabel("Epoch #")
    plt.ylabel(mtr)
    plt.legend(loc="lower left")
    plt.show()

def form_colormap(prediction,mapping):
    h,w = prediction.shape
    color_label = np.zeros((h,w,3),dtype=np.uint8)    
    color_label = mapping[prediction]
    color_label = color_label.astype(np.uint8)
    return color_label


def make_prediction(model,img):
    img = np.expand_dims(img,axis=0)
    labels = model.predict(img)
    labels = np.argmax(labels[0],axis=2)
    return labels


def display_random_test_images(model,data,label_values,colors_replacements,nb_images=1,maxnb = 100,image_size = 128):
   for i in range(nb_images):
      idx = random.randint(0, maxnb)
      temp = data[idx]
      img = img_to_array(load_img(temp[0],target_size=(image_size,2*image_size,3)))/255
      mask = img_to_array(load_img(temp[1]))
      mask = change_color(mask,colors_replacements)
      pred_label = make_prediction(model, img)
      pred_colored = form_colormap(pred_label,np.array(label_values))
      plt.figure(figsize=(15,15))
      plt.subplot(131);plt.title('Original Image')
      plt.imshow(img)
      plt.subplot(132);plt.title('True labels')
      plt.imshow(mask/255.)
      plt.subplot(133)
      plt.imshow(pred_colored/255.);plt.title('Predicted labels')
      plt.show()

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def get_scoring(model,model_name,valid_generator,buid_time,
                data_augmentation='No'):#Brightness Contrast'):#'Rotation'):#Random crop'):#'Flip left right'):
    # creation of a line for the score table
    print('MODEL: ',model_name)
    print('Build training model time: ',buid_time)
    start2 = time.time()
    y_pred=model.predict(valid_generator)
    end2 = time.time()
    test_time =timer(start2,end2)
    print('Predict test data time: ',test_time)
    scores = model.evaluate(valid_generator, verbose=1,return_dict=True);
    ligne_resultat = {'Model':model_name,
                          'Data augmentation':data_augmentation,
                          'Categorical accuracy':scores.get('categorical_accuracy')*100,
                          'Loss':scores.get('loss'),
                          'Dice Coefficient':scores.get('f1-score')*100,
                          'Intersection-Over-Union':scores.get('iou_score')*100,
                          'Build training time':buid_time,
                          'Used memory':0,
                          'Predict data time':test_time
                         }
    print('Categorical accuracy: {:.4f}'.format(scores.get('categorical_accuracy')*100))
    print('Loss : {:.4f}'.format(scores.get('loss')))
    print('Dice Coefficient : {:.4f}'.format(scores.get('f1-score')*100))
    print('Intersection-Over-Union : {:.4f}'.format(scores.get('iou_score')*100))
    return ligne_resultat
   
def get_test_data(test_data,target_size,n_classes,lb_values,col_replacements):
  imgs = list()
  labels = list()
  for data in test_data:
      img = load_img(data[0],color_mode="rgb",target_size=(target_size, 2*target_size))
      img = img_to_array(img)/255.
      label = load_img(data[1],color_mode="rgb",target_size=(target_size, 2*target_size))
      label = img_to_array(label)
      label = change_color(label,col_replacements)
      label = transform_label(label,lb_values)
      label = to_categorical(label,n_classes)
      imgs.append(img)
      labels.append(label)
  return np.array(imgs),np.array(labels)



