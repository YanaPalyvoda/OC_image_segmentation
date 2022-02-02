import numpy as np
import os.path
import os
import tensorflow as tf
from glob import glob
import functionsLabels as f
import CustomDataGenerator as gn
from config import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def get_data_to_predict(data_dir):
    test_set = [f for f in os.listdir(data_dir+'/test/images/') if not f.startswith(".")]
    return test_set


def model_load(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    
    return model

def make_prediction(model,img):
    img = np.expand_dims(img,axis=0)
    labels = model.predict(img)
    labels = np.argmax(labels[0],axis=2)
    return labels

def form_colormap(prediction,mapping):
    h,w = prediction.shape
    color_label = np.zeros((h,w,3),dtype=np.uint8)    
    color_label = mapping[prediction]
    color_label = color_label.astype(np.uint8)
    return color_label

def get_label_values():
    labels = f.get_label_info()
    label_values, colors_replacements = f.get_labels_mapping(labels)
    return label_values
    

def run():
    image_size = IMAGE_SIZE
    bath_size = BATCH_SIZE
    data_dir = DATA_DIR
    model_path = MODEL_PATH
    result_path = RESULT_PATH
    test_set = get_data_to_predict(data_dir)
    model = model_load(model_path)
    label_values = get_label_values()
    for image in test_set:
        path = os.path.join(data_dir+'/test/images/', image)
        img = img_to_array(load_img(path,target_size=(image_size,2*image_size,3)))/255
        pred_label = make_prediction(model, img)
        pred_colored = form_colormap(pred_label,np.array(label_values))
        output = os.path.join(result_path, 'prd_'+image)
        os.makedirs('../result', exist_ok=True)
        tf.keras.preprocessing.image.save_img(output,pred_colored)


if __name__ == '__main__':
    run()

 
