import flask
import json
import os
from PIL import Image
import base64
import json
import io
from colormap import rgb2hex
from config import *
from functionsLabels import *
import tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initializing flask app
app = flask.Flask(__name__)
app.secret_key = SECRET_KEY
model_path = MODEL_PATH
global model
model = tensorflow.keras.models.load_model(model_path, compile=False)



from flask import Flask, request,render_template,flash,make_response


# Controller-1
@app.route("/", methods=['GET'])
def get_home():
    # afficher le formulaire
    return render_template('index.html',imagename='',image_origine='',image_segment = '',labels = '',result = False)


@app.route("/", methods=['POST'])
def get_result():
    result = request.form
    imageId = result['imageId'] # on récupère  imageId'
    if len(imageId) > 0:
        #flash(u'Your request is in process...')
        imagename,image_origine,image_load,image_masque  = get_image_and_mask(imageId)
        if image_origine is not None:
            image_segment = predict(model,image_origine)
            labels = get_labels()
            return render_template("index.html",imagename=imagename,image_origine=get_encode_image(image_load),
                                   image_segment = get_encode_image(image_segment),image_masque=get_encode_image(image_masque),labels =                                      labels,result = True)
        else:
            flash(u'Image id number not find. Please enter another image id')
            return render_template('index.html',imagename='',image_origine='',image_segment =                      '',image_masque='',labels = '', result = False)
            
    else:
        flash(u'Error in the sent data.')
        return render_template('index.html',imagename='',image_origine='',image_segment = '',image_masque='',labels = '', result = False)

def get_image_and_mask(id):
    data_dir =DATADIR
    dataset = ['val']
    for data in dataset: 
        src_folder = os.path.join(data_dir,data,"images")
        imagename,image,image_load = find_image(id,src_folder)
        if image is not None:
            mask_folder = os.path.join(data_dir, data,"masks")
            print(mask_folder)
            _,mask, mask_load = find_image(id,mask_folder)
            if mask is not None:
                return imagename,image,image_load,mask_load
    return None, None, None, None

def find_image(id,src_folder):
    image_size = IMAGE_SIZE
    imagename = [f for f in os.listdir(src_folder) if f.find(id)!=-1][0]
    if len(imagename)>0:
        path = os.path.join(src_folder,imagename)
        image_load = Image.open(path)
        img = img_to_array(load_img(path,target_size=(image_size,2*image_size,3)))/255
        return imagename,img,image_load
    else:
        return None, None, None
    
def make_prediction(model,img):
    img = np.expand_dims(img,axis=0)
    labels = model.predict(img)
    labels = np.argmax(labels[0],axis=2)
    return labels

def form_colormap(prediction,mapping):
    h,w = prediction.shape
    color_label = np.zeros((2*h,4*w,3),dtype=np.uint8)    
    color_label = mapping[prediction]
    color_label = color_label.astype(np.uint8)
    return color_label

def get_label_values():
    labels = get_label_info()
    label_values, colors_replacements = get_labels_mapping(labels)
    return label_values
        
def predict(model,img):
    pred_label = make_prediction(model, img)
    pred_colored = form_colormap(pred_label,np.array(get_label_values()))
    load_predict = Image.fromarray(pred_colored, "RGB")
    return load_predict
    
def get_encode_image(img):
    #Convertir l'image au format envoyable et la stocker en JSON
    data = io.BytesIO()
    img.save(data, "png")
    encoded_img_data = base64.b64encode(data.getvalue())
    img_byte = encoded_img_data.decode("utf-8")
    return img_byte

    
def get_labels():
    label_map = {'void': (0, 0, 0),
  'flat': (128, 64, 128),
  'construction': (70, 70, 70),
  'object': (153, 153, 153),
  'nature': (107, 142, 35),
  'sky': (70, 130, 180),
  'human': (220, 20, 60),
  'vehicle': (0, 0, 142)}
    result = []
    for label, col in label_map.items():
        print(label, col)
        result.append((label.upper(),rgb2hex(col[0],col[1],col[2])))
    return result
    
# Running the api
if __name__ == '__main__':
    app.run(debug=True)

        
