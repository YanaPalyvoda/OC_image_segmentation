import numpy as np
from tensorflow.keras.utils import to_categorical ,Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import albumentations as A


class CustomDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, images_and_labels, label_values, colors_replacements, n_classes = 8,batch_size=10, target_size=(128,256), shuffle=True,data_augmentation=False ):
        'Initialization'
        self.target_size = target_size
        self.images_and_labels = images_and_labels
        self.label_values = label_values
        self.colors_replacements = colors_replacements
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.on_epoch_end()
      
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images_and_labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [k for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_labels = list()
        # Generate data
        for i in list_IDs_temp:
            # Store sample
            img = load_img(self.images_and_labels[i][0] ,color_mode="rgb",target_size=self.target_size)
            img = img_to_array(img)/255.
            label = load_img(self.images_and_labels[i][1],color_mode="rgb",target_size=self.target_size)
            label = img_to_array(label)
            if self.data_augmentation:
              img, label = self.__data_aug(img, label)
            label = self.__change_color(label)
            label = self.__transform_label(label)
            label = to_categorical(label,self.n_classes)
            batch_imgs.append(img)
            batch_labels.append(label)
        return np.array(batch_imgs),np.array(batch_labels)
   

    def __data_aug(self,input_image, input_mask):
        # random data augmentation
        input_mask = input_mask/255
        transform = A.Compose([A.HorizontalFlip(),A.RandomBrightnessContrast()])
        transformed = transform(image=input_image, mask=input_mask)
        input_image = transformed['image']
        input_mask = transformed['mask']
        input_mask = input_mask*255
        return input_image, input_mask

    def __change_color(self,img):
        for color in self.colors_replacements:
            color_before = np.array(color[0])
            color_after = np.array(color[1])
            img[(img == color_before).all(axis = -1)] = color_after
        return img


    def __transform_label(self,mask):
        # encode the label
        mask = mask.astype("uint8")
        label = np.zeros(mask.shape[:2],dtype= np.uint8)
        for i, rgb in enumerate(self.label_values):
            label[(mask == rgb).all(axis=2)] = i
        return label

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_and_labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
