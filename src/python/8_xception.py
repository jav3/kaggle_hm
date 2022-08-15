import os
import cv2
from PIL import Image
import zipfile
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.applications.xception import Xception,preprocess_input
import tensorflow as tf
from keras.preprocessing import image
from keras.layers import Input
from keras.backend import reshape
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm


def preprocess_img(new_image):
    dsize = (225,225)
    new_image=np.squeeze(new_image)
    new_image=cv2.resize(new_image,dsize,interpolation=cv2.INTER_NEAREST)  
    new_image=np.expand_dims(new_image,axis=0)
    new_image=preprocess_input(new_image)
    return new_image

def model():
    model=Xception(weights='imagenet',include_top=False, input_shape=(225,225,3))
    for layer in model.layers:
        layer.trainable=False
        #model.summary()
    return model

images_dir = './data/images.zip'

zf = zipfile.ZipFile(images_dir)

features_dict = {}
main_model = model()

for zf_info in tqdm(zf.infolist()):
    fn = zf_info.filename
    zip_img = zf.open(fn)
    pil_img = Image.open(zip_img)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    new_img = preprocess_img(cv_img)
    features = main_model.predict(new_img)
    features=np.array(features)
    features=features.flatten()
    features_dict[fn] = features

zf_infolist = zf.infolist()

batchsize = 1000
features_dict = {}

for i in tqdm(range(0, len(zf_infolist), batchsize)):
    btch_array = []
    fn_array = []
    for zf_info in zf_infolist[i:(i+batchsize)]:
        fn = zf_info.filename
        zip_img = zf.open(fn)
        pil_img = Image.open(zip_img)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        new_img = preprocess_img(cv_img)
        fn_array.append(fn)
        btch_array.append(new_img)
    
    images = np.vstack(btch_array)
    features = main_model.predict(images)
    features = np.array(features)
    flattened = np.array([ff.flatten() for ff in features])
    for x,y in zip(fn_array, flattened):
        features_dict[x] = y 
