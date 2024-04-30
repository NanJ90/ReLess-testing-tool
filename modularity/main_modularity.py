#import each models' h5 
import h5py
import os
import glob
import importlib
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
import sys
import numpy as np
from mnistutil import MNISTUitl
import time 
import json
import pandas as pd

# tf.compat.v1.enable_eager_execution()
#function to load models
def load_models(model_path):
    model = load_model(model_path)
    return model

#MNIST data
labs = [0,1,2,3,4,5,6,7,8,9]
sx = 28
sy = 28
mn = MNISTUitl()
X_train, Y_train, x_test, y_test = mn.getdata2(0,0,sx,sy)

print(X_train.shape)
# #MODULARITY
# #write to file. if file not exist, create it
# if not os.path.exists('../results.txt'):
#     open('../results.txt', 'w').close()
# BATH_PATH = 'remove_irrelavant_edges'
# for path in glob.glob(f'{BATH_PATH}/**/mnist.h5', recursive=True):
#     tic = time.time()  

#     model_type = path.split('/')[-2]
#     model = load_models(path)
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # nm , xt, yt = mn.train2(X, Y, x,y,sx,sy,10,50) #nm, x_test, y_test
    # xt, yt, xT, yT = mn.trainData(X, Y, x,y,sx,sy,10,50)#x_test, y_test, x_train, y_train
    
#     losses = []
#     for i in range(0,len(yt)):
#         pred = nm.predict(xt[i:i+1])
#         losses.append(pred[0][yt[i]])
#     tac = time.time()
#     with open('../results.txt', 'a') as f:
#         f.write(f'{model_type} Layer Model Loss: {np.mean(losses)} Time: {tac-tic} sec \n')

#IMAGENET
DATA_PATH = '../data/imagenet'

# Load JSON files
with open(os.path.join(DATA_PATH,'info/','class_to_label_id.json')) as f:
    class_to_label_id = json.load(f)
    
with open(os.path.join(DATA_PATH,'info/','index_to_label_id.json')) as f:
    index_to_label_id = json.load(f)
    
with open(os.path.join(DATA_PATH,'info/','label_id_to_labels.json')) as f:
    label_id_to_labels = json.load(f)
    
with open(os.path.join(DATA_PATH,'info/','index_to_labels.json')) as f:
    index_to_labels = json.load(f)

# Load CSV file
image_to_label_id = pd.read_csv(os.path.join(DATA_PATH,'info/','image_to_label_id.csv'))

# Convert label ID to actual labels in the CSV using 'label_id_to_labels' JSON
image_to_label_id['label'] = image_to_label_id['label_id'].apply(lambda x: label_id_to_labels[str(x)])

# print(image_to_label_id[image_to_label_id['image_name'] == 'ILSVRC2012_val_00010333'])
# print(image_to_label_id.shape)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [28, 28])
    image = tf.image.rgb_to_grayscale(image)
    image = tf.squeeze(image)
    image = tf.cast(image, tf.float32)
    image = image / 255.0  # normalize to [0,1] range
    return image

X = []
for img in image_to_label_id['image_name']:
    image_path = os.path.join(DATA_PATH,'imgs/',img)
    image = tf.io.read_file(image_path)
    image = preprocess_image(image)
    image = np.squeeze(image)
    X.append(image)

losseses = []
# nm = keras.Sequential([
#             keras.layers.Flatten(input_shape=(img_rows, img_cols,1), name = "Input"),
#             keras.layers.Dense(7, activation=tf.nn.relu ,name = "H"),
#             keras.layers.Dense(numclass, activation=tf.nn.softmax, name = "output")
#         ])
    
#         nm.compile(optimizer='adam',
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'])
    
#         nm.fit(x_train, y_train, epochs=10)

# def load_and_preprocess(image_path):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [28, 28])
#     image = tf.image.rgb_to_grayscale(image)
#     image = tf.squeeze(image)  # Remove color channel
#     image = tf.cast(image, tf.float32)
#     image = image / 255.0  # normalize to [0,1] range
#     return image

# # image_paths = [os.path.join(DATA_PATH, 'imgs/', img) if img else continue for img in image_to_label_id['image_name']
# image_paths = [os.path.join(DATA_PATH, 'imgs/', img) for img in image_to_label_id['image_name'] if img and os.path.isfile(os.path.join(DATA_PATH, 'imgs/', img))]
# paths_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
# num_parallel_calls = 4  # Adjust based on your system
# image_dataset = paths_dataset.map(load_and_preprocess, num_parallel_calls=num_parallel_calls)
# image_dataset = image_dataset.batch(1)  # Batch size of 1 to match the original code
# image_dataset = image_dataset.prefetch(1)  # Prefetch 1 batch at a time

# X = list(image_dataset)

# print(f'There are {len(X)} images in the dataset\n')
# print(f'Each image has the shape: {X[0].shape}\n')






