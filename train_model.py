import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import keras
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import pickle
from tensorflow.python.keras.models import Sequential , save_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
import time
import pandas as pd
from tqdm import tqdm


NAME = "car-vs-truck-cnn-64x2-{}".format(int(time.time()))

tensorBoard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

DATADIR = "F:\Mon_Hoc\coder\python\car-vs-truck\Datasets"
IMG_SIZE = 64
CATEGORIES = ["car", "truck"]   # Class 0=car, Class 1=truck

def create_training_data():
    pixels = []
    labels = []
    for category in CATEGORIES:  # do cars and trucks

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=car 1=truck

        for img in tqdm(os.listdir(path)):  # iterate over each image per cars and trucks
            try:
                # convert to array
                img_array = cv2.imread(os.path.join(path,img))
                # resize to normalize data size
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                pixels.append(new_array)  
                labels.append(class_num)
            except Exception as e:  # in the interest in keeping the output clean...
                pass
    return pixels, labels
           
pixels, labels = create_training_data()
 

def split_train_val_test_data(pixels, labels):
    # Chuẩn hoá dữ liệu img_array và labels 
    pixels = np.array(pixels) 
    labels = to_categorical(labels) 
    # Nhào trộn dữ liệu ngẫu nhiên
    randomize = np.arange(len(pixels)) 
    np.random.shuffle(randomize) 
    X = pixels[randomize] 
    print("X=", X.shape) 
    y = labels[randomize] 
    # Chia dữ liệu theo tỷ lệ 60% train và 40% còn lại cho val và test
    train_size = int(X.shape[0] * 0.5) 
    X_train, X_val = X[:train_size], X[train_size:] 
    y_train, y_val = y[:train_size], y[train_size:] 
    val_size = int(X_val.shape[0] * 0.5) # 50% của phần 40% bên trên
    X_val, X_test = X_val[:val_size], X_val[val_size:] 
    y_val, y_test = y_val[:val_size], y_val[val_size:] 
    return X_train, y_train, X_val, y_val, X_test, y_test 
X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test_data(pixels, labels)


def set_seed(seed_value=5240):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def build_model(input_shape = (64,64,3), filter_size = (3,3), pool_size = (2,2), output_size = 2):
    model = Sequential()
    model.add(Conv2D(32, filter_size, activation='relu', input_shape=input_shape, padding='same'))
    # Bình thường hóa dữ liệu
    model.add(BatchNormalization())
    model.add(Conv2D(32, filter_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    # Sử dụng max pooling để giảm kích thước ảnh và giữ nguyên các đặc trưng của ảnh
    model.add(MaxPooling2D(pool_size=pool_size))
    # Sử dụng Dropout = 0.2 để tránh hiện tượng Overfitting
    model.add(Dropout(0.2))
    model.add(Conv2D(64, filter_size, activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(64, filter_size, activation='relu', padding='same')) 
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(pool_size=pool_size)) 
    model.add(Dropout(0.2)) 
    model.add(Conv2D(128, filter_size, activation='relu', padding='same')) 
    model.add(BatchNormalization()) 
    model.add(Conv2D(128, filter_size, activation='relu', padding='same')) 
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(pool_size=pool_size)) 
    model.add(Dropout(0.2)) 
    # Làm phẳng dữ liệu trước khi đưa vào train
    model.add(Flatten()) 
    # Khai báo cấu trúc mạng neural với hàm kích hoạt lớp ẩn là relu và lớp ra là softmax 
    model.add(Dense(2048, activation='relu')) 
    model.add(Dropout(0.3)) 
    model.add(Dense(4096, activation='relu')) 
    model.add(Dropout(0.3)) 
    model.add(Dense(1024, activation='relu')) 
    model.add(Dropout(0.3)) 
    model.add(Dense(128, activation='relu')) 
    model.add(Dropout(0.3)) 
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    # In cấu trúc model ra console để kiểm tra
    # model.summary() 
    return model

model = build_model(input_shape=(64,64,3), output_size=2)
epochs = 10
batch_size = 32
model.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val))
model.save("F:\coder\python\DetaiANN\Model")

print(model.evaluate(X_test, y_test))    



