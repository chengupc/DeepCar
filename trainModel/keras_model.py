
# model
import os
from keras.models import Model, load_model
from keras.layers import Dense, Input, merge
from keras.layers import Convolution2D, MaxPool2D, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten
import numpy as np
import keras
import sys
from PIL import Image


def img2np(imgPath):
    resData = []
    for file in os.lisdir(imgPath):
        img = Image.open(imgPat+"/{}".format(file))
        img_data = np.array(img)
        resData.append(img_data)
    resData = np.array(resData)
    return resData

class deepcar():
    def __init__(self, imgData, batch_size, epochs, model_save_path, model=None):
        self.imgData = imgData
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.model = cnn_model()

    def change_Dataset(self):
        # get datas, labels
        # data = np.load(data_path)
        train_data = imgData[:,0][:,0]
        a = []
        for i in train_data:
            i = i.ravel()
            a.append(i)
        data1 = np.array(a).ravel().reshape(4815,120,160,3)  # dataset
        label1 = np.array([data[:, 0][:, 1], data[:, 0][:, 2]]).T  # feature set
        return data1, label1

    def select_data(self):
        # selection dataset
        data1, label1 = self.change_Dataset()
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_label, test_label = train_test_split(data1, label1, test_size=0.2, random_state=123)
        return train_data, test_data, train_label, test_label

    def cnn_model(self):
        img_in = Input(shape=(120, 160, 3), name='img_in')
        x = img_in
        x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)

        x = Flatten(name='flattened')(x)
        x = Dense(100, activation='linear')(x)
        x = Dropout(.1)(x)
        x = Dense(50, activation='linear')(x)
        x = Dropout(.1)(x)
        # categorical output of the angle
        angle_out = Dense(1, activation='linear', name='angle_out')(x)

        # continous output of throttle
        throttle_out = Dense(1, activation='linear', name='throttle_out')(x)

        model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

        # configure model
        model.compile(optimizer='adam',
                      loss={'angle_out': 'mean_squared_error',
                            'throttle_out': 'mean_squared_error'},
                      loss_weights={'angle_out': 0.5, 'throttle_out': .5})
        return model

    def train_model(self):
        train_data, test_data, train_label, test_label = self.select_data()
        model_1 = self.cnn_model()
        model_1.fit(train_data, [train_label[:,0], train_label[:, 1]],
                        batch_size=batch_size,  # 填充样本的个数
                        epochs=epochs,  # 训练终止时的epoch值
                        validation_data=(test_data, [test_label[:, 0], test_label[:, 1]]))
        model_1.save(os.path.join(model_save_path, "deep_model.h5"))



if __name__ == "__main__":
    imgPath = "../imgs/"
    imgData = img2np(imgPath)

    model_save_path =  r'/home/han/deepcar/src/'
    # data_path = r'/home/han/deepcar/src/RGB.npy'
    batch_size = 32
    epochs = 2
    dp = deepcar(imgData, batch_size, epochs, model_save_path)

    dp.train_model()


    """ 
    mm = load_model("deepcar_model.h5")
    jieguo = mm.predict(test_data)
    jieguo = np.array(jieguo)
    mm.evaluate(test_data, [test_label[:, 0], test_label[:, 1]], batch_size=32)
    """




