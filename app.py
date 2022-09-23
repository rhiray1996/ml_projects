import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, accuracy_score)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from control_bar import ControlBar
from image_canvas import ImageCanvas
from output_data import OutputData
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2


class Application(tk.Frame):

    def __init__(self, root) -> None:
        super().__init__(root)
        self.pack(fill=tk.BOTH)
        self.root = root
        self.control_bar = ControlBar(self)
        self.data = OutputData(self)
        self.upload_image_canvas = ImageCanvas(self, False)
        self.input_image_canvas = ImageCanvas(self, False)
        self.draw_image_canvas = ImageCanvas(self, True)
        self.numpy_image_data = None
        self.image_loaded = False

    def clean_canvas(self):
        self.image_loaded = False
        self.upload_image_canvas.delete("all")
        self.input_image_canvas.delete("all")
        self.draw_image_canvas.delete("all")
        self.data.clean_data()

    def load_procees_image(self, image_path):
        if image_path != "":
            self.cv_gray_image = self.convert_gray(cv2.imread(image_path))
            self.upload_image_canvas.load_image(self.cv_gray_image)
            self.numpy_image_data = np.asarray(cv2.resize(self.cv_gray_image, (28, 28)))
            self.input_image_canvas.load_gray_image((self.numpy_image_data < 128).astype(int))
            self.image_loaded = True

    def convert_gray(self, cv_image):
        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    def detect_digit(self, model):
        pil_image = self.draw_image_canvas.get_image()
        numpy_data_draw = np.asarray(cv2.resize(self.convert_gray(np.array(pil_image)), (28, 28)))
        self.input_image_canvas.load_gray_image((numpy_data_draw > 128).astype(int))
        x_test = pd.DataFrame(numpy_data_draw.flatten().reshape((1, 784)))
        x_test = x_test.values.reshape(-1, 28, 28, 1)
        x_test = x_test / 255.0
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        self.data.update_draw_value(y_pred[0])

        if self.image_loaded:
            x_test = pd.DataFrame(self.numpy_image_data.flatten().reshape((1, 784)))
            x_test = x_test.values.reshape(-1, 28, 28, 1)
            x_test = x_test / 255.0
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)
            self.data.update_upload_value(y_pred[0])

    def model_building(self):
        model = self.get_train_model()
        self.test_model(model)

    def test_model(self, model):
        x_test, y_test = self.get_train_test("input//mnist_test.csv")
        y_pred = model.predict(x_test)
        self.score_predictions(y_test, y_pred)

    def get_train_test(self, path):
        data = pd.read_csv(path)
        x = data.drop(columns='label')
        x = x / 255.0
        x = x.values.reshape(-1, 28, 28, 1)
        y = data['label']
        y = to_categorical(y, num_classes=10)
        return x, y

    def score_predictions(self, y_test, y_pred):
        y_test = tf.argmax(y_test, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        print(f"""
            MSE: {mean_squared_error(y_test, y_pred)}
            RMSE: {mean_squared_error(y_test, y_pred)}
            MAE: {mean_absolute_error(y_test, y_pred)}
            R_SQR: {r2_score(y_test, y_pred)}
            RMSLE: {mean_squared_log_error(y_test, y_pred)}
            Accuracy: {accuracy_score(y_test, y_pred)}
            """)

    def get_train_model(self):
        x_train, y_train = self.get_train_test("input//mnist_train.csv")
        X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2)
        datagen = self.get_datagen()
        datagen.fit(X_train)
        model = self.define_model()
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=86),
                                      epochs=10,
                                      validation_data=(X_val, Y_val),
                                      verbose=2,
                                      steps_per_epoch=X_train.shape[0] // 86,
                                      callbacks=[learning_rate_reduction])
        model.save('train_model')
        return model

    def get_datagen(self):
        return ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)

    def define_model(self):
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        return model
