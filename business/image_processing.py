import cv2
import numpy as np


class ImageProcessing():

    def __init__(self) -> None:
        pass

    def load_procees_image(self, image_path):
        if image_path != "":
            self.cv_gray_image = self.convert_to_gray(cv2.imread(image_path))
            self.upload_image_canvas.load_image(self.cv_gray_image)
            self.numpy_image_data = np.asarray(cv2.resize(self.cv_gray_image, (28, 28)))
            self.input_image_canvas.load_gray_image((self.numpy_image_data < 128).astype(int))
            self.image_loaded = True

    def convert_to_gray(self, cv_image):
        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)