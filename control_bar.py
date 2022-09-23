import tkinter as tk
from tkinter import filedialog
from keras.models import load_model


class ControlBar(tk.Frame):

    def __init__(self, root) -> None:
        super().__init__(root, height=30)
        self.pack(fill=tk.X, side=tk.BOTTOM)
        self.root = root
        self.upload_button = tk.Button(self, text="Upload", command=self.upload_image)
        self.upload_button.pack(fill=tk.BOTH, side=tk.LEFT, expand=tk.YES)
        self.detect = tk.Button(self, text="Detect", command=self.on_click_detect)
        self.detect.pack(fill=tk.BOTH, side=tk.LEFT, expand=tk.YES)
        self.clean = tk.Button(self, text="Clean", command=self.on_click_clean)
        self.clean.pack(fill=tk.BOTH, side=tk.LEFT, expand=tk.YES)
        self.train = tk.Button(self, text="Train", command=self.on_click_train)
        self.train.pack(fill=tk.BOTH, side=tk.LEFT, expand=tk.YES)
        self.test = tk.Button(self, text="Test", command=self.on_click_test)
        self.test.pack(fill=tk.BOTH, side=tk.RIGHT, expand=tk.YES)
        self.model = load_model('train_model')

    def on_click_clean(self):
        """detect digit in image
        """
        self.root.clean_canvas()

    def on_click_detect(self):
        """detect digit in image
        """
        self.root.detect_digit(self.model)

    def on_click_test(self):
        self.root.test_model(self.model)

    def on_click_train(self):
        self.root.model_building()

    def upload_image(self):
        options = {}
        image_files = "image files"
        options["filetypes"] = [
            (image_files, ".png"),
            (image_files, ".jpg"),
            (image_files, ".jpeg"),
        ]
        options["parent"] = self
        options["title"] = "Select digit image file."
        response = filedialog.askopenfilename(**options)
        self.root.load_procees_image(response)
