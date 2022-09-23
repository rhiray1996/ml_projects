import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from matplotlib.pyplot import text


class OutputData(tk.Frame):
    def __init__(self, root) -> None:
        super().__init__(root, height=30)
        self.pack(fill=tk.X, side=tk.BOTTOM)
        self.root = root
        self.upload_string = "Uploaded Detection : {}"
        self.draw_string = "Draw Detection : {}"
        self.upload = tk.Label(
            self, text=self.upload_string.format("None"))
        self.upload.pack(fill=tk.BOTH, side=tk.LEFT, expand=tk.YES)
        self.draw = tk.Label(self, text=self.draw_string.format("None"))
        self.draw.pack(fill=tk.BOTH, side=tk.LEFT, expand=tk.YES)

    def update_upload_value(self, v):
        self.upload.config(text=self.upload_string.format(v))

    def update_draw_value(self, v):
        self.draw.config(text=self.draw_string.format(v))

    def clean_data(self):
        self.upload.config(text=self.upload_string.format("None"))
        self.draw.config(text=self.draw_string.format("None"))
