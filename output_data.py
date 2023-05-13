# It creates a status bar at the bottom of the GUI window.
# It creates a status bar at the bottom of the GUI window.
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from matplotlib.pyplot import text


# This class is a tkinter frame that contains a text widget and a scrollbar
class OutputData(tk.Frame):
    def __init__(self, root) -> None:
        """
        This function creates a status bar at the bottom of the GUI window
        """
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
        """
        It updates the upload value in the GUI
        """
        self.upload.config(text=self.upload_string.format(v))

    def update_draw_value(self, v):
        """
        The function update_draw_value takes in a value v and updates the text of the draw label to the
        value of v
        """
        self.draw.config(text=self.draw_string.format(v))

    def clean_data(self):
        """
        This function resets the text of the upload and draw buttons to their default values
        """
        self.upload.config(text=self.upload_string.format("None"))
        self.draw.config(text=self.draw_string.format("None"))
