# It's a tkinter canvas that can be used to draw on and then return the drawn image
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from PIL import ImageGrab


class ImageCanvas(tk.Canvas):
    def __init__(self, root, is_draw) -> None:
        """
        The function __init__ is a constructor that initializes the class Canvas
        
        Args:
            root: The root window.
            is_draw: This is a boolean value that determines whether the user is drawing a rectangle or not.
        """
        super().__init__(root, height=300, width=300, background="black")
        self.pack(fill=tk.BOTH, side=tk.LEFT, expand=tk.YES)
        self.root = root
        self.is_draw = is_draw
        self.image_item = None
        self.image_item_gray = None

        self.bind("<B1-Motion>", self.on_move_press)
        self.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None
        self.x = self.y = 0
        self.start_x = None
        self.start_y = None

    def load_image(self, cv_image):
        if self.image_item:
            self.delete(self.image_item)
        self.img = Image.fromarray(cv_image).resize((300, 300))
        self.img_tk = ImageTk.PhotoImage(self.img)
        print("(Image : [{}   {}])".format(
            self.img_tk.height(), self.img_tk.width()))
        self.image_item = self.create_image(
            0, 0, anchor=tk.NW, image=self.img_tk)

    def load_gray_image(self, cv_image):
        if self.image_item_gray:
            self.delete(self.image_item_gray)
        self.img_gray = Image.fromarray(np.uint8(cv_image * 255)).resize(
            (300, 300),
        )
        self.img_tk_gray = ImageTk.PhotoImage(self.img_gray)
        print(
            "(Gray : [{}   {}])".format(
                self.img_tk_gray.height(), self.img_tk_gray.width()
            )
        )
        self.image_item_gray = self.create_image(
            0, 0, anchor=tk.NW, image=self.img_tk_gray
        )

    def on_move_press(self, event):
        if self.is_draw:
            cur_x = self.canvasx(event.x)
            cur_y = self.canvasy(event.y)

            self.create_circle(cur_x, cur_y, 8, fill="white", outline="white")

    def create_circle(self, x, y, r, **kwargs):
        if self.is_draw:
            return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)

    def get_image(self):
        if self.is_draw:
            self.update()
            x = self.root.winfo_rootx() + self.winfo_x()
            y = self.root.winfo_rooty() + self.winfo_y()
            x1 = x+self.winfo_width()
            y1 = y+self.winfo_height()

            image = ImageGrab.grab().crop((x * 2 + 10, y * 2 + 10, x1 * 2 - 10, y1 * 2 - 10))
            return image

    def on_button_release(self, event):
        """_summary_

        Args:
            event (_type_): _description_
        """
        pass
