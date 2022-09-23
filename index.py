import tkinter as tk
from app import Application

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x400")
    app = Application(root)
    root.mainloop()