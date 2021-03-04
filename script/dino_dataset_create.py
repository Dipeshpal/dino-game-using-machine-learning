import numpy as np
from PIL import ImageGrab
import cv2
import time
from pynput.keyboard import Listener
import os


def on_press(key):  # The function that's called when a key is pressed
    def process_img(image, last_time, key):
        key = str(key)
        # convert to gray
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # edge detection
        processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
        if key == "Key.up":
            print("dataset/{key}/{str(last_time)}_{key}.jpg")
            cv2.imwrite(f"dataset/up/{str(last_time)}_up.jpg", processed_img)
        if key == "Key.down":
            cv2.imwrite(f"dataset/down/{str(last_time)}_down .jpg", processed_img)

    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    last_time = time.time()
    process_img(screen, last_time, key)

    return key


for dir_ in ['dataset', 'dataset/up', 'dataset/down']:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

with Listener(on_press=on_press) as listener:
    listener.join()
