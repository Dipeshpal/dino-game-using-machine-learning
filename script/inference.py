import numpy as np
from PIL import ImageGrab
import cv2
import time
import os
from pynput.keyboard import Key, Controller


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import ktrain

keyboard = Controller()


def process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.resize(processed_img, (512, 512))

    return processed_img


def predict(predictor, path):
    ans = predictor.predict_filename(path)
    print(ans)
    if ans[0] == "up":
        keyboard.press(Key.up)
        keyboard.release(Key.up)
    else:
        keyboard.press(Key.down)
        time.sleep(0.2)
        keyboard.release(Key.down)


def main():
    time.sleep(2)
    predictor = ktrain.load_predictor('model/model-pretrained_mobilenet')
    keyboard.press(Key.space)
    keyboard.release(Key.space)
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        # print('Frame took {} seconds'.format(time.time()-last_time))
        new_screen = process_img(screen)
        cv2.imwrite(f"predict_tmp.jpg", new_screen)

        predict(predictor, "predict_tmp.jpg")
        cv2.imshow('window', new_screen)
        # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


main()
