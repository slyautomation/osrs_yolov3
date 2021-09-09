import colorsys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import random
import pyautogui
import numpy as np
import pytesseract
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess

import multiprocessing
from multiprocessing import Pipe
from multiprocessing import Process
import mss
import time
text = 'test'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# set start time to current time
start_time = time.time()

shoot_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
sct = mss.mss()
# Set monitor size to capture
monitor = {"top": 40, "left": 0, "width": 800, "height": 800}

width = 1920  # 800
height = 1080  # 640


def Shoot(mid_x, mid_y):
    x = int(mid_x * width)
    # y = int(mid_y*height)
    y = int(mid_y * height + height / 9)
    pyautogui.moveTo(x, y)
    pyautogui.click()


class YOLO(object):
    _defaults = {
        # "model_path": 'logs/ep050-loss21.173-val_loss19.575.h5',
        "model_path": 'weighted_files/osrs_cow2_trained_weights_final.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'weighted_files/cow2_CLASS_test_classes.txt',
        "score": 0.80,
        "iou": 0.45,
        "model_image_size": (416, 416), #416
        "text_size": 3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        global coords, classes, percent
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],  # [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale = 1
        ObjectsList = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            # label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            percent = scores
            classes = predicted_class
            coords = box

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom - top) / 2 + top
            mid_v = (right - left) / 2 + left

            # put object rectangle
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

            # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                                  thickness / self.text_size, 1)

            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c],
                          thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX, thickness / self.text_size, (0, 0, 0),
                        1)

            # add everything to list
            ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return image, ObjectsList

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        r_image, ObjectsList = self.detect_image(original_image_color)
        return r_image, ObjectsList


def TestImage(preprocess, image):
    global text
    # construct the argument parse and parse the arguments
    image = cv2.imread(image)
    image = cv2.bitwise_not(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # check to see if we should apply thresholding to preprocess the
    # image
    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove
    # noise
    if preprocess == "blur":
        gray = cv2.medianBlur(gray, 3)

    if preprocess == 'adaptive':
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    f = open("action.txt", "w")
    f.write(text)
    f.close()
    #print(text)
    # show the output images
    #cv2.imshow("Image", image)
    #cv2.imshow("Output", gray)
    #cv2.waitKey(0)

def resizeImage():
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save('screen.png')
    png = 'screen.png'
    im = Image.open(png)  # uses PIL library to open image in memory
    left = 10
    top = 51
    right = 140
    bottom = 69
    im = im.crop((left, top, right, bottom))  # defines crop points
    im.save('textshot.png')  # saves new cropped image
    width, height = im.size
    new_size = (width * 4, height * 4)
    im1 = im.resize(new_size)
    im1.save('textshot.png')

def click_attack(x, y):
    b = random.uniform(0.05, 0.09)
    x = random.uniform(x-2.5, x+2.5)
    y = random.uniform(y-2.5, y+2.5)
    c = random.uniform(0.03, 0.06)
    pyautogui.moveTo(x, y, duration=b)
    time.sleep(c)
    pyautogui.click(x, y, 1, button='left')

def GRABMSS_screen(p_input):
    while True:
        # Grab screen image
        img = np.array(sct.grab(monitor))

        # Put image from pipe
        p_input.send(img)

def READ_Text():
    while True:
        global fished
        # Grab screen image
        resizeImage()
        TestImage('thresh', 'textshot.png')
        # Put image from pipe


def SHOWMSS_screen(p_output):
    global fps, start_time, coords, classes, percent
    shoot_time = time.time()
    yolo = YOLO()
    while True:
        coords = 0
        percent = 0
        classes = 0
        img = p_output.recv()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r_image, ObjectsList = yolo.detect_image(img)
        f = open("action.txt", "r")
        #print(f.readline().strip())
        object = str(f.readline().strip())
        print(object)
        cv2.imshow("YOLO v3", r_image)
        attack = time.time() - shoot_time
        c = random.uniform(5, 8)
        if float(percent) > 0.8 and attack > c:
            if object != "Cow" and object != "Cou" and object != "Cow calf":
                mid_x = (coords[1] + coords[3]) / 2
                mid_y = ((coords[0] + coords[2]) / 2) + 40
                click_attack(mid_x, mid_y)
                shoot_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
        fps += 1
        TIME = time.time() - start_time
        if (TIME) >= display_time:
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    yolo.close_session()


if __name__ == "__main__":

    x = random.randrange(625, 635)
    y = random.randrange(345, 355)
    pyautogui.click(x, y)
    j = 0
    p_output, p_input = Pipe()
    # creating new processes
    p1 = multiprocessing.Process(target=GRABMSS_screen, args=(p_input,))
    p2 = multiprocessing.Process(target=SHOWMSS_screen, args=(p_output,))
    p3 = Process(target=READ_Text)

    # starting our processes
    p1.start()
    p2.start()
    p3.start()
    p3.join()
