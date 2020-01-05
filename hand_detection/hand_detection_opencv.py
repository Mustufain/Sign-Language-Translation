import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
background = None
from PIL import Image


class HandDetecter(object):

    def __init__(self, frame):
        self.x1 = 10
        self.y1 = 10
        self.x2 = 400
        self.y2 = 400
        self.upper_left = (self.x1, self.y1)
        self.bottom_right = (self.x2, self.y2)
        self.image = frame

    def draw_rectangle(self):
        """
        x1,y1 ------
        |          |
        |          |
        |          |
        --------x2,y2

        :return:
        """

        cv2.rectangle(self.image, self.upper_left, self.bottom_right, (0, 255, 0), 2)

    def preprocess(self):
        self.draw_rectangle()
        roi = self.crop_image()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        return gray

    def running_average(self,image, aWeight):
        """
        Background subtraction
        :return:
        """
        global background
        # initialize the background
        if background is None:
            background = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, background, aWeight)

    def segment(self, image, threshold):
        global background
        # find the absolute difference between background and current frame
        diff = cv2.absdiff(background.astype("uint8"), image)

        # threshold the diff image so that we get the foreground
        thresholded = cv2.threshold(diff,
                                    threshold,
                                    255,
                                    cv2.THRESH_BINARY)[1]

        # get the contours in the thresholded image
        cnts, _ = cv2.findContours(thresholded.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        # return None, if no contours detected
        if len(cnts) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)

    def crop_image(self):
        """
        Crop an image inside the rectangle
        :param new_image:
        :return:
        """
        roi = self.image[self.upper_left[1]:self.bottom_right[1], self.upper_left[0]:self.bottom_right[0]]
        return roi

    def identifyGesture(self, handTrainImage):
        # saving the sent image for checking
        # cv2.imwrite("/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/a0.jpeg", handTrainImage)

        # converting the image to same resolution as training data by padding to reach 1:1 aspect ration and then
        # resizing to 400 x 400. Same is done with training data in preprocess_image.py. Opencv image is first
        # converted to Pillow image to do this.
        handTrainImage = cv2.cvtColor(handTrainImage, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(handTrainImage)
        img_w, img_h = img.size
        M = max(img_w, img_h)
        background = Image.new('RGB', (M, M), (0, 0, 0))
        bg_w, bg_h = background.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        background.paste(img, offset)
        size = 400, 400
        background = background.resize(size, Image.ANTIALIAS)
        # background.save('test_images/a.png')
        # get image as numpy array and predict using model
        open_cv_image = np.array(background)
        background = open_cv_image.astype('float32')
        background = background / 255
        background = background.reshape((1,) + background.shape)
        return background







