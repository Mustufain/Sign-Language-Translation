import cv2
import imutils
from hand_detection.hand_detection_opencv import HandDetecter
import numpy as np
import requests
import json
import time
import tensorflow as tf
import math
from PIL import Image

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.width = self.video.get(3)  # float # 1280.0
        self.height = self.video.get(4)  # float # 720.0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()

        frame = imutils.resize(frame, width=1000)
        # flip the frame so that it is not the mirror view
        #frame = cv2.flip(frame, 1)

        # clone the frame
        #clone = frame.copy()

        return frame

    def get_fps(self):
        fps = self.video.get(cv2.cv.CV_CAP_PROP_FPS)
        return fps

        #ret, buffer = cv2.imencode('.jpg', frame)
        #frame = buffer.tobytes()
        #return frame

if __name__ == '__main__':

    def identifyGesture(handTrainImage):
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

        # get image as numpy array and predict using model
        open_cv_image = np.array(background)
        background = open_cv_image.astype('float32')
        background = background / 255
        background = background.reshape((1,) + background.shape)
        return background


    camera = VideoCamera()
    # previous values of cropped variable
    x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0
    # previous frame contour of hand. Used to compare with new contour to find if gesture has changed.
    prevcnt = np.array([], dtype=np.int32)

    # gesture static increments when gesture doesn't change till it reaches 10 (frames) and then resets to 0.
    # gesture detected is set to 10 when gesture static reaches 10."Gesture Detected is displayed for next
    # 10 frames till gestureDetected decrements to 0.
    gestureStatic = 0
    gestureDetected = 0

    def nothing(x):
        pass


    from keras.models import load_model

    #new_model = load_model('checkpoint_model/inceptionv3.h5')
    palm_cascade = cv2.CascadeClassifier('hand_detection/palm.xml')


    cv2.namedWindow('VideoFeed')
    cv2.namedWindow('handTrain')
    # TrackBars for fixing skin color of the person
    cv2.createTrackbar('B for min', 'VideoFeed', 0, 255, nothing)
    cv2.createTrackbar('G for min', 'VideoFeed', 0, 255, nothing)
    cv2.createTrackbar('R for min', 'VideoFeed', 0, 255, nothing)
    cv2.createTrackbar('B for max', 'VideoFeed', 0, 255, nothing)
    cv2.createTrackbar('G for max', 'VideoFeed', 0, 255, nothing)
    cv2.createTrackbar('R for max', 'VideoFeed', 0, 255, nothing)

    # Default skin color values in indoor lighting
    cv2.setTrackbarPos('B for min', 'VideoFeed', 0)
    cv2.setTrackbarPos('G for min', 'VideoFeed', 130)
    cv2.setTrackbarPos('R for min', 'VideoFeed', 103)
    cv2.setTrackbarPos('B for max', 'VideoFeed', 255)
    cv2.setTrackbarPos('G for max', 'VideoFeed', 182)
    cv2.setTrackbarPos('R for max', 'VideoFeed', 130)

    while (True):

        # Getting min and max colors for skin
        min_YCrCb = np.array([cv2.getTrackbarPos('B for min', 'VideoFeed'),
                                 cv2.getTrackbarPos('G for min', 'VideoFeed'),
                                 cv2.getTrackbarPos('R for min', 'VideoFeed')], np.uint8)
        max_YCrCb = np.array([cv2.getTrackbarPos('B for max', 'VideoFeed'),
                                 cv2.getTrackbarPos('G for max', 'VideoFeed'),
                                 cv2.getTrackbarPos('R for max', 'VideoFeed')], np.uint8)
        frame = camera.get_frame()
        # Convert image to YCrCb
        imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)
        # Find region with skin tone in YCrCb image
        skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        # Do contour detection on skin region
        contours, _ = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # sorting contours by area. Largest area first.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # get largest contour and compare with largest contour from previous frame.
        # set previous contour to this one after comparison.
        cnt = contours[0]
        ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
        prevcnt = contours[0]
        # once we get contour, extract it without background into a new window called handTrainImage
        stencil = np.zeros(frame.shape).astype(frame.dtype)
        color = [255, 255, 255]
        cv2.fillPoly(stencil, [cnt], color)
        handTrainImage = cv2.bitwise_and(frame, stencil)
        # if comparison returns a high value (shapes are different), start gestureStatic over. Else increment it.
        if (ret > 0.70):
            gestureStatic = 0
        else:
            gestureStatic += 1

        # crop coordinates for hand.
        x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)

        # place a rectange around the hand.
        cv2.rectangle(frame, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)
        # if the crop area has changed drastically form previous frame, update it.
        if (abs(x_crop - x_crop_prev) > 50 or abs(y_crop - y_crop_prev) > 50 or
                abs(w_crop - w_crop_prev) > 50 or abs(h_crop - h_crop_prev) > 50):
            x_crop_prev = x_crop
            y_crop_prev = y_crop
            h_crop_prev = h_crop
            w_crop_prev = w_crop

        # create crop image
        handImage = frame.copy()[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
                    max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]

        # Training image with black background
        handTrainImage = handTrainImage[max(0, y_crop_prev - 15):y_crop_prev + h_crop_prev + 15,
                         max(0, x_crop_prev - 15):x_crop_prev + w_crop_prev + 15]

        # if gesture is static for 10 frames, set gestureDetected to 10 and display "gesture detected"
        # on screen for 10 frames.
        if gestureStatic == 10:
            gestureDetected = 10
            print("Gesture Detected")
            letterDetected =  identifyGesture(handTrainImage)
            letterDetected.save("test_images/a.jpeg")

            #letterDetected = identifyGesture(
                #handTrainImage)  # todo: Ashish fill this function to return actual character

        if gestureDetected > 0:
            if (letterDetected != None):
                cv2.putText(frame, str(letterDetected), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            gestureDetected -= 1
            # haar cascade classifier to detect palm and gestures. Not very accurate though.
            # Needs more training to become accurate.
        gray = cv2.cvtColor(handImage, cv2.COLOR_BGR2HSV)
        palm = palm_cascade.detectMultiScale(gray)
        for (x, y, w, h) in palm:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

        # to show convex hull in the image
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        # counting defects in convex hull. To find center of palm. Center is average of defect points.
        count_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            if count_defects == 0:
                center_of_palm = far
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
            if angle <= 90:
                count_defects += 1
                if count_defects < 5:
                    # cv2.circle(sourceImage, far, 5, [0, 0, 255], -1)
                    center_of_palm = (far[0] + center_of_palm[0]) / 2, (far[1] + center_of_palm[1]) / 2
            cv2.line(frame, start, end, [0, 255, 0], 2)
        # cv2.circle(sourceImage, avr, 10, [255, 255, 255], -1)

        # drawing the largest contour
        cv2.drawContours(frame, contours, 0, (0, 255, 0), 1)
        cv2.imshow('handTrain', handTrainImage)

        cv2.imshow('VideoFeed', frame)
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            cv2.destroyAllWindows()
            camera.__del__()
            break
