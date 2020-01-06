import cv2
import imutils
from detection.hand_detection import HandDetector
import numpy as np
from keras.models import load_model


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.width = self.video.get(3)
        self.height = self.video.get(4)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()
        frame = imutils.resize(frame, width=1000)
        # clone the frame
        clone = frame.copy()

        return clone

    def get_fps(self):
        fps = self.video.get(cv2.cv.CV_CAP_PROP_FPS)
        return fps


class HandTranslator(object):

    def __init__(self):
        self.alpha = 0.5
        self.gesture_static = 0
        self.gesture_detected = 0
        self.num_frames = 0

    def translate(self):
        camera = VideoCamera()
        # previous values of cropped variable
        x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0
        # previous frame contour of hand. Used to compare with new contour to find if gesture has changed.
        prev_contour = np.array([], dtype=np.int32)
        self.gesture_static = 0
        self.gesture_detected = 0
        self.num_frames = 0
        new_model = load_model('checkpoint_model/inception_v3.h5')
        cv2.namedWindow('VideoFeed')
        cv2.namedWindow('hand')
        cv2.createTrackbar('Threshold value', 'hand', 0, 255, self.nothing)
        cv2.setTrackbarPos('Threshold value', 'hand', 10)
        while True:
            frame = camera.get_frame()
            detector = HandDetector(frame=frame)
            gray = detector.preprocess()
            if self.num_frames > 5:
                cv2.putText(frame, 'Put your hand into the box', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

            if self.num_frames < 30:
                detector.running_average(image=gray, alpha=self.alpha)
            else:
                # segment the hand region
                threshold = cv2.getTrackbarPos('Threshold value', 'hand')
                hand = detector.segment(gray, threshold=threshold)
                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    thresholded, segmented = hand
                    # segment is the largest contour by area
                    # match this segment with previous segment
                    ret = cv2.matchShapes(segmented, prev_contour, 2, 0.0)
                    # update previous contour
                    prev_contour = segmented
                    # if comparison returns a high value (shapes are different),
                    # start self.gesture_static over. else increment it.
                    if ret > 0.70:
                        self.gesture_static = 0
                    else:
                        self.gesture_static += 1
                        # draw the segmented region and display the frame
                        cnt = segmented + (10, 10)
                        # crop coordinates for hand.
                        x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)
                        # place a rectangle around the hand
                        hand = frame[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
                        thresholded = cv2.dilate(thresholded, (7, 7), iterations=4)
                        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, (7, 7))
                        mask = cv2.merge(mv=[thresholded, thresholded, thresholded])
                        roi = detector.crop_image()
                        sel = cv2.bitwise_and(src1=roi, src2=mask)
                        if self.gesture_static == 10:
                            # generate predictions
                            self.gesture_detected = 10
                            print("Gesture Detected")
                            input_image = detector.identify_gesture(sel)
                            prediction = new_model.predict(input_image)
                            digit = np.argmax(prediction, axis=1)[0]
                            hand_image = cv2.resize(sel, (400, 400))
                            cv2.imshow('hand', hand_image)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                        if self.gesture_detected > 0:
                            if digit is not None:
                                cv2.putText(frame, str(digit), (10, 400), font, 3, (0, 0, 255), 2)
                            self.gesture_detected -= 1

            # draw the segmented hand
            cv2.rectangle(frame, (10, 10), (400, 400), (0, 255, 0), 2)
            self.num_frames += 1
            cv2.imshow("VideoFeed", frame)
            keypress = cv2.waitKey(1) & 0xFF
            # if the user pressed "q", then stop looping
            if keypress == ord("q"):
                cv2.destroyAllWindows()
                camera.__del__()
                break

    def nothing(self, x):
        pass



