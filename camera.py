import cv2
import imutils
from hand_detection.hand_detection_opencv import HandDetecter
import numpy as np
import requests
import json
import time
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
        # flip the frame so that it is not the mirror view
        #frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        return clone

    def get_fps(self):
        fps = self.video.get(cv2.cv.CV_CAP_PROP_FPS)
        return fps

        #ret, buffer = cv2.imencode('.jpg', frame)
        #frame = buffer.tobytes()
        #return frame

if __name__ == '__main__':

    aWeight = 0.5
    camera = VideoCamera()
    # previous values of cropped variable
    x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0
    # previous frame contour of hand. Used to compare with new contour to find if gesture has changed.
    prevcnt = np.array([], dtype=np.int32)
    gestureStatic = 0
    gestureDetected = 0
    num_frames = 0


    def nothing(x):
        pass


    from keras.models import load_model

    new_model = load_model('checkpoint_model/inception_v3.h5')
    cv2.namedWindow('VideoFeed')
    cv2.namedWindow('hand')
    #cv2.namedWindow('thresholded')
    cv2.createTrackbar('Threshold value', 'hand', 0, 255, nothing)
    cv2.setTrackbarPos('Threshold value', 'hand', 10)

    while (True):
        frame = camera.get_frame()
        detector = HandDetecter(frame=frame)
        gray = detector.preprocess()

        if num_frames > 5:
            cv2.putText(frame, 'Put your hand into the box', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

        if num_frames < 30:
            detector.running_average(image=gray, aWeight=aWeight)

        else:
            # segment the hand region
            threshold = cv2.getTrackbarPos('Threshold value', 'hand')
            hand = detector.segment(gray, threshold=threshold)
            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand
                # segment is the largest contour by area
                # match this segment with previous segment
                ret = cv2.matchShapes(segmented, prevcnt, 2, 0.0)
                # update previous contour
                prevcnt = segmented
                # if comparison returns a high value (shapes are different), start gestureStatic over. Else increment it.
                if (ret > 0.70):
                    gestureStatic = 0
                else:
                    gestureStatic += 1

                # draw the segmented region and display the frame
                cnt = segmented + (10, 10)
                # cv2.drawContours(frame, [cnt], -1, (0, 0, 255))
                # crop coordinates for hand.
                x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)
                # place a rectange around the hand
                hand = frame[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
                cv2.rectangle(frame, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (255, 0, 0), 2)
                thresholded = cv2.dilate(thresholded, (7, 7), iterations=4)
                thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, (7, 7))
                #cv2.imshow("thresholded", thresholded)
                mask = cv2.merge(mv=[thresholded, thresholded, thresholded])
                roi = detector.crop_image()
                sel = cv2.bitwise_and(src1=roi, src2=mask)
                if gestureStatic == 10:
                    # generate predictions
                    gestureDetected = 10
                    print("Gesture Detected")
                    input_image = detector.identifyGesture(sel)
                    prediction = new_model.predict(input_image)
                    digit = np.argmax(prediction, axis=1)[0]
                    hand_image = cv2.resize(sel,(400, 400))
                    font = cv2.FONT_HERSHEY_SIMPLEX

                if gestureDetected > 0:
                    if (digit != None):
                        cv2.putText(frame, str(digit), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
                    gestureDetected -= 1
                # payload = {"instances": [{'input_image': hand_image.tolist()}]}



                # sending post request to TensorFlow Serving server

                # r = requests.post('http://localhost:8501/v1/models/inception_v3:predict', json=payload)
                # if r.status_code == 404:
                #     print ('Error')
                # else:
                #     pred = json.loads(r.content.decode('utf-8'))
                #     predictions = pred['predictions']
                #     digit = str(np.argmax(predictions))
                #     cv2.putText(frame, digit, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
                #prediction = new_model.predict(input_image)
                #digit = np.argmax(prediction, axis=1)[0]
                #cv2.putText(frame, str(digit), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)


            # draw the segmented hand
        cv2.rectangle(frame, (10, 10), (400, 400), (0, 255, 0), 2)
        num_frames += 1

        cv2.imshow("VideoFeed", frame)
        cv2.imshow('hand', hand_image)
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            cv2.destroyAllWindows()
            camera.__del__()
            break
