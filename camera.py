import cv2
import imutils


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.width = 500
        self.height = 500

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        frame = imutils.resize(frame, width=self.width, height=self.height)
        # frame is the image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        return frame
